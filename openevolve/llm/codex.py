"""Native direct-call provider for ChatGPT subscription-backed Codex models."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from openevolve.llm.base import LLMInterface
from openevolve.llm.codex_auth import CodexAuthError, CodexAuthManager

logger = logging.getLogger(__name__)

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"


class CodexHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"Codex backend request failed ({status}): {detail[:500]}")
        self.status = status
        self.detail = detail


def _response_url(base_url: str) -> str:
    normalized = (base_url or DEFAULT_CODEX_BASE_URL).rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _sse_text(response, timeout: float) -> str:
    """Read a streamed Codex response and accumulate output text."""
    chunks: List[str] = []
    while True:
        line = response.readline()
        if not line:
            break
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("Ignoring non-JSON Codex SSE event")
            continue

        event_type = event.get("type", "")
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            chunks.append(str(event.get("delta", "")))
        elif event_type in {"response.failed", "error"}:
            error = event.get("error") or event.get("response", {}).get("error") or event
            raise RuntimeError(f"Codex response failed: {error}")

    result = "".join(chunks)
    if not result:
        raise RuntimeError("Codex returned an empty text response")
    return result


def _post_sse(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _sse_text(response, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CodexHTTPError(exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise TimeoutError(f"Codex network request failed: {exc.reason}") from exc


class CodexLLM(LLMInterface):
    """LLM backend using Codex OAuth and direct HTTPS/SSE calls."""

    def __init__(self, model_cfg=None):
        self.model = getattr(model_cfg, "name", None) or "gpt-5.6-luna"
        # Codex subscription auth is OAuth-only. Deliberately discard any
        # inherited api_key so OPENAI_API_KEY cannot be used accidentally by
        # this provider, especially when configs are assembled from shared LLM
        # settings.
        self.api_key = None
        self.system_message = getattr(model_cfg, "system_message", None) or ""
        self.max_tokens = getattr(model_cfg, "max_tokens", None)
        self.timeout = getattr(model_cfg, "timeout", None) or 300
        self.retries = getattr(model_cfg, "retries", None)
        self.retries = 3 if self.retries is None else self.retries
        self.retry_delay = getattr(model_cfg, "retry_delay", None)
        self.retry_delay = 5 if self.retry_delay is None else self.retry_delay
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        configured_base = getattr(model_cfg, "api_base", None)
        # LLMConfig's general default is the public OpenAI API. Codex OAuth
        # requires the ChatGPT Codex backend unless the user explicitly sets a
        # Codex-compatible base URL.
        if not configured_base or configured_base == "https://api.openai.com/v1":
            configured_base = DEFAULT_CODEX_BASE_URL
        self.api_base = configured_base
        auth_path = getattr(model_cfg, "codex_auth_path", None)
        self.auth = CodexAuthManager(auth_path)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=kwargs.pop("system_message", self.system_message),
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "store": False,
            "stream": True,
            "instructions": system_message or self.system_message or "You are a helpful assistant.",
            "input": [
                {"role": message.get("role", "user"), "content": message.get("content", "")}
                for message in messages
            ],
            "text": {"verbosity": kwargs.get("verbosity", "low")},
        }
        reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if reasoning_effort:
            payload["reasoning"] = {
                "effort": reasoning_effort,
                "summary": kwargs.get("reasoning_summary", "auto"),
            }
        # The subscription-backed Codex endpoint currently rejects the public
        # Responses API's max_output_tokens field. Keep max_tokens in the
        # provider config for interface compatibility, but do not transmit it.

        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        force_refresh = False
        failed_access_token = None

        attempt = 0
        max_attempts = retries + 1
        while attempt < max_attempts:
            try:
                credentials = await asyncio.to_thread(
                    self.auth.get_credentials,
                    300,
                    force_refresh,
                    failed_access_token,
                )
                headers = {
                    "Authorization": f"Bearer {credentials.access_token}",
                    "chatgpt-account-id": credentials.account_id,
                    "originator": "openevolve",
                    "OpenAI-Beta": "responses=experimental",
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                }
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        _post_sse,
                        _response_url(self.api_base),
                        headers,
                        payload,
                        timeout,
                    ),
                    timeout=timeout + 5,
                )
            except CodexHTTPError as exc:
                if exc.status == 401 and not force_refresh:
                    force_refresh = True
                    failed_access_token = credentials.access_token
                    # A refresh retry is mandatory even when the configured
                    # normal retry count is zero.
                    max_attempts += 1
                    attempt += 1
                    continue
                if 400 <= exc.status < 500:
                    raise
                error: Exception = exc
            except (CodexAuthError, asyncio.TimeoutError, TimeoutError, RuntimeError) as exc:
                error = exc

            if attempt < max_attempts - 1:
                logger.warning(
                    "Codex request failed on attempt %s/%s: %s; retrying",
                    attempt + 1,
                    max_attempts,
                    error,
                )
                await asyncio.sleep(retry_delay)
            else:
                raise error
            attempt += 1


def init_codex_client(model_cfg):
    """Factory compatible with ``LLMModelConfig.init_client``."""
    return CodexLLM(model_cfg)
