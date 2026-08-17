"""Unit tests for the native ChatGPT Codex provider."""

import asyncio
import base64
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openevolve.config import Config
from openevolve.llm.codex import CodexLLM, _response_url, _sse_text
from openevolve.llm.codex_auth import (
    CodexAuthManager,
    CodexCredentials,
    CodexCredentialStore,
    _credentials_from_token_response,
)
from openevolve.llm.ensemble import _PROVIDER_REGISTRY, _create_model


def _jwt(account_id="acct_test"):
    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"header.{encoded}.signature"


class TestCodexAuth(unittest.TestCase):
    def test_token_response_extracts_account_id(self):
        credentials = _credentials_from_token_response(
            {"access_token": _jwt(), "refresh_token": "refresh", "expires_in": 3600}
        )
        self.assertEqual(credentials.account_id, "acct_test")
        self.assertGreater(credentials.expires_at, time.time())

    def test_store_accepts_pi_field_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            path.write_text(
                json.dumps(
                    {
                        "access": _jwt("acct_pi"),
                        "refresh": "refresh",
                        "expires": time.time() + 3600,
                        "accountId": "acct_pi",
                    }
                )
            )
            credentials = CodexCredentialStore(path).load()
            self.assertEqual(credentials.account_id, "acct_pi")

    def test_manager_refreshes_expired_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            store = CodexCredentialStore(path)
            store.save(CodexCredentials(_jwt(), "old-refresh", time.time() - 1, "acct_test"))
            oauth = MagicMock()
            oauth.refresh.return_value = CodexCredentials(
                _jwt(), "new-refresh", time.time() + 3600, "acct_test"
            )
            credentials = CodexAuthManager(path, oauth=oauth).get_credentials()
            self.assertEqual(credentials.refresh_token, "new-refresh")
            oauth.refresh.assert_called_once_with("old-refresh")


class TestCodexTransport(unittest.TestCase):
    def _cfg(self, path):
        return SimpleNamespace(
            name="gpt-5.6-luna",
            system_message="system",
            max_tokens=128,
            timeout=10,
            retries=0,
            retry_delay=0,
            reasoning_effort="medium",
            api_base="https://api.openai.com/v1",
            codex_auth_path=str(path),
        )

    def test_codex_base_url(self):
        self.assertEqual(
            _response_url("https://chatgpt.com/backend-api"),
            "https://chatgpt.com/backend-api/codex/responses",
        )

    def test_sse_parser_accumulates_text_deltas(self):
        stream = io.BytesIO(
            b'data: {"type":"response.output_text.delta","delta":"hello "}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"world"}\n\n'
            b'data: {"type":"response.completed"}\n\n'
        )
        self.assertEqual(_sse_text(stream, timeout=10), "hello world")
        self.assertEqual(
            _response_url("https://chatgpt.com/backend-api/codex"),
            "https://chatgpt.com/backend-api/codex/responses",
        )

    @patch("openevolve.llm.codex._post_sse")
    def test_generate_uses_direct_transport(self, post):
        post.return_value = "generated text"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            CodexCredentialStore(path).save(
                CodexCredentials(_jwt(), "refresh", time.time() + 3600, "acct_test")
            )
            result = asyncio.run(CodexLLM(self._cfg(path)).generate("hello"))
        self.assertEqual(result, "generated text")
        url, headers, payload, _timeout = post.call_args.args
        self.assertTrue(url.endswith("/codex/responses"))
        self.assertEqual(headers["chatgpt-account-id"], "acct_test")
        self.assertEqual(payload["input"][0]["content"], "hello")
        self.assertEqual(payload["reasoning"]["effort"], "medium")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "should-not-be-used"})
    def test_openai_api_key_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            llm = CodexLLM(self._cfg(Path(directory) / "auth.json"))
        self.assertIsNone(llm.api_key)

    def test_provider_is_registered(self):
        self.assertIn("codex", _PROVIDER_REGISTRY)
        cfg = self._cfg(Path("/tmp/nonexistent-codex-auth.json"))
        cfg.init_client = None
        cfg.provider = "codex"
        self.assertIsInstance(_create_model(cfg), CodexLLM)

    def test_config_propagates_auth_path(self):
        config = Config.from_dict(
            {
                "llm": {
                    "provider": "codex",
                    "codex_auth_path": "/tmp/codex-auth.json",
                    "models": [{"name": "gpt-5.6-luna"}],
                }
            }
        )
        self.assertEqual(config.llm.models[0].codex_auth_path, "/tmp/codex-auth.json")


if __name__ == "__main__":
    unittest.main()
