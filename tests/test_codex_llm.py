"""Unit tests for the native ChatGPT Codex provider."""

import asyncio
import base64
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openevolve.config import Config
from openevolve.llm.codex import CodexLLM, _response_url, _sse_text
from openevolve.llm.codex_auth import (
    CodexAuthError,
    CodexAuthManager,
    CodexCredentials,
    CodexCredentialStore,
    CodexOAuthClient,
    _credentials_from_token_response,
)
from openevolve.llm.ensemble import _PROVIDER_REGISTRY, _create_model


def _jwt(account_id="acct_test", marker=""):
    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"header.{encoded}.signature{marker}"


class TestCodexAuth(unittest.TestCase):
    def test_missing_credentials_explain_how_to_login(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            with self.assertRaises(CodexAuthError) as context:
                CodexAuthManager(path).get_credentials()
        message = str(context.exception)
        self.assertIn("openevolve-auth login", message)
        self.assertIn("opens a browser", message)
        self.assertIn("credentials not found", message)

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

    def test_malformed_credentials_explain_how_to_login(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            path.write_text(json.dumps({"access_token": "only-access"}))
            with self.assertRaises(CodexAuthError) as context:
                CodexCredentialStore(path).load()
        self.assertIn("openevolve-auth login", str(context.exception))
        self.assertIn("missing required fields", str(context.exception))

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

    def test_refresh_failure_explains_how_to_reauthenticate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            CodexCredentialStore(path).save(
                CodexCredentials(_jwt(), "old-refresh", time.time() - 1, "acct_test")
            )
            oauth = MagicMock()
            oauth.refresh.side_effect = CodexAuthError("invalid refresh token")
            with self.assertRaises(CodexAuthError) as context:
                CodexAuthManager(path, oauth=oauth).get_credentials()
        self.assertIn("openevolve-auth login", str(context.exception))
        self.assertIn("token refresh failed", str(context.exception))

    def test_concurrent_forced_refresh_reuses_rotated_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            store = CodexCredentialStore(path)
            store.save(CodexCredentials(_jwt(), "old-refresh", time.time() + 3600, "acct_test"))
            refreshed = CodexCredentials(
                _jwt("acct_test", marker="-new"),
                "new-refresh",
                time.time() + 3600,
                "acct_test",
            )
            oauth = MagicMock()

            def refresh(refresh_token):
                time.sleep(0.05)
                self.assertEqual(refresh_token, "old-refresh")
                return refreshed

            oauth.refresh.side_effect = refresh
            barrier = threading.Barrier(2)
            results = []
            errors = []

            def worker():
                try:
                    barrier.wait()
                    results.append(
                        CodexAuthManager(path, oauth=oauth).get_credentials(
                            force_refresh=True, failed_access_token=_jwt()
                        )
                    )
                except Exception as exc:  # pragma: no cover - failure assertion below
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertTrue(all(item.refresh_token == "new-refresh" for item in results))
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

    @patch("openevolve.llm.codex_auth.webbrowser.open")
    @patch("openevolve.llm.codex_auth._OAuthCallbackServer")
    def test_oauth_callback_is_closed_when_wait_fails(self, callback_cls, _open):
        callback = callback_cls.return_value
        callback.wait.side_effect = CodexAuthError("callback timed out")
        oauth = CodexOAuthClient()
        with self.assertRaises(CodexAuthError):
            oauth.login()
        callback.start.assert_called_once_with()
        callback.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
