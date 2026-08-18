"""OAuth authentication and credential storage for the ChatGPT Codex backend.

This module follows the OAuth shape used by Pi's ``pi-ai`` provider: PKCE browser
login, rotating refresh tokens, and a short-lived access token containing the
ChatGPT account id. The Codex backend is not the public OpenAI API, so this code
is intentionally isolated from the normal OpenAI-compatible provider.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

AUTH_BASE_URL = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE_URL}/oauth/authorize"
TOKEN_URL = f"{AUTH_BASE_URL}/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
# Keep this identical to the redirect URI used by Pi/Codex OAuth clients.
# The callback server itself binds only to loopback.
REDIRECT_URI = "http://localhost:1455/auth/callback"
OAUTH_SCOPE = "openid profile email offline_access"
JWT_AUTH_CLAIM = "https://api.openai.com/auth"
DEFAULT_AUTH_PATH = "~/.openevolve/codex_auth.json"


class CodexAuthError(RuntimeError):
    """Raised when Codex credentials cannot be loaded, refreshed, or created."""


@dataclass
class CodexCredentials:
    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodexCredentials":
        # Accept Pi's field names as an import convenience without sharing the
        # credential file by default.
        return cls(
            access_token=str(data.get("access_token") or data.get("access") or ""),
            refresh_token=str(data.get("refresh_token") or data.get("refresh") or ""),
            expires_at=float(data.get("expires_at", data.get("expires", 0))),
            account_id=str(data.get("account_id") or data.get("accountId") or ""),
        )

    def validate(self) -> None:
        if not self.access_token or not self.refresh_token or not self.account_id:
            raise CodexAuthError("Codex credentials are missing required fields")


def default_auth_path() -> Path:
    return Path(os.environ.get("OPENEVOLVE_CODEX_AUTH_PATH", DEFAULT_AUTH_PATH)).expanduser()


def login_command(path: Path) -> str:
    command = "openevolve-auth login"
    if path != default_auth_path():
        command += f" --auth-path {path}"
    return command


def login_required_message(path: Path, detail: str) -> str:
    return (
        "Codex authentication is required. "
        f"Run `{login_command(path)}` to sign in with your ChatGPT account; "
        "the command opens a browser for OAuth login. "
        f"Details: {detail}"
    )


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


class CodexCredentialStore:
    """Persistent credential store with a process-safe refresh lock."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path).expanduser() if path else default_auth_path()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def load(self) -> CodexCredentials:
        if not self.path.exists():
            raise CodexAuthError(
                login_required_message(self.path, f"credentials not found at {self.path}")
            )
        try:
            credentials = CodexCredentials.from_dict(
                json.loads(self.path.read_text(encoding="utf-8"))
            )
            credentials.validate()
            return credentials
        except (CodexAuthError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise CodexAuthError(
                login_required_message(self.path, f"could not read credentials: {exc}")
            ) from exc

    def save(self, credentials: CodexCredentials) -> None:
        credentials.validate()
        _atomic_write_json(self.path, asdict(credentials))

    def delete(self) -> None:
        for path in (self.path, self.lock_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def locked(self):
        return _FileLock(self.lock_path)


class _FileLock:
    """Small cross-process lock; fcntl is available on supported Unix targets."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows fallback
            _fallback_lock.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is None:
            return
        try:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover - Windows fallback
            _fallback_lock.release()
        self.handle.close()


_fallback_lock = threading.Lock()


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except (IndexError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise CodexAuthError("Codex access token is not a valid JWT") from exc


def account_id_from_token(access_token: str) -> str:
    payload = _decode_jwt_payload(access_token)
    account_id = payload.get(JWT_AUTH_CLAIM, {}).get("chatgpt_account_id")
    if not account_id:
        raise CodexAuthError("Codex access token does not contain a ChatGPT account id")
    return str(account_id)


def _request_form(url: str, values: Dict[str, str], timeout: float = 30) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(values).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CodexAuthError(f"Codex OAuth request failed ({exc.code}): {detail[:500]}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise CodexAuthError(f"Codex OAuth request failed: {exc}") from exc


def _credentials_from_token_response(
    data: Dict[str, Any], old_refresh: Optional[str] = None
) -> CodexCredentials:
    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token") or old_refresh
    expires_in = data.get("expires_in")
    if not access_token or not refresh_token or not isinstance(expires_in, (int, float)):
        raise CodexAuthError("Codex OAuth response did not contain usable token fields")
    return CodexCredentials(
        access_token=str(access_token),
        refresh_token=str(refresh_token),
        expires_at=time.time() + float(expires_in),
        account_id=account_id_from_token(str(access_token)),
    )


class CodexOAuthClient:
    """Browser login and token refresh client."""

    def refresh(self, refresh_token: str) -> CodexCredentials:
        data = _request_form(
            TOKEN_URL,
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
        )
        return _credentials_from_token_response(data, old_refresh=refresh_token)

    def login(self) -> CodexCredentials:
        verifier = _base64url(secrets.token_bytes(32))
        challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
        state = secrets.token_urlsafe(24)
        authorization_url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode({
            'response_type': 'code',
            'client_id': CLIENT_ID,
            'redirect_uri': REDIRECT_URI,
            'scope': OAUTH_SCOPE,
            'code_challenge': challenge,
            'code_challenge_method': 'S256',
            'state': state,
            'id_token_add_organizations': 'true',
            'codex_cli_simplified_flow': 'true',
            'originator': 'openevolve',
        })}"

        callback = _OAuthCallbackServer(state)
        callback.start()
        print("Opening OpenAI login in your browser...")
        print(f"If it does not open, visit:\n{authorization_url}")
        webbrowser.open(authorization_url)
        try:
            code = callback.wait()
        finally:
            callback.close()
        if not code:
            raise CodexAuthError("Codex OAuth login did not return an authorization code")

        data = _request_form(
            TOKEN_URL,
            {
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": REDIRECT_URI,
            },
        )
        return _credentials_from_token_response(data)


class _OAuthCallbackServer:
    def __init__(self, state: str):
        self.state = state
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.code: Optional[str] = None
        self.error: Optional[str] = None
        self.event = threading.Event()

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != "/auth/callback":
                    self.send_error(404)
                    return
                params = urllib.parse.parse_qs(parsed.query)
                if params.get("state", [None])[0] != owner.state:
                    owner.error = "OAuth state mismatch"
                else:
                    owner.code = params.get("code", [None])[0]
                    owner.error = params.get("error", [None])[0]
                body = b"OpenEvolve authentication complete. You can close this window."
                self.send_response(200 if owner.code else 400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                owner.event.set()

            def log_message(self, format, *args):
                return

        try:
            self.server = HTTPServer(("127.0.0.1", 1455), Handler)
        except OSError as exc:
            raise CodexAuthError("Could not bind OAuth callback on 127.0.0.1:1455") from exc
        # Keep the server loop alive until close() calls shutdown(), so the
        # listening socket is released cleanly after the callback request.
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def wait(self, timeout: float = 600) -> str:
        if not self.event.wait(timeout):
            raise CodexAuthError("Timed out waiting for Codex OAuth callback")
        if self.error:
            raise CodexAuthError(self.error)
        return self.code or ""

    def close(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=1)


class CodexAuthManager:
    """Load valid credentials and refresh them safely when needed."""

    def __init__(self, path: Optional[str | Path] = None, oauth: Optional[CodexOAuthClient] = None):
        self.store = CodexCredentialStore(path)
        self.oauth = oauth or CodexOAuthClient()

    def login(self) -> CodexCredentials:
        credentials = self.oauth.login()
        self.store.save(credentials)
        return credentials

    def get_credentials(
        self,
        minimum_validity: float = 300,
        force_refresh: bool = False,
        failed_access_token: Optional[str] = None,
    ) -> CodexCredentials:
        credentials = self.store.load()
        if not force_refresh and credentials.expires_at > time.time() + minimum_validity:
            return credentials

        with self.store.locked():
            # Another process may have refreshed while we waited for the lock.
            credentials = self.store.load()
            if failed_access_token and credentials.access_token != failed_access_token:
                # A concurrent request already rotated the refresh token. Reuse
                # the credentials it persisted instead of rotating again.
                return credentials
            if not force_refresh and credentials.expires_at > time.time() + minimum_validity:
                return credentials
            try:
                refreshed = self.oauth.refresh(credentials.refresh_token)
            except CodexAuthError as exc:
                raise CodexAuthError(
                    login_required_message(self.store.path, f"token refresh failed: {exc}")
                ) from exc
            self.store.save(refreshed)
            return refreshed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manage OpenEvolve Codex OAuth credentials")
    parser.add_argument("command", choices=["login", "status", "logout"])
    parser.add_argument("--auth-path", default=None)
    args = parser.parse_args(argv)
    manager = CodexAuthManager(args.auth_path)
    if args.command == "login":
        credentials = manager.login()
        print(f"Codex login complete for account {credentials.account_id}")
    elif args.command == "status":
        credentials = manager.get_credentials(minimum_validity=0)
        remaining = max(0, int(credentials.expires_at - time.time()))
        print(
            f"Codex credentials are available for account {credentials.account_id} ({remaining}s remaining)"
        )
    else:
        manager.store.delete()
        print(f"Removed Codex credentials from {manager.store.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
