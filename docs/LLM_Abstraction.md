# LLM Abstraction

OpenEvolve exposes a small asynchronous interface for making LLM calls and routes requests to one or more configured backends.

## Request flow

```text
Prompt builder / worker
        |
        v
LLMEnsemble
        |
        v
Configured provider backend
        |
        +--> OpenAILLM       -> OpenAI-compatible Chat Completions API
        |
        +--> ClaudeCodeLLM   -> local `claude` CLI
```

The main implementation files are:

- `openevolve/llm/base.py` — provider-independent interface
- `openevolve/llm/ensemble.py` — model selection and provider routing
- `openevolve/llm/openai.py` — OpenAI-compatible API implementation
- `openevolve/llm/claude_code.py` — Claude Code CLI implementation
- `openevolve/llm/codex.py` — direct ChatGPT Codex OAuth/Responses implementation
- `openevolve/llm/codex_auth.py` — OAuth login, refresh, and credential storage
- `openevolve/config.py` — model, endpoint, and credential configuration
- `openevolve/process_parallel.py` — main evolution call site

## Core interface

`LLMInterface` defines two asynchronous methods:

```python
async def generate(prompt: str, **kwargs) -> str

async def generate_with_context(
    system_message: str,
    messages: list[dict[str, str]],
    **kwargs,
) -> str
```

`generate()` is a convenience method. Implementations normally convert it into a call to `generate_with_context()` with one user message.

## Provider routing

`LLMEnsemble` creates one backend per `LLMModelConfig`:

1. If `model_cfg.init_client` is set, that custom factory is used.
2. Otherwise, `provider` is looked up in the provider registry.
3. If no provider is specified, the model defaults to `OpenAILLM`.

The built-in providers are:

```python
"openai"       -> OpenAILLM
"claude_code" -> ClaudeCodeLLM
"codex"       -> CodexLLM
```

Multiple configured models are sampled using their normalized `weight` values. Evaluator models are configured separately through `llm.evaluator_models`; if omitted, they inherit the evolution models.

## OpenAI-compatible API backend

`OpenAILLM` constructs an OpenAI Python client using:

```python
openai.OpenAI(
    api_key=model_cfg.api_key,
    base_url=model_cfg.api_base,
    timeout=model_cfg.timeout,
    max_retries=model_cfg.retries,
)
```

The request is sent through:

```python
client.chat.completions.create(**params)
```

Because the endpoint is configurable, this backend supports OpenAI and other services exposing an OpenAI-compatible API, including Gemini’s compatibility endpoint, OpenRouter, local Ollama/vLLM servers, and proxies such as OptiLLM.

Generation parameters such as `temperature`, `top_p`, `max_tokens`, `reasoning_effort`, and `seed` are taken from the model configuration and may be overridden per call. Calls are executed asynchronously and retried after errors or timeouts.

## Authentication

### API-key providers

Credentials are stored in `LLMModelConfig.api_key` and passed to the OpenAI client. They can be supplied directly:

```python
LLMModelConfig(
    name="gpt-4o",
    api_key="sk-...",
)
```

The preferred YAML form is environment-variable substitution:

```yaml
llm:
  api_base: "https://api.openai.com/v1"
  api_key: ${OPENAI_API_KEY}
  models:
    - name: "gpt-4o"
      weight: 1.0
```

The `${VAR}` syntax is resolved by `openevolve.config._resolve_env_var`. If the referenced variable is missing, configuration loading raises an error.

When using `load_config()` without a config file, OpenEvolve also reads:

```text
OPENAI_API_KEY
OPENAI_API_BASE
```

Top-level `llm` values are propagated to each model unless that model explicitly defines its own value. This includes `api_key`, `api_base`, and `provider`.

When `provider: codex` is selected, the Codex backend deliberately ignores
`api_key` and `OPENAI_API_KEY`; it authenticates only with its ChatGPT OAuth
credential store.

### Claude Code

`ClaudeCodeLLM` does not make a direct API request. It invokes the locally installed Claude Code CLI:

```text
claude -p --model <model> --output-format text ...
```

Authentication is handled by the CLI’s local OAuth session rather than by `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`:

```bash
npm install -g @anthropic-ai/claude-code
claude login
```

Configuration:

```yaml
llm:
  provider: "claude_code"
  models:
    - name: "sonnet"
      weight: 1.0
      timeout: 300
```

### ChatGPT subscription-backed Codex

The `codex` provider uses a native OAuth flow and direct HTTPS/SSE calls to the
ChatGPT Codex backend. It does not invoke the Codex CLI.

Authenticate once:

```bash
openevolve-auth login
```

Then configure OpenEvolve:

```yaml
llm:
  provider: codex
  models:
    - name: gpt-5.6-luna
      weight: 1.0
      timeout: 300
```

Credentials are stored by default at `~/.openevolve/codex_auth.json`. Override
that location with `OPENEVOLVE_CODEX_AUTH_PATH` or `llm.codex_auth_path`.
The provider refreshes access tokens automatically and uses a file lock to
serialize refreshes across worker processes.

## Main evolution call

The parallel worker builds a system prompt and user message, then calls:

```python
await ensemble.generate_with_context(
    system_message=prompt["system"],
    messages=[{"role": "user", "content": prompt["user"]}],
)
```

The selected backend returns plain text. The worker then parses that text as either a diff or a full rewritten program, depending on `diff_based_evolution`.
