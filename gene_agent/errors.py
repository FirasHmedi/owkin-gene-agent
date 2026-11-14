"""Turn local model server errors into messages a non-technical user can act on."""

from __future__ import annotations

from ollama import ResponseError


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, ConnectionError):
        return (
            "Cannot reach the local model server (Ollama). "
            "With Docker, check that `docker compose up` is still running; "
            "otherwise start Ollama and try again."
        )
    if isinstance(exc, ResponseError):
        if exc.status_code == 404:
            return "The model is not downloaded yet. Restart the app to download it."
        if "does not support tools" in str(exc.error):
            return "The configured model cannot call tools. Choose another one with OLLAMA_MODEL."
        return f"The local model server returned an error: {exc.error}"
    return f"Unexpected error: {exc}"
