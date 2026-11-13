"""Turn Anthropic SDK exceptions into messages a non-technical user can act on."""

from __future__ import annotations

import anthropic


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, anthropic.AuthenticationError):
        return "The Anthropic API key was rejected. Check the key and try again."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "This API key does not have access to the configured model."
    if isinstance(exc, anthropic.NotFoundError):
        return "The configured model was not found. Check ANTHROPIC_MODEL."
    if isinstance(exc, anthropic.RateLimitError):
        return "The model is rate limited right now. Wait a moment and try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API. Check the internet connection."
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code >= 500:
            return "The model service is temporarily unavailable. Try again shortly."
        return f"The request was rejected by the API: {exc.message}"
    if isinstance(exc, anthropic.AnthropicError) or "authentication method" in str(exc):
        # The SDK raises a plain TypeError on the first request when it finds no credentials.
        return "No Anthropic API key is configured. Set ANTHROPIC_API_KEY (see .env.example) and try again."
    return f"Unexpected error: {exc}"
