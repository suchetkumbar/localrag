"""Small shared HTTP helpers."""

import httpx


def raise_for_status(resp: httpx.Response) -> None:
    """Like ``resp.raise_for_status()`` but keeps the server's error message.

    Ollama reports problems such as "model requires more system memory" only in the
    response body; httpx's default exception drops it.
    """
    if resp.is_error:
        raise httpx.HTTPStatusError(
            f"{resp.status_code} from {resp.request.url}: {resp.text[:500]}",
            request=resp.request,
            response=resp,
        )
