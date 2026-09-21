"""Async Ollama client.

The two failure modes here are silent, which is why they get explicit guards:
  * a thinking-capable model left in thinking mode consumes the entire
    num_predict budget on reasoning and returns an empty `response`;
  * an unset num_ctx defaults to 4096 and truncates long source text without
    any error.
"""
import re
from dataclasses import dataclass

import httpx


class LLMError(Exception):
    pass


class ModelNotFound(LLMError):
    pass


def coerce_keep_alive(value: str):
    """Ollama wants a bare integer (-1 = pin forever) or a duration string.

    Sending the *string* "-1" is rejected with
    `time: missing unit in duration "-1"`, so integer-looking values must be
    converted while "5m"/"30s" pass through untouched.
    """
    v = str(value).strip()
    return int(v) if re.fullmatch(r"[+-]?\d+", v) else v


@dataclass
class Generation:
    text: str
    prompt_tokens: int | None
    output_tokens: int | None
    prefill_tok_s: float | None
    decode_tok_s: float | None
    duration_s: float
    passed: int


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 900.0,
        keep_alive: str = "-1",
        temperature: float = 0.2,
        think: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = coerce_keep_alive(keep_alive)
        self.temperature = temperature
        self.think = think

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def generate(
        self, prompt: str, num_ctx: int, num_predict: int, passed: int = 1
    ) -> Generation:
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": self.temperature,
            },
        }
        # Only send `think` for models that understand it; older Ollama builds
        # reject the field on non-thinking models.
        if self.think is False:
            payload["think"] = False

        async with self._client() as c:
            r = await c.post("/api/generate", json=payload)

        if r.status_code == 404:
            raise ModelNotFound(
                f"model {self.model!r} is not pulled on the Ollama host "
                f"(run: ollama pull {self.model})"
            )
        if r.status_code != 200:
            raise LLMError(f"ollama HTTP {r.status_code}: {r.text[:400]}")

        try:
            d = r.json()
        except ValueError as exc:
            raise LLMError(f"ollama returned non-JSON body: {exc}") from exc

        if d.get("error"):
            raise LLMError(f"ollama error: {d['error']}")

        text = (d.get("response") or "").strip()
        if not text:
            raise LLMError(
                "model returned an empty response; it most likely spent the whole "
                "token budget thinking. Raise num_predict or confirm think=false "
                "is honoured by this model."
            )

        pe, ped = d.get("prompt_eval_count"), d.get("prompt_eval_duration") or 0
        ec, ed = d.get("eval_count"), d.get("eval_duration") or 0
        return Generation(
            text=text,
            prompt_tokens=pe,
            output_tokens=ec,
            prefill_tok_s=round(pe / ped * 1e9, 1) if pe and ped else None,
            decode_tok_s=round(ec / ed * 1e9, 1) if ec and ed else None,
            duration_s=round((d.get("total_duration") or 0) / 1e9, 3),
            passed=passed,
        )

    async def healthcheck(self) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=15.0) as c:
                r = await c.get("/api/tags")
                if r.status_code != 200:
                    return False, f"HTTP {r.status_code}"
                names = [m.get("name", "") for m in r.json().get("models", [])]
                if not any(n == self.model or n.startswith(f"{self.model}:") for n in names):
                    return False, f"model {self.model!r} not present; available: {names}"
                return True, f"model {self.model!r} available ({len(names)} pulled)"
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim in /healthz
            return False, f"{type(exc).__name__}: {exc}"
