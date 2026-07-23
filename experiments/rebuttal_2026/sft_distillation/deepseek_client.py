#!/usr/bin/env python3
"""Minimal DeepSeek JSON-output client with safe caching and retry."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .protocol import canonical_json


DEFAULT_BASE_URL = "https://api.deepseek.com"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _secret_safe(text: str, secret: str) -> str:
    cleaned = str(text)
    if secret:
        cleaned = cleaned.replace(secret, "<redacted>")
    return cleaned[:2000]


@dataclass(frozen=True)
class Pricing:
    input_cache_hit_usd_per_million: Optional[float] = None
    input_cache_miss_usd_per_million: Optional[float] = None
    output_usd_per_million: Optional[float] = None

    @classmethod
    def from_environment(cls) -> "Pricing":
        def optional_float(name: str) -> Optional[float]:
            value = os.environ.get(name)
            if value is None or not value.strip():
                return None
            parsed = float(value)
            if parsed < 0:
                raise ValueError(f"{name} must be non-negative")
            return parsed

        return cls(
            input_cache_hit_usd_per_million=optional_float(
                "DEEPSEEK_INPUT_CACHE_HIT_USD_PER_MILLION"
            ),
            input_cache_miss_usd_per_million=optional_float(
                "DEEPSEEK_INPUT_CACHE_MISS_USD_PER_MILLION"
            ),
            output_usd_per_million=optional_float(
                "DEEPSEEK_OUTPUT_USD_PER_MILLION"
            ),
        )

    def estimate(self, usage: Mapping[str, Any]) -> Optional[float]:
        rates = (
            self.input_cache_hit_usd_per_million,
            self.input_cache_miss_usd_per_million,
            self.output_usd_per_million,
        )
        if any(rate is None for rate in rates):
            return None
        hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        if hit + miss == 0:
            miss = prompt
        completion = int(usage.get("completion_tokens", 0) or 0)
        return (
            hit * float(self.input_cache_hit_usd_per_million)
            + miss * float(self.input_cache_miss_usd_per_million)
            + completion * float(self.output_usd_per_million)
        ) / 1_000_000.0

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "input_cache_hit_usd_per_million": (
                self.input_cache_hit_usd_per_million
            ),
            "input_cache_miss_usd_per_million": (
                self.input_cache_miss_usd_per_million
            ),
            "output_usd_per_million": self.output_usd_per_million,
        }


@dataclass(frozen=True)
class CompletionResult:
    content: Dict[str, Any]
    response_id: str
    response_model: str
    system_fingerprint: str
    finish_reason: str
    usage: Dict[str, int]
    request_sha256: str
    response_sha256: str
    cache_hit: bool
    created_at: str


class DeepSeekClient:
    """OpenAI-compatible DeepSeek client without logging credentials."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        timeout_seconds: float = 600.0,
        transport_retries: int = 5,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = os.environ.get("DEEPSEEK_MODEL", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set in the current process environment"
            )
        if not self.model:
            raise RuntimeError(
                "DEEPSEEK_MODEL is not set in the current process environment"
            )
        self.base_url = (
            base_url
            or os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.transport_retries = int(transport_retries)
        if self.transport_retries < 0:
            raise ValueError("transport_retries must be non-negative")

    def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        body = (
            None
            if payload is None
            else canonical_json(payload).encode("utf-8")
        )
        request = Request(
            self.base_url + endpoint,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "evq-sft-distill/1.0",
            },
        )
        retryable_status = {408, 409, 429, 500, 502, 503, 504}
        last_error: Optional[BaseException] = None
        for attempt in range(self.transport_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                parsed = json.loads(raw.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise RuntimeError("DeepSeek response is not a JSON object")
                return parsed
            except HTTPError as error:
                last_error = error
                try:
                    detail = error.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                if error.code not in retryable_status or attempt >= self.transport_retries:
                    raise RuntimeError(
                        f"DeepSeek HTTP {error.code}: "
                        f"{_secret_safe(detail, self.api_key)}"
                    ) from None
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                last_error = error
                if attempt >= self.transport_retries:
                    raise RuntimeError(
                        "DeepSeek transport/JSON failure: "
                        + _secret_safe(str(error), self.api_key)
                    ) from None
            delay = min(60.0, 1.5 * (2 ** attempt))
            delay *= 0.85 + random.random() * 0.30
            time.sleep(delay)
        raise RuntimeError(
            "DeepSeek request failed: "
            + _secret_safe(str(last_error), self.api_key)
        )

    def list_models(self) -> Sequence[str]:
        payload = self._request_json("GET", "/models")
        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("DeepSeek /models response has no data list")
        models = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(item["id"])
        return tuple(sorted(set(models)))

    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float = 0.7,
        cache_namespace: str = "generation",
    ) -> CompletionResult:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "user_id": "evq-rebuttal-sft-generator",
        }
        request_sha = hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / cache_namespace / f"{request_sha}.json"
        cache_hit = cache_path.is_file()
        if cache_hit:
            response = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            response = self._request_json(
                "POST", "/chat/completions", payload=payload
            )
            write_json_atomic(cache_path, response)
        response_bytes = canonical_json(response).encode("utf-8")
        response_sha = hashlib.sha256(response_bytes).hexdigest()
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("DeepSeek response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("DeepSeek choice is not an object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("DeepSeek choice has no message")
        content_text = message.get("content")
        if not isinstance(content_text, str) or not content_text.strip():
            raise RuntimeError("DeepSeek returned empty JSON content")
        try:
            content = json.loads(content_text)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"DeepSeek content is invalid JSON: {error}") from None
        if not isinstance(content, dict):
            raise RuntimeError("DeepSeek JSON content is not an object")
        usage_raw = response.get("usage")
        usage: Dict[str, int] = {}
        if isinstance(usage_raw, dict):
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            ):
                usage[key] = int(usage_raw.get(key, 0) or 0)
        return CompletionResult(
            content=content,
            response_id=str(response.get("id", "")),
            response_model=str(response.get("model", "")),
            system_fingerprint=str(response.get("system_fingerprint", "")),
            finish_reason=str(choice.get("finish_reason", "")),
            usage=usage,
            request_sha256=request_sha,
            response_sha256=response_sha,
            cache_hit=cache_hit,
            created_at=utc_now(),
        )

    def verify(self) -> Dict[str, Any]:
        models = self.list_models()
        if self.model not in models:
            raise RuntimeError(
                "DEEPSEEK_MODEL is not present in the authenticated /models list"
            )
        messages = [
            {
                "role": "system",
                "content": "Return valid json with exactly one boolean field named ok.",
            },
            {
                "role": "user",
                "content": 'Return this json object: {"ok": true}',
            },
        ]
        result = self.complete_json(
            messages,
            max_tokens=64,
            temperature=0.0,
            cache_namespace="verification",
        )
        if result.content != {"ok": True}:
            raise RuntimeError("DeepSeek JSON verification returned an unexpected object")
        if result.response_model and result.response_model != self.model:
            raise RuntimeError(
                "DeepSeek response model does not match DEEPSEEK_MODEL"
            )
        return {
            "status": "PASS",
            "verified_at": utc_now(),
            "base_url": self.base_url,
            "model": self.model,
            "model_list_contains_requested_model": True,
            "chat_json_output": True,
            "response_model": result.response_model,
            "system_fingerprint": result.system_fingerprint,
            "usage": result.usage,
            "request_sha256": result.request_sha256,
            "response_sha256": result.response_sha256,
            "cache_hit": result.cache_hit,
        }
