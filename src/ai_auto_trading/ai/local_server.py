from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlparse

from ai_auto_trading.settings import Settings, load_settings


@dataclass(frozen=True)
class LocalModelServerConfig:
    host: str
    port: int
    model_id: str
    model_path: str
    max_tokens: int
    offload_dir: str
    mps_max_memory: str
    cpu_max_memory: str


def default_local_server_config(settings: Settings | None = None) -> LocalModelServerConfig:
    resolved = settings or load_settings()
    endpoint = urlparse(resolved.local_model_endpoint)
    model_path = resolved.local_model_path
    offload_dir = ""
    if model_path:
        model_dir = Path(model_path).expanduser().resolve()
        if model_dir.name == "model" and model_dir.parent.exists():
            offload_dir = str(model_dir.parent / "offload_e2b")
        else:
            offload_dir = str(model_dir.parent / "offload")
    return LocalModelServerConfig(
        host=endpoint.hostname or "127.0.0.1",
        port=endpoint.port or 8080,
        model_id=resolved.local_model_id,
        model_path=model_path,
        max_tokens=resolved.ai_max_tokens,
        offload_dir=offload_dir,
        mps_max_memory=resolved.local_model_mps_max_memory,
        cpu_max_memory=resolved.local_model_cpu_max_memory,
    )


class GemmaLocalRuntime:
    def __init__(self, config: LocalModelServerConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._processor = None
        self._model = None

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None and self._processor is not None:
                return
            from transformers import AutoModelForCausalLM, AutoProcessor

            source = self.config.model_path or self.config.model_id
            if not source:
                raise ValueError("local model path or model id must be configured")

            self._processor = AutoProcessor.from_pretrained(source)
            load_kwargs: dict[str, Any] = {
                "dtype": "auto",
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "offload_state_dict": True,
            }
            if self.config.offload_dir:
                offload_dir = Path(self.config.offload_dir)
                offload_dir.mkdir(parents=True, exist_ok=True)
                load_kwargs["offload_folder"] = str(offload_dir)
            max_memory: dict[str, str] = {}
            if self.config.mps_max_memory:
                max_memory["mps"] = self.config.mps_max_memory
            if self.config.cpu_max_memory:
                max_memory["cpu"] = self.config.cpu_max_memory
            if max_memory:
                load_kwargs["max_memory"] = max_memory
            self._model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)

    def chat_completion(self, *, messages: list[dict[str, Any]], max_tokens: int) -> str:
        with self._generation_lock:
            self.ensure_loaded()
            assert self._processor is not None
            assert self._model is not None
            import torch

            try:
                prompt = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = self._processor(text=prompt, return_tensors="pt")
            target_device = _single_runtime_device(self._model)
            if target_device is not None:
                inputs = {key: value.to(target_device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self._model.generate(**inputs, max_new_tokens=max_tokens)
            input_length = inputs["input_ids"].shape[-1]
            decoded = self._processor.decode(outputs[0][input_length:], skip_special_tokens=False)
            parse_response = getattr(self._processor, "parse_response", None)
            parsed = None
            if callable(parse_response):
                try:
                    parsed = parse_response(decoded)
                except Exception:
                    parsed = None
            return _assistant_text(parsed, decoded)


def _assistant_text(parsed: Any, decoded: str) -> str:
    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()
    if isinstance(parsed, dict):
        for key in ("content", "text", "response"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(parsed, list):
        chunks: list[str] = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                chunks.append(item.strip())
                continue
            if isinstance(item, dict):
                for key in ("content", "text"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        chunks.append(value.strip())
                        break
        if chunks:
            return "\n".join(chunks)
    cleaned = decoded.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    return cleaned


def _single_runtime_device(model: Any) -> str | None:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            normalized = _normalize_device_name(device)
            if normalized and normalized not in {"cpu", "disk", "meta"}:
                return normalized
        return None
    device = getattr(model, "device", None)
    if device is None:
        return None
    return _normalize_device_name(device)


def _normalize_device_name(device: Any) -> str | None:
    if device is None:
        return None
    if isinstance(device, int):
        return f"cuda:{device}"
    normalized = str(device).strip().lower()
    return normalized or None


def _handler(runtime: GemmaLocalRuntime):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _json_response(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._json_response(
                    200,
                    {
                        "status": "ok",
                        "model": runtime.config.model_id,
                        "model_path": runtime.config.model_path,
                    },
                )
                return
            if parsed.path == "/v1/models":
                self._json_response(
                    200,
                    {
                        "data": [
                            {
                                "id": runtime.config.model_id,
                                "object": "model",
                                "owned_by": "local",
                                "model_path": runtime.config.model_path,
                            }
                        ],
                        "object": "list",
                    },
                )
                return
            self._json_response(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/v1/chat/completions":
                self._json_response(404, {"error": {"message": "not found"}})
                return

            try:
                payload = self._read_json()
                messages = payload.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("messages must be a non-empty list")
                requested_model_path = str(payload.get("model_path") or runtime.config.model_path or "")
                if runtime.config.model_path and requested_model_path:
                    expected = str(Path(runtime.config.model_path).expanduser().resolve())
                    actual = str(Path(requested_model_path).expanduser().resolve())
                    if expected != actual:
                        raise ValueError("requested model_path does not match configured server model")
                max_tokens = int(payload.get("max_tokens") or runtime.config.max_tokens)
                content = runtime.chat_completion(messages=messages, max_tokens=max_tokens)
                self._json_response(
                    200,
                    {
                        "id": f"chatcmpl-local-{int(time.time() * 1000)}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": runtime.config.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": content},
                            }
                        ],
                    },
                )
            except Exception as exc:
                self._json_response(500, {"error": {"message": str(exc)}})

    return Handler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a local Gemma model over an OpenAI-compatible API.")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--model-id")
    parser.add_argument("--model-path")
    parser.add_argument("--offload-dir")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="Do not preload the model before accepting requests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    defaults = default_local_server_config()
    config = LocalModelServerConfig(
        host=args.host or defaults.host,
        port=args.port or defaults.port,
        model_id=args.model_id or defaults.model_id,
        model_path=args.model_path or defaults.model_path,
        max_tokens=args.max_tokens or defaults.max_tokens,
        offload_dir=args.offload_dir or defaults.offload_dir,
        mps_max_memory=defaults.mps_max_memory,
        cpu_max_memory=defaults.cpu_max_memory,
    )
    runtime = GemmaLocalRuntime(config)
    if not args.lazy_load:
        runtime.ensure_loaded()
    server = ThreadingHTTPServer((config.host, config.port), _handler(runtime))
    server.daemon_threads = True
    print(
        json.dumps(
            {
                "status": "serving",
                "host": config.host,
                "port": config.port,
                "model_id": config.model_id,
                "model_path": config.model_path,
            },
            sort_keys=True,
        )
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
