"""Local OpenAI-compatible streaming server for replay timing and protocol tests."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from aiohttp import web

from agentkv.hints import validate_envelope


class FakeServer:
    def __init__(self, service_time_s=0.5, observation_dir=None, fail_ids=()):
        if service_time_s <= 0:
            raise ValueError("Service time must be positive")
        self.service_time_s = service_time_s
        self.observation_dir = observation_dir
        self.fail_ids = set(fail_ids)
        self.requests, self.active, self.peak = [], 0, 0
        self.app = web.Application(client_max_size=8 * 1024**2)
        self.app.router.add_post("/v1/completions", self.complete)
        self.app.router.add_get("/health", self.health)
        self.app.router.add_get("/metrics", self.metrics)

    async def health(self, request):
        return web.json_response({"status":"ok"})

    async def metrics(self, request):
        return web.Response(text=f"vllm:num_requests_running {self.active}\nvllm:num_requests_waiting 0\n")

    def record(self, event, **fields):
        if self.observation_dir:
            self.observation_dir.mkdir(parents=True, exist_ok=True)
            with (self.observation_dir / "requests-fake.jsonl").open("a") as sink:
                sink.write(json.dumps(dict(event=event, ns=time.monotonic_ns(), **fields)) + "\n")

    async def complete(self, request):
        body = await request.json()
        if (not body.get("stream") or not body.get("ignore_eos")
                or not isinstance(body.get("prompt"), list) or body.get("max_tokens", 0) < 1):
            return web.json_response({"error":"Invalid completion request"}, status=400)
        idx = body["request_id"]
        envelope = body.get("kv_transfer_params", {}).get("kv_hints")
        if envelope is not None:
            validate_envelope(envelope)
            self.record("hint_received", request_id=idx, message_id=envelope["message_id"])
        body["received_ns"] = time.monotonic_ns()
        self.requests.append(body)
        if idx in self.fail_ids:
            return web.json_response({"error":"Injected failure"}, status=500)
        self.active += 1
        self.peak = max(self.peak, self.active)
        server_id = "cmpl-" + idx
        self.record("prefill", request_id=server_id, prompt_tokens=len(body["prompt"]),
                    gpu_hit=0, cpu_hit=0, recomputed=len(body["prompt"]))
        response = web.StreamResponse(headers={"Content-Type":"text/event-stream"})
        await response.prepare(request)
        # Observation logging and header preparation are part of service time.
        origin = body["received_ns"] / 1e9
        count = body["max_tokens"]
        try:
            for i in range(count):
                await asyncio.sleep(max(0, origin + self.service_time_s * (i + 1) / count - time.monotonic()))
                chunk = dict(id=server_id, object="text_completion", choices=[dict(
                    index=0, text="x", token_ids=[100 + i], finish_reason="length" if i == count-1 else None)])
                await response.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            usage = dict(id=server_id, choices=[], usage=dict(prompt_tokens=len(body["prompt"]), completion_tokens=count))
            await response.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
        finally:
            self.active -= 1
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--service-time", type=float, default=0.5)
    parser.add_argument("--observation-dir", type=Path)
    args = parser.parse_args()
    web.run_app(FakeServer(args.service_time, args.observation_dir).app,
                host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
