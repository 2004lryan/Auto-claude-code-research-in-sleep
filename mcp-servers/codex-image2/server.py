#!/usr/bin/env python3
"""Codex app-server image bridge for Claude-first ARIS workflows.

This server exposes a narrow MCP interface that asks the local Codex desktop
app to generate an image through the app-server path. It is intentionally kept
small and dependency-free so users can copy it into `~/.claude/mcp-servers/`
and register it with Claude Code.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.stdout = os.fdopen(sys.stdout.fileno(), "wb", buffering=0)
sys.stdin = os.fdopen(sys.stdin.fileno(), "rb", buffering=0)

SERVER_NAME = os.environ.get("CODEX_IMAGE2_SERVER_NAME", "codex-image2")
CODEX_BIN = os.environ.get("CODEX_IMAGE2_CODEX_BIN", "codex")
DEFAULT_TIMEOUT_SEC = int(os.environ.get("CODEX_IMAGE2_TIMEOUT_SEC", "600"))
DEFAULT_MODEL = os.environ.get("CODEX_IMAGE2_MODEL", "")
DEBUG_LOG = Path(
    os.environ.get("CODEX_IMAGE2_DEBUG_LOG", f"/tmp/{SERVER_NAME}-mcp-debug.log")
)
STATE_DIR = Path(
    os.environ.get(
        "CODEX_IMAGE2_STATE_DIR",
        str(Path.home() / ".claude" / "state" / SERVER_NAME),
    )
)
JOBS_DIR = STATE_DIR / "jobs"
RUNS_DIR = STATE_DIR / "runs"

TERMINAL_JOB_STATES = {"completed", "failed"}
_use_ndjson = False


def debug_log(message: str) -> None:
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"{message}\n")
    except OSError:
        pass


def send_response(response: dict[str, Any]) -> None:
    global _use_ndjson

    payload = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    debug_log(f"SEND {payload.decode('utf-8', errors='replace')}")
    if _use_ndjson:
        sys.stdout.write(payload + b"\n")
    else:
        header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
        sys.stdout.write(header + payload)
    sys.stdout.flush()


def read_message() -> dict[str, Any] | None:
    global _use_ndjson

    line = sys.stdin.readline()
    if not line:
        return None

    line_text = line.decode("utf-8").rstrip("\r\n")
    if line_text.lower().startswith("content-length:"):
        try:
            content_length = int(line_text.split(":", 1)[1].strip())
        except ValueError:
            return None

        while True:
            header_line = sys.stdin.readline()
            if not header_line:
                return None
            if header_line in {b"\r\n", b"\n"}:
                break

        body = sys.stdin.read(content_length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    if line_text.startswith("{") or line_text.startswith("["):
        _use_ndjson = True
        try:
            return json.loads(line_text)
        except json.JSONDecodeError:
            return None

    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def job_state_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def is_pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def find_codex_bin() -> str | None:
    if Path(CODEX_BIN).is_file():
        return CODEX_BIN
    return shutil.which(CODEX_BIN)


def normalize_string_list(raw_value: Any) -> tuple[list[str], str | None]:
    if raw_value is None:
        return [], None
    if isinstance(raw_value, str):
        candidate = raw_value.strip()
        return ([candidate] if candidate else []), None
    if not isinstance(raw_value, list):
        return [], "referenceImagePaths must be a string or an array of strings"

    values: list[str] = []
    for item in raw_value:
        if not isinstance(item, str):
            return [], "referenceImagePaths entries must be strings"
        candidate = item.strip()
        if candidate:
            values.append(candidate)
    return values, None


def resolve_cwd(raw_cwd: str | None) -> tuple[Path, str | None]:
    if raw_cwd:
        cwd = Path(raw_cwd).expanduser()
    else:
        cwd = Path.cwd()
    try:
        cwd = cwd.resolve()
    except OSError as exc:
        return cwd, f"failed to resolve cwd {raw_cwd!r}: {exc}"
    if not cwd.exists():
        return cwd, f"working directory does not exist: {cwd}"
    if not cwd.is_dir():
        return cwd, f"working directory is not a directory: {cwd}"
    return cwd, None


def resolve_output_path(raw_output_path: str | None, *, cwd: Path, job_id: str) -> Path:
    if raw_output_path:
        path = Path(raw_output_path).expanduser()
        if not path.is_absolute():
            path = cwd / path
    else:
        path = cwd / "figures" / "ai_generated" / f"codex-image2-{job_id}.png"
    return path.resolve()


def build_bridge_prompt(
    prompt: str,
    *,
    system: str | None,
    reference_image_paths: list[str],
) -> str:
    sections: list[str] = [
        "You are operating behind a Codex native image-generation MCP bridge.",
        "Use native image generation through the Codex app-server.",
        "Do not use shell commands, Python, SVG, HTML, Canvas, or manual bitmap encoding.",
        "Do not fabricate success. If native image generation is unavailable, reply exactly NATIVE_IMAGE_UNAVAILABLE.",
        "Generate exactly one publication-quality raster image unless the user explicitly requests multiple variants.",
        "",
    ]
    selected_system = (system or "").strip()
    if selected_system:
        sections.extend(["## System Instructions", selected_system, ""])
    if reference_image_paths:
        sections.append("## Reference Images")
        sections.append(
            "These local image files are available. If image viewing tools are available in this Codex session, inspect them before generating."
        )
        for path in reference_image_paths:
            sections.append(f"- {path}")
        sections.append("")
    sections.extend(["## User Request", prompt.strip()])
    return "\n".join(sections).strip()


def parse_debug_json_messages(raw_stdout: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    buffer: list[str] = []
    collecting = False

    for raw_line in raw_stdout.splitlines():
        if not collecting:
            if raw_line.startswith("< {"):
                collecting = True
                buffer = [raw_line[2:]]
            continue

        if not raw_line.startswith("< "):
            collecting = False
            buffer = []
            continue

        buffer.append(raw_line[2:])
        candidate = "\n".join(buffer)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        messages.append(payload)
        collecting = False
        buffer = []

    return messages


def extract_run_summary(messages: list[dict[str, Any]]) -> dict[str, Any]:
    thread_id: str | None = None
    agent_messages: list[str] = []
    image_items: list[dict[str, Any]] = []
    command_items: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        params = message.get("params")
        if isinstance(params, dict):
            candidate_thread_id = params.get("threadId")
            if isinstance(candidate_thread_id, str) and candidate_thread_id:
                thread_id = candidate_thread_id
            item = params.get("item")
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "agentMessage":
                    text = item.get("text")
                    if isinstance(text, str):
                        agent_messages.append(text)
                elif item_type == "imageGeneration":
                    image_items.append(item)
                elif item_type == "commandExecution":
                    command_items.append(item)

        result = message.get("result")
        if isinstance(result, dict):
            thread = result.get("thread")
            if isinstance(thread, dict):
                candidate_thread_id = thread.get("id")
                if isinstance(candidate_thread_id, str) and candidate_thread_id:
                    thread_id = candidate_thread_id

    return {
        "threadId": thread_id,
        "agentMessages": agent_messages,
        "imageItems": image_items,
        "commandItems": command_items,
    }


def materialize_generated_image(
    image_item: dict[str, Any],
    output_path: Path,
) -> tuple[Path | None, str | None, str | None, str | None]:
    saved_path_value = image_item.get("savedPath")
    revised_prompt = image_item.get("revisedPrompt")
    revised_prompt_text = revised_prompt if isinstance(revised_prompt, str) else None

    if isinstance(saved_path_value, str) and saved_path_value:
        source_path = Path(saved_path_value).expanduser()
        if source_path.is_file():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, output_path)
            return output_path, str(source_path), revised_prompt_text, None

    raw_result = image_item.get("result")
    if isinstance(raw_result, str) and raw_result.strip():
        try:
            decoded = base64.b64decode(raw_result)
        except ValueError as exc:
            return None, None, None, f"imageGeneration result was not valid base64: {exc}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(decoded)
        return output_path, None, revised_prompt_text, None

    return None, None, None, "imageGeneration item did not contain a savedPath or decodable result"


def run_codex_image(
    prompt: str,
    *,
    cwd: Path,
    output_path: Path,
    system: str | None = None,
    model: str | None = None,
    reference_image_paths: list[str] | None = None,
    timeout_sec: int | None = None,
    run_log_path: Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    bin_path = find_codex_bin()
    if not bin_path:
        return None, f"Codex CLI not found: {CODEX_BIN}"

    normalized_refs = reference_image_paths or []
    prompt_text = build_bridge_prompt(
        prompt,
        system=system,
        reference_image_paths=normalized_refs,
    )
    cmd = [bin_path]
    selected_model = model or DEFAULT_MODEL
    if selected_model:
        cmd.extend(["-c", f'model="{selected_model}"'])
    cmd.extend(["debug", "app-server", "send-message-v2", prompt_text])

    effective_timeout = timeout_sec or DEFAULT_TIMEOUT_SEC
    debug_log(f"RUN {' '.join(cmd)} cwd={cwd}")
    try:
        started = time.monotonic()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=str(cwd),
            timeout=effective_timeout,
            check=False,
        )
        duration_ms = int((time.monotonic() - started) * 1000)
    except subprocess.TimeoutExpired:
        return None, f"Codex image generation timed out after {effective_timeout} seconds"

    if run_log_path is not None:
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        run_log_path.write_text(
            result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""),
            encoding="utf-8",
        )

    messages = parse_debug_json_messages(result.stdout)
    summary = extract_run_summary(messages)

    if summary["commandItems"]:
        return None, (
            "Codex attempted shell-based image creation instead of native image generation. "
            "This bridge only accepts native imageGeneration events."
        )

    image_items = summary["imageItems"]
    if not image_items:
        final_message = ""
        for candidate in reversed(summary["agentMessages"]):
            if candidate and candidate.strip():
                final_message = candidate.strip()
                break
        stderr_text = result.stderr.strip()
        if final_message == "NATIVE_IMAGE_UNAVAILABLE":
            return None, "Codex app-server reported that native image generation is unavailable in this session."
        if final_message:
            return None, f"Codex did not emit an imageGeneration item. Final message: {final_message}"
        if stderr_text:
            return None, f"Codex did not emit an imageGeneration item. stderr: {stderr_text}"
        return None, "Codex did not emit an imageGeneration item."

    image_item = image_items[-1]
    generated_output, source_saved_path, revised_prompt, materialize_error = materialize_generated_image(
        image_item,
        output_path,
    )
    if materialize_error:
        return None, materialize_error
    if generated_output is None:
        return None, "Codex image bridge failed to materialize the generated image"

    response_text = ""
    for candidate in reversed(summary["agentMessages"]):
        if candidate is not None:
            response_text = candidate
            break

    return {
        "threadId": summary["threadId"],
        "response": response_text,
        "model": selected_model or None,
        "duration_ms": duration_ms,
        "nativeToolConfirmed": True,
        "imageCount": len(image_items),
        "outputPath": str(generated_output),
        "sourceSavedPath": source_saved_path,
        "revisedPrompt": revised_prompt,
        "runLogPath": str(run_log_path) if run_log_path is not None else None,
    }, None


def serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result") or {}
    return {
        "jobId": job.get("jobId"),
        "status": job.get("status"),
        "done": job.get("status") in TERMINAL_JOB_STATES,
        "threadId": result.get("threadId"),
        "response": result.get("response"),
        "model": result.get("model"),
        "duration_ms": result.get("duration_ms"),
        "nativeToolConfirmed": result.get("nativeToolConfirmed"),
        "imageCount": result.get("imageCount"),
        "outputPath": result.get("outputPath"),
        "sourceSavedPath": result.get("sourceSavedPath"),
        "revisedPrompt": result.get("revisedPrompt"),
        "runLogPath": result.get("runLogPath"),
        "error": job.get("error"),
        "createdAt": job.get("createdAt"),
        "startedAt": job.get("startedAt"),
        "completedAt": job.get("completedAt"),
        "updatedAt": job.get("updatedAt"),
        "resumeHint": "Call generate_status with this jobId until done=true.",
    }


def start_async_generate(
    prompt: str,
    *,
    cwd: str | None = None,
    output_path: str | None = None,
    system: str | None = None,
    model: str | None = None,
    reference_image_paths: Any = None,
) -> tuple[dict[str, Any] | None, str | None]:
    resolved_cwd, cwd_error = resolve_cwd(cwd)
    if cwd_error:
        return None, cwd_error

    refs, refs_error = normalize_string_list(reference_image_paths)
    if refs_error:
        return None, refs_error

    job_id = uuid.uuid4().hex
    resolved_output_path = resolve_output_path(output_path, cwd=resolved_cwd, job_id=job_id)
    created_at = utc_now()
    job = {
        "jobId": job_id,
        "status": "queued",
        "createdAt": created_at,
        "startedAt": None,
        "completedAt": None,
        "updatedAt": created_at,
        "error": None,
        "result": None,
        "workerPid": None,
        "request": {
            "prompt": prompt,
            "cwd": str(resolved_cwd),
            "outputPath": str(resolved_output_path),
            "system": system,
            "model": model,
            "referenceImagePaths": refs,
        },
    }

    job_path = job_state_path(job_id)
    write_json(job_path, job)

    try:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--run-job", job_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as exc:
        job["status"] = "failed"
        job["completedAt"] = utc_now()
        job["updatedAt"] = job["completedAt"]
        job["error"] = f"Failed to launch background image worker: {exc}"
        write_json(job_path, job)
        return None, job["error"]

    job["workerPid"] = worker.pid
    job["updatedAt"] = utc_now()
    write_json(job_path, job)
    debug_log(f"JOB_START job_id={job_id} worker_pid={worker.pid}")
    return serialize_job(job), None


def get_generate_status(job_id: str, *, wait_seconds: int = 0) -> tuple[dict[str, Any] | None, str | None]:
    job_path = job_state_path(job_id)
    if not job_path.exists():
        return None, f"Unknown jobId: {job_id}"

    deadline = time.monotonic() + max(wait_seconds, 0)
    while True:
        job = read_json(job_path)
        if job.get("status") in {"queued", "running"} and not is_pid_alive(job.get("workerPid")):
            job["status"] = "failed"
            job["error"] = "Background image worker exited before writing a final result"
            job["completedAt"] = utc_now()
            job["updatedAt"] = job["completedAt"]
            write_json(job_path, job)
        if job.get("status") in TERMINAL_JOB_STATES:
            return serialize_job(job), None
        if time.monotonic() >= deadline:
            return serialize_job(job), None
        time.sleep(min(0.5, max(deadline - time.monotonic(), 0.0)))


def run_async_job(job_id: str) -> int:
    job_path = job_state_path(job_id)
    if not job_path.exists():
        debug_log(f"JOB_MISSING job_id={job_id}")
        return 1

    job = read_json(job_path)
    job["status"] = "running"
    job["startedAt"] = utc_now()
    job["updatedAt"] = job["startedAt"]
    job["workerPid"] = os.getpid()
    write_json(job_path, job)
    debug_log(f"JOB_RUNNING job_id={job_id} worker_pid={os.getpid()}")

    request = job.get("request") or {}
    run_log_path = RUNS_DIR / f"{job_id}.log"
    try:
        payload, error = run_codex_image(
            str(request.get("prompt", "")),
            cwd=Path(str(request.get("cwd"))),
            output_path=Path(str(request.get("outputPath"))),
            system=request.get("system"),
            model=request.get("model"),
            reference_image_paths=request.get("referenceImagePaths") or [],
            run_log_path=run_log_path,
        )
    except Exception as exc:
        payload = None
        error = f"Background image generation crashed: {exc}"
        debug_log(traceback.format_exc())

    finished_at = utc_now()
    job = read_json(job_path)
    job["updatedAt"] = finished_at
    job["completedAt"] = finished_at
    if error:
        job["status"] = "failed"
        job["error"] = error
        job["result"] = None
        write_json(job_path, job)
        debug_log(f"JOB_FAILED job_id={job_id} error={error}")
        return 1

    job["status"] = "completed"
    job["error"] = None
    job["result"] = payload
    write_json(job_path, job)
    debug_log(f"JOB_COMPLETED job_id={job_id} output={(payload or {}).get('outputPath')}")
    return 0


def tool_success(request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        },
    }


def tool_error(request_id: Any, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": json.dumps({"error": message}, ensure_ascii=False)}],
            "isError": True,
        },
    }


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params", {})
    debug_log(f"REQUEST id={request_id!r} method={method} params={json.dumps(params, ensure_ascii=False)}")

    if request_id is None:
        if method in {"notifications/initialized", "initialized"}:
            return None
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": "0.1.0"},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}

    if method == "resources/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"resources": []}}

    if method == "resources/templates/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"resourceTemplates": []}}

    if method in {"notifications/initialized", "initialized"}:
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "generate",
                        "description": "Generate a raster image through the local Codex app-server native image tool and return the copied output path.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "Image generation prompt"},
                                "cwd": {"type": "string", "description": "Optional working directory"},
                                "outputPath": {"type": "string", "description": "Optional output file path; defaults to figures/ai_generated"},
                                "system": {"type": "string", "description": "Optional extra bridge instructions"},
                                "model": {"type": "string", "description": "Optional Codex text model override"},
                                "referenceImagePaths": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Optional local reference image paths that Codex may inspect before generating",
                                },
                            },
                            "required": ["prompt"],
                        },
                    },
                    {
                        "name": "generate_start",
                        "description": "Start a background native image generation job through the Codex app-server.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "Image generation prompt"},
                                "cwd": {"type": "string", "description": "Optional working directory"},
                                "outputPath": {"type": "string", "description": "Optional output file path; defaults to figures/ai_generated"},
                                "system": {"type": "string", "description": "Optional extra bridge instructions"},
                                "model": {"type": "string", "description": "Optional Codex text model override"},
                                "referenceImagePaths": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "array", "items": {"type": "string"}},
                                    ],
                                    "description": "Optional local reference image paths that Codex may inspect before generating",
                                },
                            },
                            "required": ["prompt"],
                        },
                    },
                    {
                        "name": "generate_status",
                        "description": "Check whether a background image generation job has finished and fetch the output path when available.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "jobId": {"type": "string", "description": "Background image job id"},
                                "job_id": {"type": "string", "description": "Alias of jobId"},
                                "waitSeconds": {"type": "integer", "description": "Optional bounded wait before returning status"},
                            },
                            "required": ["jobId"],
                        },
                    },
                ]
            },
        }

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        if not isinstance(args, dict):
            return tool_error(request_id, "tool arguments must be an object")

        if name == "generate":
            prompt = str(args.get("prompt", "")).strip()
            if not prompt:
                return tool_error(request_id, "prompt is required")
            resolved_cwd, cwd_error = resolve_cwd(args.get("cwd"))
            if cwd_error:
                return tool_error(request_id, cwd_error)
            refs, refs_error = normalize_string_list(args.get("referenceImagePaths"))
            if refs_error:
                return tool_error(request_id, refs_error)
            output_path = resolve_output_path(args.get("outputPath"), cwd=resolved_cwd, job_id=uuid.uuid4().hex)
            run_log_path = RUNS_DIR / f"sync-{uuid.uuid4().hex}.log"
            payload, error = run_codex_image(
                prompt,
                cwd=resolved_cwd,
                output_path=output_path,
                system=args.get("system"),
                model=args.get("model"),
                reference_image_paths=refs,
                run_log_path=run_log_path,
            )
            if error:
                return tool_error(request_id, error)
            return tool_success(request_id, payload or {})

        if name == "generate_start":
            prompt = str(args.get("prompt", "")).strip()
            if not prompt:
                return tool_error(request_id, "prompt is required")
            payload, error = start_async_generate(
                prompt,
                cwd=args.get("cwd"),
                output_path=args.get("outputPath"),
                system=args.get("system"),
                model=args.get("model"),
                reference_image_paths=args.get("referenceImagePaths"),
            )
            if error:
                return tool_error(request_id, error)
            return tool_success(request_id, payload or {})

        if name == "generate_status":
            job_id = args.get("jobId") or args.get("job_id")
            if not job_id:
                return tool_error(request_id, "jobId or job_id is required")
            wait_seconds = args.get("waitSeconds", 0)
            try:
                wait_seconds = int(wait_seconds)
            except (TypeError, ValueError):
                wait_seconds = 0
            payload, error = get_generate_status(str(job_id), wait_seconds=max(wait_seconds, 0))
            if error:
                return tool_error(request_id, error)
            return tool_success(request_id, payload or {})

        return tool_error(request_id, f"unknown tool: {name}")

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    }


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--run-job":
        return run_async_job(sys.argv[2])

    while True:
        request = read_message()
        if request is None:
            return 0
        response = handle_request(request)
        if response is not None:
            send_response(response)


if __name__ == "__main__":
    raise SystemExit(main())
