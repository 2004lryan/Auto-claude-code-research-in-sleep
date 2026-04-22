"""Unit tests for the experimental codex-image2 MCP bridge."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERVER_PATH = ROOT / "mcp-servers" / "codex-image2" / "server.py"
SPEC = importlib.util.spec_from_file_location("codex_image2_server", SERVER_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


SAMPLE_STDOUT = """
< {
<   "method": "item/completed",
<   "params": {
<     "item": {
<       "type": "imageGeneration",
<       "id": "ig_test",
<       "status": "generating",
<       "revisedPrompt": "blue circle",
<       "result": "aGVsbG8="
<     },
<     "threadId": "thread-123",
<     "turnId": "turn-123"
<   }
< }
< {
<   "method": "item/completed",
<   "params": {
<     "item": {
<       "type": "agentMessage",
<       "id": "msg-1",
<       "text": "",
<       "phase": "final_answer",
<       "memoryCitation": null
<     },
<     "threadId": "thread-123",
<     "turnId": "turn-123"
<   }
< }
"""


SAMPLE_WITH_COMMAND = """
< {
<   "method": "item/completed",
<   "params": {
<     "item": {
<       "type": "commandExecution",
<       "id": "cmd-1",
<       "command": "python3 make_png.py",
<       "cwd": "/tmp",
<       "processId": null,
<       "source": "model",
<       "status": "completed",
<       "commandActions": [],
<       "aggregatedOutput": "",
<       "exitCode": 0,
<       "durationMs": 12
<     },
<     "threadId": "thread-123",
<     "turnId": "turn-123"
<   }
< }
"""


class CodexImage2ServerTests(unittest.TestCase):
    def test_parse_debug_json_messages(self) -> None:
        messages = MODULE.parse_debug_json_messages(SAMPLE_STDOUT)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["params"]["item"]["type"], "imageGeneration")

    def test_extract_run_summary(self) -> None:
        messages = MODULE.parse_debug_json_messages(SAMPLE_STDOUT + SAMPLE_WITH_COMMAND)
        summary = MODULE.extract_run_summary(messages)
        self.assertEqual(summary["threadId"], "thread-123")
        self.assertEqual(len(summary["imageItems"]), 1)
        self.assertEqual(len(summary["commandItems"]), 1)

    def test_build_bridge_prompt_includes_references(self) -> None:
        prompt = MODULE.build_bridge_prompt(
            "Draw a workflow.",
            system="Academic style only.",
            reference_image_paths=["/tmp/ref1.png", "/tmp/ref2.png"],
        )
        self.assertIn("Academic style only.", prompt)
        self.assertIn("/tmp/ref1.png", prompt)
        self.assertIn("Draw a workflow.", prompt)
        self.assertIn("native image generation", prompt)

    def test_materialize_generated_image_from_base64(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.png"
            image_item = {
                "type": "imageGeneration",
                "revisedPrompt": "blue circle",
                "result": "aGVsbG8=",
            }
            generated_path, source_saved_path, revised_prompt, error = MODULE.materialize_generated_image(
                image_item,
                output_path,
            )
            self.assertEqual(generated_path, output_path)
            self.assertIsNone(source_saved_path)
            self.assertEqual(revised_prompt, "blue circle")
            self.assertIsNone(error)
            self.assertEqual(output_path.read_bytes(), b"hello")


if __name__ == "__main__":
    unittest.main()
