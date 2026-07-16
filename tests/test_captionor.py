from __future__ import annotations

import base64
import http.client
import json
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import captionor


class _RecordingServer(ThreadingHTTPServer):
    def __init__(self, responses):
        super().__init__(("127.0.0.1", 0), _RecordingHandler)
        self.responses = list(responses)
        self.recorded_requests = []


class _RecordingHandler(BaseHTTPRequestHandler):
    server: _RecordingServer

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.recorded_requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "json": json.loads(body.decode("utf-8")),
            }
        )
        status, payload = self.server.responses.pop(0)
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):  # noqa: A002
        pass


class ServerContext:
    def __init__(self, responses):
        self.server = _RecordingServer(responses)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self.server

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


def make_test_config(base_url: str = "http://127.0.0.1:1/v1") -> captionor.AppConfig:
    return captionor.AppConfig(
        api=captionor.ApiSettings(
            base_url=base_url,
            api_key="test-key",
            model="test-model",
            timeout_seconds=2,
            max_retries=0,
            retry_delay_seconds=0,
        ),
        image=captionor.ImageSettings(
            max_dimension=32,
            resize_mode="exact",
            resampling="bicubic",
        ),
        files=captionor.FileSettings(
            image_extensions=(".jpg", ".png", ".webp"),
            tag_extension=".txt",
            tag_filename_mode="stem",
            output_filename_mode="image_name",
            text_encoding="utf-8-sig",
        ),
        system_prompt="system instructions",
        request_options={
            "seed": 699,
            "temperature": 1,
            "repetition_penalty": 0.5,
        },
    )


class ConfigAndPathTests(unittest.TestCase):
    def test_default_config_preserves_workflow_values(self):
        config = captionor.load_config(captionor.DEFAULT_CONFIG_PATH)
        self.assertEqual(config.api.base_url, "http://localhost:8080/v1")
        self.assertEqual(config.api.max_retries, 2)
        self.assertEqual(config.image.max_dimension, 1024)
        self.assertEqual(config.image.resize_mode, "exact")
        self.assertEqual(config.request_options["seed"], 699)
        self.assertEqual(config.request_options["max_tokens"], 16384)
        self.assertIn("exactly 7 sentences", config.system_prompt)

    def test_make_job_uses_stem_tags_and_full_image_name_output(self):
        config = make_test_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "images"
            image = input_root / "nested" / "sample.v2.WEBP"
            output_root = root / "captions"
            job = captionor.make_job(
                image,
                input_path=input_root,
                tag_root=None,
                output_root=output_root,
                files=config.files,
            )
            self.assertEqual(job.tag_path, input_root / "nested" / "sample.v2.txt")
            self.assertEqual(job.output_path, output_root / "nested" / "sample.v2.WEBP.txt")

    def test_discovery_is_sorted_recursive_and_excludes_output_root(self):
        config = make_test_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "captions"
            (root / "sub").mkdir()
            output.mkdir()
            Image.new("RGB", (2, 2)).save(root / "z.PNG")
            Image.new("RGB", (2, 2)).save(root / "sub" / "A.jpg")
            Image.new("RGB", (2, 2)).save(output / "ignored.png")
            images = captionor.discover_images(
                root,
                recursive=True,
                extensions=config.files.image_extensions,
                excluded_root=output,
            )
            self.assertEqual(
                [path.relative_to(root).as_posix() for path in images],
                ["sub/A.jpg", "z.PNG"],
            )

    def test_output_root_equal_to_or_above_input_does_not_hide_images(self):
        config = make_test_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "images"
            input_root.mkdir()
            image_path = input_root / "one.png"
            Image.new("RGB", (2, 2)).save(image_path)
            for output_root in (input_root, root):
                with self.subTest(output_root=output_root):
                    images = captionor.discover_images(
                        input_root,
                        recursive=True,
                        extensions=config.files.image_extensions,
                        excluded_root=output_root,
                    )
                    self.assertEqual(images, [image_path])

    def test_discovery_has_casefold_tie_breaker(self):
        config = make_test_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upper = root / "A.JPG"
            lower = root / "a.jpg"
            Image.new("RGB", (2, 2)).save(upper)
            Image.new("RGB", (2, 2)).save(lower)
            images = captionor.discover_images(
                root,
                recursive=False,
                extensions=config.files.image_extensions,
            )
            self.assertEqual(images, [upper, lower])

    def test_cli_overrides_validate_url_dimension_and_preserve_none_resize(self):
        config = replace(
            make_test_config(),
            image=captionor.ImageSettings(32, "none", "bicubic"),
        )
        parser = captionor.build_argument_parser()
        args = parser.parse_args(["images", "--no-upscale"])
        self.assertEqual(captionor.apply_overrides(config, args).image.resize_mode, "none")

        bad_url = parser.parse_args(["images", "--base-url", "ftp://localhost/v1"])
        with self.assertRaises(captionor.ConfigError):
            captionor.apply_overrides(config, bad_url)

        zero_size = parser.parse_args(["images", "--max-dimension", "0"])
        with self.assertRaises(captionor.ConfigError):
            captionor.apply_overrides(config, zero_size)

    def test_extension_rejects_path_traversal(self):
        for extension in ("../txt", ".foo/bar", ".foo\\bar", "..", ".caption.txt"):
            with self.subTest(extension=extension):
                with self.assertRaises(captionor.ConfigError):
                    captionor._normalise_extension(extension, "test.extension")

    def test_job_preflight_rejects_original_and_duplicate_output_paths(self):
        config = make_test_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "one.jpg"
            Image.new("RGB", (2, 2)).save(image)
            image_collision_files = replace(
                config.files,
                tag_extension=".jpg",
                tag_filename_mode="image_name",
                output_filename_mode="stem",
            )
            with self.assertRaises(captionor.ConfigError):
                captionor.make_job(
                    image,
                    input_path=root,
                    tag_root=None,
                    output_root=root,
                    files=image_collision_files,
                )

            second = root / "one.png"
            Image.new("RGB", (2, 2)).save(second)
            duplicate_files = replace(config.files, output_filename_mode="stem")
            with self.assertRaises(captionor.ConfigError):
                captionor.prepare_jobs(
                    [image, second],
                    input_path=root,
                    tag_root=None,
                    output_root=root / "captions",
                    files=duplicate_files,
                )

    def test_job_preflight_rejects_output_subdirectory_symlink_escape(self):
        config = make_test_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "images"
            image = input_root / "nested" / "one.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(image)
            output_root = root / "captions"
            outside = root / "outside"
            output_root.mkdir()
            outside.mkdir()
            (output_root / "nested").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(captionor.ConfigError):
                captionor.prepare_jobs(
                    [image],
                    input_path=input_root,
                    tag_root=None,
                    output_root=output_root,
                    files=config.files,
                )

    def test_job_preflight_checks_all_images_before_limit_and_all_tag_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.jpg"
            second = root / "a.jpg.png"
            Image.new("RGB", (2, 2)).save(first)
            Image.new("RGB", (2, 2)).save(second)

            image_collision_config = make_test_config()
            image_collision_files = replace(
                image_collision_config.files,
                tag_extension=".png",
                output_filename_mode="image_name",
            )
            image_collision_config = replace(
                image_collision_config,
                files=image_collision_files,
            )
            with self.assertRaises(captionor.ConfigError):
                captionor.run_batch(
                    config=image_collision_config,
                    input_path=root,
                    output_root=root,
                    tag_root=None,
                    recursive=False,
                    overwrite=False,
                    missing_tags="error",
                    dry_run=True,
                    limit=1,
                    fail_fast=False,
                )

            tag_collision_config = make_test_config()
            with self.assertRaises(captionor.ConfigError):
                captionor.prepare_jobs(
                    [first, second],
                    input_path=root,
                    tag_root=None,
                    output_root=root,
                    files=tag_collision_config.files,
                )


class ImageAndPayloadTests(unittest.TestCase):
    def test_exact_resize_upscales_and_shrink_mode_does_not(self):
        exact = captionor.ImageSettings(1024, "exact", "bicubic")
        shrink = replace(exact, resize_mode="shrink")
        self.assertEqual(captionor._target_size(800, 600, exact), (1024, 768))
        self.assertEqual(captionor._target_size(600, 800, exact), (768, 1024))
        self.assertEqual(captionor._target_size(800, 600, shrink), (800, 600))
        self.assertEqual(captionor._target_size(1600, 900, shrink), (1024, 576))

    def test_image_is_reencoded_as_resized_rgb_png_data_url(self):
        settings = captionor.ImageSettings(32, "exact", "bicubic")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpha.webp"
            Image.new("RGBA", (20, 10), (255, 0, 0, 128)).save(path)
            data_url, size = captionor.prepare_png_data_url(path, settings)
            self.assertTrue(data_url.startswith("data:image/png;base64,"))
            decoded = base64.b64decode(data_url.split(",", 1)[1])
            with Image.open(BytesIO(decoded)) as image:
                self.assertEqual(image.size, (32, 16))
                self.assertEqual(image.mode, "RGB")
            self.assertEqual(size, (32, 16))

    def test_payload_matches_workflow_message_order_and_options(self):
        config = make_test_config()
        payload = captionor.build_payload(
            config,
            tags="1girl, blue hair",
            image_data_url="data:image/png;base64,AAAA",
        )
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "system instructions"})
        user_content = payload["messages"][1]["content"]
        self.assertEqual(user_content[0]["type"], "image_url")
        self.assertEqual(user_content[1], {"type": "text", "text": "1girl, blue hair"})
        self.assertEqual(payload["seed"], 699)
        self.assertEqual(payload["repetition_penalty"], 0.5)
        self.assertNotIn("stream", payload)

    def test_extract_caption_supports_string_and_text_parts(self):
        self.assertEqual(
            captionor.extract_caption({"choices": [{"message": {"content": "  result  "}}]}),
            "result",
        )
        self.assertEqual(
            captionor.extract_caption(
                {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, "b"]}}]}
            ),
            "ab",
        )
        with self.assertRaises(captionor.ApiError):
            captionor.extract_caption({"choices": []})


class HttpAndPipelineTests(unittest.TestCase):
    def test_tag_reader_preserves_non_comment_line_spacing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tags.txt"
            path.write_text("  first  \n  # ignored\n\n  second  ", encoding="utf-8")
            self.assertEqual(captionor.read_tags(path, "utf-8"), "  first  \n\n  second  ")

    def test_atomic_write_does_not_follow_predictable_temp_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "caption.txt"
            victim = root / "victim.txt"
            victim.write_text("keep", encoding="utf-8")
            old_predictable_temp = root / f".{output.name}.{captionor.os.getpid()}.tmp"
            old_predictable_temp.symlink_to(victim)
            captionor.write_caption_atomic(output, "new caption")
            self.assertEqual(victim.read_text(encoding="utf-8"), "keep")
            self.assertEqual(output.read_text(encoding="utf-8"), "new caption\n")

    def test_post_json_retries_transient_error_and_sends_auth(self):
        responses = [
            (501, {"error": "temporary"}),
            (200, {"choices": [{"message": {"content": "caption"}}]}),
        ]
        with ServerContext(responses) as server:
            result = captionor.post_json(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                {"model": "m"},
                api_key="secret",
                timeout_seconds=2,
                max_retries=1,
                retry_delay_seconds=0,
            )
            self.assertEqual(captionor.extract_caption(result), "caption")
            self.assertEqual(len(server.recorded_requests), 2)
            self.assertEqual(server.recorded_requests[0]["path"], "/v1/chat/completions")
            self.assertEqual(server.recorded_requests[0]["headers"]["Authorization"], "Bearer secret")

    def test_http_400_is_not_retried(self):
        with ServerContext([(400, {"error": "bad request"})]) as server:
            with self.assertRaises(captionor.ApiError):
                captionor.post_json(
                    f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    {"model": "m"},
                    api_key="",
                    timeout_seconds=2,
                    max_retries=2,
                    retry_delay_seconds=0,
                )
            self.assertEqual(len(server.recorded_requests), 1)

    def test_incomplete_response_is_retried_then_reported(self):
        with patch(
            "captionor.urlrequest.urlopen",
            side_effect=http.client.IncompleteRead(b"partial", 100),
        ) as mocked_open:
            with self.assertRaises(captionor.ApiError):
                captionor.post_json(
                    "http://localhost:8080/v1/chat/completions",
                    {"model": "m"},
                    api_key="",
                    timeout_seconds=2,
                    max_retries=1,
                    retry_delay_seconds=0,
                )
            self.assertEqual(mocked_open.call_count, 2)

    def test_end_to_end_batch_and_resume(self):
        response = {"choices": [{"message": {"role": "assistant", "content": "A caption."}}]}
        with tempfile.TemporaryDirectory() as directory, ServerContext([(200, response)]) as server:
            root = Path(directory)
            input_root = root / "images"
            output_root = root / "captions"
            input_root.mkdir()
            image_path = input_root / "장면.one.WEBP"
            Image.new("RGB", (20, 10), "blue").save(image_path, format="WEBP")
            (input_root / "장면.one.txt").write_text("1girl, blue hair", encoding="utf-8")

            config = make_test_config(f"http://127.0.0.1:{server.server_port}/v1")
            stats = captionor.run_batch(
                config=config,
                input_path=input_root,
                output_root=output_root,
                tag_root=None,
                recursive=False,
                overwrite=False,
                missing_tags="error",
                dry_run=False,
                limit=None,
                fail_fast=False,
            )
            output_path = output_root / "장면.one.WEBP.txt"
            self.assertEqual(stats.generated, 1)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "A caption.\n")
            request_payload = server.recorded_requests[0]["json"]
            self.assertEqual(request_payload["messages"][1]["content"][1]["text"], "1girl, blue hair")

            resumed = captionor.run_batch(
                config=config,
                input_path=input_root,
                output_root=output_root,
                tag_root=None,
                recursive=False,
                overwrite=False,
                missing_tags="error",
                dry_run=False,
                limit=None,
                fail_fast=False,
            )
            self.assertEqual(resumed.skipped_existing, 1)
            self.assertEqual(len(server.recorded_requests), 1)


if __name__ == "__main__":
    unittest.main()
