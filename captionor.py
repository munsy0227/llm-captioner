#!/usr/bin/env python3
"""Standalone batch image captioner converted from a ComfyUI workflow.

The program sends a resized image and its tag sidecar to an OpenAI-compatible
vision endpoint, then stores one caption per image.  It intentionally has no
ComfyUI dependency; Pillow and its JPEG XL plugin handle image decoding.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import re
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse

try:
    import pillow_jxl  # noqa: F401 - importing registers JPEG XL with Pillow
except (ImportError, OSError) as exc:
    _JXL_IMPORT_ERROR: Exception | None = exc
else:
    _JXL_IMPORT_ERROR = None

from PIL import Image, ImageOps, UnidentifiedImageError


PROGRAM_NAME = "captionor"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("captionor_config.json")
RETRYABLE_HTTP_STATUS = {408, 409, 429}
RESAMPLING_METHODS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "area": Image.Resampling.BOX,
}
RESERVED_REQUEST_KEYS = {"model", "messages", "n", "stream"}
JXL_DECODER_AVAILABLE = ".jxl" in Image.registered_extensions()
TEXT_PREVIEW_CHARACTERS = 2_048


class CaptionorError(Exception):
    """Base class for expected, user-facing failures."""


class ConfigError(CaptionorError):
    """Raised when configuration or CLI input is invalid."""


class ApiError(CaptionorError):
    """Raised when the OpenAI-compatible endpoint returns an invalid result."""


@dataclass(frozen=True)
class ApiSettings:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float
    max_retries: int
    retry_delay_seconds: float


@dataclass(frozen=True)
class ImageSettings:
    max_dimension: int
    resize_mode: str
    resampling: str


@dataclass(frozen=True)
class FileSettings:
    image_extensions: tuple[str, ...]
    tag_extension: str
    tag_filename_mode: str
    output_filename_mode: str
    text_encoding: str


@dataclass(frozen=True)
class CaptionSettings:
    max_output_bytes: int
    max_attempts: int


@dataclass(frozen=True)
class AppConfig:
    api: ApiSettings
    image: ImageSettings
    files: FileSettings
    caption: CaptionSettings
    system_prompt: str
    request_options: dict[str, Any]


@dataclass(frozen=True)
class Job:
    image_path: Path
    tag_path: Path
    output_path: Path


@dataclass
class RunStats:
    discovered: int = 0
    selected: int = 0
    processed: int = 0
    generated: int = 0
    skipped_existing: int = 0
    skipped_missing_tags: int = 0
    failed: int = 0

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"설정 '{name}'은 JSON 객체여야 합니다.")
    return value


def _require_string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f"설정 '{name}'은 비어 있지 않은 문자열이어야 합니다.")
    return value


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"설정 '{name}'은 {minimum} 이상의 정수여야 합니다.")
    return value


def _require_number(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ConfigError(f"설정 '{name}'은 {minimum} 이상의 숫자여야 합니다.")
    return float(value)


def _normalise_extension(extension: str, name: str) -> str:
    extension = extension.strip().lower()
    if not extension:
        raise ConfigError(f"설정 '{name}'에 빈 확장자를 사용할 수 없습니다.")
    extension = extension if extension.startswith(".") else f".{extension}"
    if re.fullmatch(r"\.[a-z0-9][a-z0-9_-]*", extension) is None:
        raise ConfigError(
            f"설정 '{name}'의 확장자가 올바르지 않습니다: {extension}. "
            "영문 소문자, 숫자, 밑줄, 하이픈만 사용할 수 있습니다."
        )
    return extension


def _normalise_base_url(value: object, name: str) -> str:
    base_url = _require_string(value, name).rstrip("/")
    try:
        parsed = urlparse(base_url)
        _ = parsed.port  # Access validates malformed port syntax when present.
    except ValueError as exc:
        raise ConfigError(f"설정 '{name}'의 URL이 올바르지 않습니다: {exc}") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"설정 '{name}'은 올바른 http(s) URL이어야 합니다.")
    return base_url


def load_config(path: Path) -> AppConfig:
    """Load and validate a JSON configuration file."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"설정 파일을 찾을 수 없습니다: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"설정 파일을 읽을 수 없습니다: {path}: {exc}") from exc

    root = _require_mapping(raw, "root")
    api_raw = _require_mapping(root.get("api"), "api")
    image_raw = _require_mapping(root.get("image"), "image")
    files_raw = _require_mapping(root.get("files"), "files")
    caption_raw = _require_mapping(root.get("caption", {}), "caption")

    base_url = _normalise_base_url(api_raw.get("base_url"), "api.base_url")

    api = ApiSettings(
        base_url=base_url,
        api_key=_require_string(api_raw.get("api_key", ""), "api.api_key", allow_empty=True),
        model=_require_string(api_raw.get("model"), "api.model"),
        timeout_seconds=_require_number(
            api_raw.get("timeout_seconds", 600), "api.timeout_seconds", minimum=0.001
        ),
        max_retries=_require_int(api_raw.get("max_retries", 2), "api.max_retries"),
        retry_delay_seconds=_require_number(
            api_raw.get("retry_delay_seconds", 1),
            "api.retry_delay_seconds",
        ),
    )

    resize_mode = _require_string(
        image_raw.get("resize_mode", "exact"), "image.resize_mode"
    ).lower()
    if resize_mode not in {"exact", "shrink", "none"}:
        raise ConfigError("설정 'image.resize_mode'은 exact, shrink, none 중 하나여야 합니다.")
    resampling = _require_string(
        image_raw.get("resampling", "bicubic"), "image.resampling"
    ).lower()
    if resampling not in RESAMPLING_METHODS:
        choices = ", ".join(sorted(RESAMPLING_METHODS))
        raise ConfigError(f"지원하지 않는 리사이즈 방식입니다: {resampling} (지원: {choices})")
    image = ImageSettings(
        max_dimension=_require_int(
            image_raw.get("max_dimension", 1024), "image.max_dimension", minimum=1
        ),
        resize_mode=resize_mode,
        resampling=resampling,
    )

    raw_extensions = files_raw.get("image_extensions")
    if not isinstance(raw_extensions, list) or not raw_extensions:
        raise ConfigError("설정 'files.image_extensions'은 하나 이상의 문자열 배열이어야 합니다.")
    image_extensions = tuple(
        dict.fromkeys(
            _normalise_extension(_require_string(item, "files.image_extensions[]"),
                                 "files.image_extensions[]")
            for item in raw_extensions
        )
    )
    tag_filename_mode = _require_string(
        files_raw.get("tag_filename_mode", "stem"), "files.tag_filename_mode"
    ).lower()
    if tag_filename_mode not in {"stem", "image_name"}:
        raise ConfigError("설정 'files.tag_filename_mode'은 stem 또는 image_name이어야 합니다.")
    output_filename_mode = _require_string(
        files_raw.get("output_filename_mode", "image_name"),
        "files.output_filename_mode",
    ).lower()
    if output_filename_mode not in {"stem", "image_name"}:
        raise ConfigError("설정 'files.output_filename_mode'은 stem 또는 image_name이어야 합니다.")
    files = FileSettings(
        image_extensions=image_extensions,
        tag_extension=_normalise_extension(
            _require_string(files_raw.get("tag_extension", ".txt"), "files.tag_extension"),
            "files.tag_extension",
        ),
        tag_filename_mode=tag_filename_mode,
        output_filename_mode=output_filename_mode,
        text_encoding=_require_string(
            files_raw.get("text_encoding", "utf-8-sig"), "files.text_encoding"
        ),
    )
    caption = CaptionSettings(
        max_output_bytes=_require_int(
            caption_raw.get("max_output_bytes", 2_048),
            "caption.max_output_bytes",
            minimum=1,
        ),
        max_attempts=_require_int(
            caption_raw.get("max_attempts", 2),
            "caption.max_attempts",
            minimum=1,
        ),
    )

    inline_prompt = root.get("system_prompt")
    prompt_file = root.get("system_prompt_file")
    if inline_prompt is not None and prompt_file is not None:
        raise ConfigError("'system_prompt'와 'system_prompt_file'은 동시에 지정할 수 없습니다.")
    if prompt_file is not None:
        prompt_path = Path(_require_string(prompt_file, "system_prompt_file"))
        if not prompt_path.is_absolute():
            prompt_path = path.parent / prompt_path
        try:
            system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ConfigError(f"시스템 프롬프트 파일을 읽을 수 없습니다: {prompt_path}: {exc}") from exc
    else:
        system_prompt = _require_string(inline_prompt, "system_prompt").strip()
    if not system_prompt:
        raise ConfigError("시스템 프롬프트가 비어 있습니다.")

    request_options_raw = _require_mapping(root.get("request_options", {}), "request_options")
    request_options = dict(request_options_raw)
    conflicts = RESERVED_REQUEST_KEYS.intersection(request_options)
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ConfigError(f"request_options에서 예약 필드를 덮어쓸 수 없습니다: {names}")

    return AppConfig(
        api=api,
        image=image,
        files=files,
        caption=caption,
        system_prompt=system_prompt,
        request_options=request_options,
    )


def discover_images(
    input_path: Path,
    *,
    recursive: bool,
    extensions: Sequence[str],
    excluded_root: Path | None = None,
) -> list[Path]:
    """Return a deterministic list of supported input images."""

    if not input_path.exists():
        raise ConfigError(f"입력 경로를 찾을 수 없습니다: {input_path}")
    allowed = {extension.lower() for extension in extensions}
    if input_path.is_file():
        if input_path.suffix.lower() not in allowed:
            raise ConfigError(f"지원하지 않는 이미지 확장자입니다: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise ConfigError(f"입력 경로는 이미지 파일 또는 폴더여야 합니다: {input_path}")

    candidates: Iterable[Path] = input_path.rglob("*") if recursive else input_path.iterdir()
    input_root = input_path.resolve()
    excluded = excluded_root.resolve() if excluded_root is not None else None
    if excluded is not None:
        try:
            excluded_relative = excluded.relative_to(input_root)
        except ValueError:
            excluded = None
        else:
            # Exclude only a strict output subdirectory. If output_root is the
            # input directory itself (or one of its parents), no image should
            # disappear from discovery.
            if excluded_relative == Path("."):
                excluded = None
    images: list[Path] = []
    for candidate in candidates:
        if not candidate.is_file() or candidate.suffix.lower() not in allowed:
            continue
        if excluded is not None:
            try:
                candidate.resolve().relative_to(excluded)
            except ValueError:
                pass
            else:
                continue
        images.append(candidate)
    images.sort(
        key=lambda item: (
            item.relative_to(input_path).as_posix().casefold(),
            item.relative_to(input_path).as_posix(),
        )
    )
    return images


def _sidecar_name(image: Path, extension: str, mode: str) -> str:
    if mode == "stem":
        return f"{image.stem}{extension}"
    return f"{image.name}{extension}"


def make_job(
    image_path: Path,
    *,
    input_path: Path,
    tag_root: Path | None,
    output_root: Path,
    files: FileSettings,
) -> Job:
    """Resolve tag and output paths while preserving relative subfolders."""

    if image_path == input_path:
        relative = Path(image_path.name)
    else:
        try:
            relative = image_path.relative_to(input_path)
        except ValueError:
            relative = Path(image_path.name)
    relative_parent = relative.parent

    if tag_root is None:
        tag_parent = image_path.parent
    else:
        tag_parent = tag_root / relative_parent
    tag_path = tag_parent / _sidecar_name(
        image_path, files.tag_extension, files.tag_filename_mode
    )
    output_path = output_root / relative_parent / _sidecar_name(
        image_path, files.tag_extension, files.output_filename_mode
    )
    if tag_path.resolve() == output_path.resolve():
        raise ConfigError(
            f"입력 태그와 출력 캡션 경로가 같습니다: {tag_path}. "
            "다른 --output-dir 또는 output_filename_mode를 사용하세요."
        )
    if image_path.resolve() == output_path.resolve():
        raise ConfigError(
            f"원본 이미지와 출력 캡션 경로가 같습니다: {image_path}. "
            "원본 보호를 위해 작업을 중단합니다."
        )
    return Job(image_path=image_path, tag_path=tag_path, output_path=output_path)


def prepare_jobs(
    images: Sequence[Path],
    *,
    input_path: Path,
    tag_root: Path | None,
    output_root: Path,
    files: FileSettings,
) -> list[Job]:
    """Resolve every path and reject collisions before processing anything."""

    input_paths = {image.resolve(): image for image in images}
    resolved_output_root = output_root.resolve()
    jobs = [
        make_job(
            image,
            input_path=input_path,
            tag_root=tag_root,
            output_root=output_root,
            files=files,
        )
        for image in images
    ]
    tag_paths: dict[Path, Path] = {}
    for job in jobs:
        resolved_tag = job.tag_path.resolve()
        tag_paths.setdefault(resolved_tag, job.image_path)
    output_paths: dict[Path, Path] = {}
    for job in jobs:
        image = job.image_path
        resolved_output = job.output_path.resolve()
        try:
            resolved_output.relative_to(resolved_output_root)
        except ValueError as exc:
            raise ConfigError(
                f"출력 캡션 경로가 출력 폴더 밖을 가리킵니다: "
                f"{job.output_path} (출력 폴더: {output_root})"
            ) from exc
        conflicting_input = input_paths.get(resolved_output)
        if conflicting_input is not None:
            raise ConfigError(
                f"출력 캡션 경로가 입력 이미지와 겹칩니다: "
                f"{job.output_path} == {conflicting_input}"
            )
        conflicting_tag_image = tag_paths.get(resolved_output)
        if conflicting_tag_image is not None:
            raise ConfigError(
                f"출력 캡션 경로가 입력 태그와 겹칩니다: "
                f"{job.output_path} (태그 대상 이미지: {conflicting_tag_image})"
            )
        previous_image = output_paths.get(resolved_output)
        if previous_image is not None:
            raise ConfigError(
                f"여러 이미지의 출력 캡션 경로가 같습니다: "
                f"{previous_image}, {image} -> {job.output_path}"
            )
        output_paths[resolved_output] = image
    return jobs


def _target_size(width: int, height: int, settings: ImageSettings) -> tuple[int, int]:
    if settings.resize_mode == "none":
        return width, height
    largest = max(width, height)
    if settings.resize_mode == "shrink" and largest <= settings.max_dimension:
        return width, height
    if width > height:
        return settings.max_dimension, max(1, round(height / width * settings.max_dimension))
    if height > width:
        return max(1, round(width / height * settings.max_dimension)), settings.max_dimension
    return settings.max_dimension, settings.max_dimension


def prepare_png_data_url(
    image_path: Path, settings: ImageSettings
) -> tuple[str, tuple[int, int]]:
    """Apply ComfyUI-like preprocessing and return a base64 PNG data URL."""

    if image_path.suffix.lower() == ".jxl" and not JXL_DECODER_AVAILABLE:
        reason = f" (불러오기 오류: {_JXL_IMPORT_ERROR})" if _JXL_IMPORT_ERROR else ""
        raise CaptionorError(
            f"JPEG XL 이미지를 열 수 없습니다: {image_path}. "
            "pillow-jxl-plugin을 설치하려면 "
            "'python3 -m pip install -r requirements.txt'를 실행하세요."
            f"{reason}"
        )

    try:
        with Image.open(image_path) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (
        OSError,
        RuntimeError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as exc:
        raise CaptionorError(f"이미지를 열 수 없습니다: {image_path}: {exc}") from exc

    target = _target_size(image.width, image.height, settings)
    if image.size != target:
        image = image.resize(target, RESAMPLING_METHODS[settings.resampling])

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", target


def build_payload(
    config: AppConfig,
    *,
    tags: str,
    image_data_url: str,
    attempt: int = 1,
) -> dict[str, Any]:
    """Build the non-streaming Chat Completions request used by the workflow."""

    user_content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": image_data_url},
        },
        {"type": "text", "text": tags},
    ]
    if attempt > 1:
        user_content.append(
            {
                "type": "text",
                "text": (
                    "The previous generated caption was too long. Generate a new, "
                    "shorter caption so the complete UTF-8 output file, including "
                    "its final newline, is at most "
                    f"{config.caption.max_output_bytes} bytes."
                ),
            }
        )
    payload: dict[str, Any] = {
        "model": config.api.model,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "n": 1,
    }
    options = dict(config.request_options)
    seed = options.get("seed")
    if attempt > 1 and isinstance(seed, int) and not isinstance(seed, bool):
        options["seed"] = seed + attempt - 1
    payload.update(options)
    return payload


def chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _error_body_text(response: Any, *, limit: int = 2_000) -> str:
    try:
        raw = response.read(limit + 1)
    except (OSError, http.client.HTTPException):
        return ""
    text = raw.decode("utf-8", errors="replace")
    return text[:limit] + ("…" if len(text) > limit else "")


def _retry_delay(base_delay: float, attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return max(0.0, min(float(retry_after), 60.0))
        except ValueError:
            pass
    return min(base_delay * (2**attempt), 30.0)


def post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    api_key: str,
    timeout_seconds: float,
    max_retries: int,
    retry_delay_seconds: float,
) -> Mapping[str, Any]:
    """POST JSON with bounded retries for transient endpoint failures."""

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(max_retries + 1):
        request = urlrequest.Request(url, data=body, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(request, timeout=timeout_seconds) as response:
                response_body = response.read()
        except urlerror.HTTPError as exc:
            detail = _error_body_text(exc)
            retryable = exc.code in RETRYABLE_HTTP_STATUS or exc.code >= 500
            if retryable and attempt < max_retries:
                delay = _retry_delay(
                    retry_delay_seconds, attempt, exc.headers.get("Retry-After")
                )
                if delay:
                    time.sleep(delay)
                continue
            suffix = f": {detail}" if detail else ""
            raise ApiError(f"API가 HTTP {exc.code} 오류를 반환했습니다{suffix}") from exc
        except (
            urlerror.URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            http.client.HTTPException,
        ) as exc:
            if attempt < max_retries:
                delay = _retry_delay(retry_delay_seconds, attempt, None)
                if delay:
                    time.sleep(delay)
                continue
            reason = getattr(exc, "reason", exc)
            raise ApiError(f"API 서버에 연결할 수 없습니다: {reason}") from exc

        try:
            decoded = json.loads(response_body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            preview = response_body[:500].decode("utf-8", errors="replace")
            raise ApiError(f"API 응답이 올바른 JSON이 아닙니다: {preview}") from exc
        if not isinstance(decoded, Mapping):
            raise ApiError("API 응답의 최상위 값이 JSON 객체가 아닙니다.")
        return decoded

    raise AssertionError("retry loop exited unexpectedly")


def extract_caption(response: Mapping[str, Any]) -> str:
    """Extract and validate choices[0].message.content."""

    try:
        choices = response["choices"]
        first_choice = choices[0]
        content = first_choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        error_value = response.get("error")
        detail = f": {error_value}" if error_value is not None else ""
        raise ApiError(f"API 응답에 choices[0].message.content가 없습니다{detail}") from exc

    if isinstance(content, str):
        caption = content.strip()
    elif isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                pieces.append(part["text"])
        caption = "".join(pieces).strip()
    else:
        caption = ""
    if not caption:
        raise ApiError("API 응답의 캡션 내용이 비어 있습니다.")
    return caption


def _stored_text_size(text: str, encoding: str) -> int:
    try:
        return len(f"{text}\n".encode(encoding))
    except (LookupError, UnicodeError) as exc:
        raise CaptionorError(f"텍스트를 {encoding} 형식으로 인코딩할 수 없습니다: {exc}") from exc


def _text_preview(text: str, *, max_characters: int = TEXT_PREVIEW_CHARACTERS) -> str:
    truncated = len(text) > max_characters
    visible = text[:max_characters] + ("…" if truncated else "")
    return json.dumps(visible, ensure_ascii=False)


def _print_text_preview(label: str, text: str, byte_size: int) -> None:
    print(
        f"{label} ({byte_size:,}바이트): {_text_preview(text)}",
        flush=True,
    )


def generate_caption(
    config: AppConfig,
    *,
    tags: str,
    image_data_url: str,
) -> str:
    """Generate a caption, retrying only when its saved UTF-8 form is too large."""

    last_size = 0
    for attempt in range(1, config.caption.max_attempts + 1):
        print(f"Gemma 캡션 요청: {attempt}/{config.caption.max_attempts}", flush=True)
        response = post_json(
            chat_completions_url(config.api.base_url),
            build_payload(
                config,
                tags=tags,
                image_data_url=image_data_url,
                attempt=attempt,
            ),
            api_key=config.api.api_key,
            timeout_seconds=config.api.timeout_seconds,
            max_retries=config.api.max_retries,
            retry_delay_seconds=config.api.retry_delay_seconds,
        )
        caption = extract_caption(response)
        last_size = _stored_text_size(caption, "utf-8")
        _print_text_preview("Gemma 캡션 응답", caption, last_size)
        if last_size <= config.caption.max_output_bytes:
            return caption
        if attempt < config.caption.max_attempts:
            print(
                f"캡션 저장 크기 {last_size:,}바이트가 기준 "
                f"{config.caption.max_output_bytes:,}바이트를 초과하여 다시 요청합니다.",
                flush=True,
            )
    raise ApiError(
        f"{config.caption.max_attempts}회의 Gemma 캡션 결과가 모두 "
        f"{config.caption.max_output_bytes:,}바이트 기준을 초과했습니다 "
        f"(마지막 결과: {last_size:,}바이트)."
    )


def read_tags(path: Path, encoding: str) -> str:
    try:
        text = path.read_text(encoding=encoding)
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, LookupError) as exc:
        raise CaptionorError(f"태그 파일을 읽을 수 없습니다: {path}: {exc}") from exc

    # WAS Load Text File ignores comment-only lines and normalises newlines.
    lines = [line for line in text.splitlines() if not line.strip().startswith("#")]
    tags = "\n".join(lines)
    if not tags.strip():
        raise CaptionorError(f"태그 파일이 비어 있습니다: {path}")
    return tags


def _write_text_atomic(path: Path, text: str, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="\n",
            prefix=".captionor-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def write_caption_atomic(path: Path, caption: str) -> None:
    """Atomically replace a caption only after a valid response exists."""

    _write_text_atomic(path, caption, "utf-8")


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _print_progress(*, processed: int, total: int, elapsed: float) -> None:
    percentage = processed / total * 100
    expected_total = elapsed / processed * total
    remaining = max(0.0, expected_total - elapsed)
    print(
        f"진행률: {processed}/{total} ({percentage:.1f}%) | "
        f"경과 {_format_duration(elapsed)} | "
        f"예상 총 {_format_duration(expected_total)} | "
        f"예상 남은 시간 {_format_duration(remaining)}",
        flush=True,
    )


def run_batch(
    *,
    config: AppConfig,
    input_path: Path,
    output_root: Path,
    tag_root: Path | None,
    recursive: bool,
    overwrite: bool,
    missing_tags: str,
    dry_run: bool,
    limit: int | None,
    fail_fast: bool,
    clock: Callable[[], float] | None = None,
) -> RunStats:
    """Discover and process images sequentially."""

    images = discover_images(
        input_path,
        recursive=recursive,
        extensions=config.files.image_extensions,
        excluded_root=output_root if input_path.is_dir() else None,
    )
    stats = RunStats(discovered=len(images))
    if not images:
        raise ConfigError(f"처리할 이미지가 없습니다: {input_path}")
    if limit is not None and limit < 1:
        raise ConfigError("limit은 1 이상이어야 합니다.")
    all_jobs = prepare_jobs(
        images,
        input_path=input_path,
        tag_root=tag_root,
        output_root=output_root,
        files=config.files,
    )
    jobs = all_jobs[:limit] if limit is not None else all_jobs
    stats.selected = len(jobs)
    timer = time.monotonic if clock is None else clock
    started_at = timer()

    print(
        f"처리 대상: {len(jobs)}개 | 예상 시간: 첫 항목 완료 후 계산",
        flush=True,
    )
    for index, job in enumerate(jobs, start=1):
        image_path = job.image_path
        try:
            prefix = f"[현재 항목 {index}/{len(jobs)}] {image_path}"
            print(f"{prefix} -> 처리 시작", flush=True)
            should_skip_existing = not overwrite and _is_nonempty_file(job.output_path)

            try:
                tag_byte_size = job.tag_path.stat().st_size
                try:
                    tags = read_tags(job.tag_path, config.files.text_encoding)
                except CaptionorError as exc:
                    if should_skip_existing:
                        stats.skipped_existing += 1
                        print(
                            f"{prefix} -> 건너뜀 (출력 있음, 태그 표시 불가: {exc})",
                            flush=True,
                        )
                        continue
                    raise
            except FileNotFoundError:
                if should_skip_existing:
                    stats.skipped_existing += 1
                    print(
                        f"{prefix} -> 건너뜀 (출력 있음, 태그 없음: {job.tag_path})",
                        flush=True,
                    )
                    continue
                if missing_tags == "skip":
                    stats.skipped_missing_tags += 1
                    print(
                        f"{prefix} -> 건너뜀 (태그 없음: {job.tag_path})",
                        flush=True,
                    )
                    continue
                raise CaptionorError(f"태그 파일을 찾을 수 없습니다: {job.tag_path}")

            _print_text_preview("현재 태그", tags, tag_byte_size)

            if should_skip_existing:
                stats.skipped_existing += 1
                print(f"{prefix} -> 건너뜀 (출력 있음)", flush=True)
                continue

            if dry_run:
                print(
                    f"{prefix} + 태그 {job.tag_path} -> {job.output_path} "
                    "(캡션 생성 예정, 미리보기)",
                    flush=True,
                )
                continue

            image_data_url, size = prepare_png_data_url(job.image_path, config.image)
            caption = generate_caption(
                config,
                tags=tags,
                image_data_url=image_data_url,
            )
            write_caption_atomic(job.output_path, caption)
            stats.generated += 1
            print(
                f"{prefix} -> 완료 {size[0]}x{size[1]} -> {job.output_path}",
                flush=True,
            )
        except (CaptionorError, OSError, ValueError) as exc:
            stats.failed += 1
            print(
                f"[{index}/{len(jobs)}] 실패: {image_path}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if fail_fast:
                break
        finally:
            stats.processed = index
            _print_progress(
                processed=index,
                total=len(jobs),
                elapsed=max(0.0, timer() - started_at),
            )
    return stats


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=(
            "이미지와 같은 stem의 태그 .txt를 로컬 OpenAI 호환 비전 API에 보내 "
            "자연어 캡션을 생성합니다."
        ),
    )
    parser.add_argument("input", type=Path, help="입력 이미지 파일 또는 폴더")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON 설정 파일 (기본값: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--output-dir", type=Path, help="출력 폴더 (기본값: INPUT/captions)")
    parser.add_argument("--tag-dir", type=Path, help="태그 파일 루트 (기본값: 이미지와 같은 폴더)")
    parser.add_argument("--recursive", action="store_true", help="하위 폴더까지 처리")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 캡션 파일도 새로 생성해 덮어쓰기",
    )
    parser.add_argument(
        "--missing-tags",
        choices=("error", "skip"),
        default="error",
        help="태그 파일 누락 처리 방식 (기본값: error)",
    )
    parser.add_argument("--dry-run", action="store_true", help="API 호출과 파일 쓰기 없이 작업 목록 확인")
    parser.add_argument("--limit", type=int, help="정렬된 앞쪽 N개 이미지만 처리")
    parser.add_argument("--fail-fast", action="store_true", help="첫 실패에서 즉시 중단")
    parser.add_argument("--base-url", help="설정의 API base_url 덮어쓰기")
    parser.add_argument("--api-key", help="설정의 API 키 덮어쓰기 (환경변수 CAPTIONOR_API_KEY 우선 지원)")
    parser.add_argument("--model", help="설정의 model ID 덮어쓰기")
    parser.add_argument("--system-prompt-file", type=Path, help="설정의 시스템 프롬프트 덮어쓰기")
    parser.add_argument("--max-dimension", type=int, help="리사이즈 후 긴 변 크기")
    parser.add_argument(
        "--no-upscale",
        action="store_true",
        help="긴 변이 max-dimension보다 작은 이미지는 확대하지 않음",
    )
    return parser


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    api_key = os.environ.get("CAPTIONOR_API_KEY")
    if api_key is None:
        api_key = args.api_key if args.api_key is not None else config.api.api_key
    api = ApiSettings(
        base_url=_normalise_base_url(
            args.base_url if args.base_url is not None else config.api.base_url,
            "--base-url",
        ),
        api_key=api_key,
        model=args.model or config.api.model,
        timeout_seconds=config.api.timeout_seconds,
        max_retries=config.api.max_retries,
        retry_delay_seconds=config.api.retry_delay_seconds,
    )
    max_dimension = (
        args.max_dimension if args.max_dimension is not None else config.image.max_dimension
    )
    if max_dimension < 1:
        raise ConfigError("--max-dimension은 1 이상이어야 합니다.")
    resize_mode = config.image.resize_mode
    if args.no_upscale and resize_mode == "exact":
        resize_mode = "shrink"
    image = ImageSettings(
        max_dimension=max_dimension,
        resize_mode=resize_mode,
        resampling=config.image.resampling,
    )
    system_prompt = config.system_prompt
    if args.system_prompt_file is not None:
        try:
            system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ConfigError(
                f"시스템 프롬프트 파일을 읽을 수 없습니다: {args.system_prompt_file}: {exc}"
            ) from exc
        if not system_prompt:
            raise ConfigError("시스템 프롬프트 파일이 비어 있습니다.")
    return AppConfig(
        api=api,
        image=image,
        files=config.files,
        caption=config.caption,
        system_prompt=system_prompt,
        request_options=config.request_options,
    )


def default_output_root(input_path: Path) -> Path:
    return input_path / "captions" if input_path.is_dir() else input_path.parent / "captions"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit은 1 이상이어야 합니다.")

    try:
        config = apply_overrides(load_config(args.config), args)
        input_path = args.input.expanduser().resolve()
        output_root = (args.output_dir or default_output_root(input_path)).expanduser().resolve()
        tag_root = args.tag_dir.expanduser().resolve() if args.tag_dir else None
        stats = run_batch(
            config=config,
            input_path=input_path,
            output_root=output_root,
            tag_root=tag_root,
            recursive=args.recursive,
            overwrite=args.overwrite,
            missing_tags=args.missing_tags,
            dry_run=args.dry_run,
            limit=args.limit,
            fail_fast=args.fail_fast,
        )
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다.", file=sys.stderr)
        return 130
    except ConfigError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2

    print(
        "요약: "
        f"발견 {stats.discovered}, 선택 {stats.selected}, "
        f"처리 {stats.processed}/{stats.selected}, 생성 {stats.generated}, "
        f"기존 출력 건너뜀 {stats.skipped_existing}, "
        f"태그 없음 건너뜀 {stats.skipped_missing_tags}, 실패 {stats.failed}"
    )
    return stats.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
