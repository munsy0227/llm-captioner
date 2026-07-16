# Captionor

`자연어 캡션 뉴뉴 수정.json`의 실제 실행 흐름을 ComfyUI 없이 사용할 수 있도록 옮긴 독립 실행형 Python CLI입니다.

지정한 폴더의 이미지를 순서대로 찾고, 이미지와 같은 stem의 Danbooru 태그 `.txt`를 읽은 뒤, 로컬 OpenAI 호환 비전 API에 이미지와 태그를 보냅니다. 생성된 자연어 캡션은 별도 출력 폴더에 저장합니다.

## 처리 흐름

```text
image.webp + image.txt
        │
        ├─ EXIF 방향 보정 및 RGB 변환
        ├─ 긴 변을 정확히 1024px로 bicubic 리사이즈
        ├─ PNG data URL로 인코딩
        └─ system prompt + 이미지 + 태그를 /v1/chat/completions로 전송
                                      │
                                      └─ captions/image.webp.txt
```

원본 워크플로와 비교하면 다음 동작을 유지합니다.

- API 주소: `http://localhost:8080/v1`
- 모델 ID: `/home/munsy0227/checkpoints/gemma-4-12b-it-Q4_K_M.gguf`
- timeout 600초, 최대 재시도 2회
- 이미지의 긴 변을 항상 1024px로 맞춤. 1024px보다 작은 이미지도 확대
- 이미지 먼저, 태그 텍스트를 나중에 넣는 멀티모달 메시지 구조
- seed, temperature, top-p 등의 생성 옵션과 긴 영문 system prompt
- 입력 태그는 `image.txt`, 출력 캡션은 `image.webp.txt` 형식

다음 부분은 안전하게 개선했습니다.

- 원본의 단순한 `webp` 문자열 치환 대신 모든 이미지 형식에 대해 올바른 stem `.txt` 경로를 계산합니다.
- 원본의 append 저장 대신 기존 출력은 기본적으로 건너뜁니다. `--overwrite`를 준 경우에만 교체합니다.
- 정상적인 비어 있지 않은 응답을 받은 뒤 임시 파일과 `os.replace()`를 사용해 원자적으로 저장합니다.
- 출력 경로가 원본 이미지와 겹치거나 여러 이미지가 같은 출력 경로를 사용하면 작업 시작 전에 중단합니다.
- 한 이미지가 실패해도 다음 이미지를 계속 처리하고 마지막에 실패 개수를 반환합니다.

긴 변의 계산과 bicubic 방식은 원본을 따르지만, 이 프로그램은 의존성을 줄이기 위해 Pillow로 리사이즈합니다. 따라서 ComfyUI의 텐서 기반 `common_upscale`과 결과 픽셀이 비트 단위로 완전히 같지는 않을 수 있습니다.

## 중요한 전제

이 프로그램은 ComfyUI를 전혀 사용하지 않지만, 비전 모델 서버까지 직접 실행하지는 않습니다. 원본 JSON에도 모델 서버의 실행 명령, 멀티모달 projector, context 크기 설정은 들어 있지 않고 `http://localhost:8080/v1`에 접속하는 정보만 들어 있습니다.

따라서 실행 전에 llama.cpp, vLLM, Ollama 등에서 OpenAI 호환 비전 API 서버가 떠 있어야 합니다. 다음 주소가 응답하는지 먼저 확인할 수 있습니다.

```bash
curl http://localhost:8080/v1/models
```

서버가 반환하는 모델 ID가 설정 파일의 `api.model`과 다르면 [captionor_config.json](./captionor_config.json)의 값을 서버 모델 ID로 바꿔야 합니다. GGUF 파일 경로는 이 Python 프로그램이 모델을 직접 로드한다는 뜻이 아니라, API 요청의 `model` 문자열로 전송됩니다.

## 설치

Python 3.10 이상을 권장합니다.

```bash
cd /home/munsy0227/captionor
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

런타임 외부 의존성은 Pillow 하나뿐입니다. HTTP 요청에는 Python 표준 라이브러리를 사용합니다.

## 입력 폴더 준비

각 이미지와 같은 폴더에 같은 stem의 태그 파일을 둡니다.

```text
/path/to/images/
├── 001.webp
├── 001.txt
├── 002.png
└── 002.txt
```

예를 들어 `001.webp`의 입력 태그는 `001.txt`에서 읽습니다. 태그 파일은 기본적으로 UTF-8 또는 UTF-8 BOM 형식으로 읽습니다. `#`으로 시작하는 줄은 무시합니다.

## 실행

가장 기본적인 실행은 다음과 같습니다.

```bash
python3 captionor.py "/path/to/images"
```

출력 폴더를 지정하려면 다음과 같이 실행합니다.

```bash
python3 captionor.py "/path/to/images" \
  --output-dir "/path/to/captions"
```

결과는 다음처럼 생성됩니다.

```text
/path/to/captions/
├── 001.webp.txt
└── 002.png.txt
```

기본 출력 폴더는 입력이 폴더일 때 `<입력 폴더>/captions`입니다. 출력 이름에 원본 이미지 확장자를 남기므로 입력 태그 `.txt`와 충돌하지 않습니다.

### 자주 쓰는 옵션

```bash
# 실제 API 호출 없이 대상 이미지, 태그, 출력 경로 확인
python3 captionor.py "/path/to/images" --dry-run

# 하위 폴더까지 처리하고 출력에도 같은 폴더 구조 유지
python3 captionor.py "/path/to/images" --recursive

# 기존 캡션도 다시 생성하여 덮어쓰기
python3 captionor.py "/path/to/images" --overwrite

# 태그 파일이 없는 이미지는 실패 대신 건너뛰기
python3 captionor.py "/path/to/images" --missing-tags skip

# 앞의 10개만 처리
python3 captionor.py "/path/to/images" --limit 10

# 작은 이미지를 1024px까지 확대하지 않기
python3 captionor.py "/path/to/images" --no-upscale

# 다른 API 주소와 모델 ID 사용
python3 captionor.py "/path/to/images" \
  --base-url "http://localhost:11434/v1" \
  --model "qwen3-vl:8b"
```

별도 폴더에 태그가 있고 이미지 폴더와 같은 하위 구조라면 `--tag-dir`을 사용합니다.

```bash
python3 captionor.py "/path/to/images" \
  --tag-dir "/path/to/tags" \
  --output-dir "/path/to/captions" \
  --recursive
```

API 키는 명령행에 노출하지 않고 환경변수로 지정할 수도 있습니다.

```bash
export CAPTIONOR_API_KEY="your-api-key"
python3 captionor.py "/path/to/images"
```

## 설정 파일

기본 설정은 [captionor_config.json](./captionor_config.json), system prompt는 [system_prompt.txt](./system_prompt.txt)에 있습니다.

주요 설정은 다음과 같습니다.

| 설정 | 기본값 | 의미 |
|---|---:|---|
| `api.base_url` | `http://localhost:8080/v1` | OpenAI 호환 API base URL |
| `api.model` | GGUF 절대 경로 | 요청에 넣는 모델 ID |
| `api.timeout_seconds` | `600` | 요청 제한 시간 |
| `api.max_retries` | `2` | 일시적 오류 뒤 재시도 횟수 |
| `image.max_dimension` | `1024` | 리사이즈한 이미지의 긴 변 |
| `image.resize_mode` | `exact` | `exact`, `shrink`, `none` 중 하나 |
| `files.tag_filename_mode` | `stem` | 입력 태그를 `image.txt`에서 읽음 |
| `files.output_filename_mode` | `image_name` | 출력을 `image.webp.txt`로 저장 |
| `request_options` | 워크플로 값 | API 요청 최상위 생성 옵션 |

`request_options.repetition_penalty`는 원본 워크플로의 필드명을 그대로 보존한 서버 확장 옵션입니다. 사용하는 서버가 `repeat_penalty`만 지원한다면 해당 키 이름을 바꿔야 합니다.

일반적인 `image.txt` 출력명을 원하면 `files.output_filename_mode`을 `stem`으로 바꿀 수 있습니다. 이 경우 입력 태그 폴더와 출력 폴더는 반드시 다르게 두어야 하며, 프로그램도 동일 경로 충돌을 거부합니다.

## 문제 해결

- `API 서버에 연결할 수 없습니다`가 나오면 먼저 8080 포트의 비전 API 서버를 실행합니다.
- `model not found`가 나오면 `/v1/models`의 모델 ID를 `api.model` 또는 `--model`에 사용합니다.
- 알 수 없는 `repetition_penalty` 필드 오류가 나오면 사용하는 서버 문서에 맞춰 `repeat_penalty` 등으로 키를 바꿉니다.
- context 또는 token 제한 오류가 나오면 `request_options.max_tokens`를 줄입니다.

## 테스트

테스트는 실제 모델 없이 로컬 모의 API로 요청 본문, 인증 헤더, 재시도, 이미지 인코딩, 저장 및 재개 동작을 검증합니다.

```bash
python3 -m unittest discover -s tests -v
```

원본 [자연어 캡션 뉴뉴 수정.json](./자연어%20캡션%20뉴뉴%20수정.json)은 참고용으로 그대로 보존되어 있습니다.

변환 동작은 [comfyui-openai-api v2.0.1의 요청 구성](https://github.com/hekmon/comfyui-openai-api/blob/v2.0.1/completions.py), [ComfyUI의 ImageScaleToMaxDimension 구현](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_images.py), [WAS Node Suite의 배치 이미지·텍스트 로더](https://github.com/WASasquatch/was-node-suite-comfyui/blob/ea935d1044ae5a26efa54ebeb18fe9020af49a45/WAS_Node_Suite.py)를 기준으로 확인했습니다.
