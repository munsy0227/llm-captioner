# Captionor

`자연어 캡션 뉴뉴 수정.json`의 실제 실행 흐름을 ComfyUI 없이 사용할 수 있도록 옮긴 독립 실행형 Python CLI입니다.

지정한 폴더의 이미지를 순서대로 찾고, 이미지와 같은 stem의 입력 태그 `.txt`를 읽은 뒤, 로컬 OpenAI 호환 비전 API에 이미지와 태그를 보냅니다. JPEG XL(`.jxl`)을 포함한 설정된 이미지 형식을 처리하며, 생성된 자연어 캡션은 별도 출력 폴더에 저장합니다.

## 처리 흐름

```text
image.webp + image.txt
        │
        ├─ 현재 태그와 파일 크기 표시
        ├─ EXIF 방향 보정 및 RGB 변환
        ├─ 긴 변을 정확히 1024px로 bicubic 리사이즈
        ├─ PNG data URL로 인코딩
        └─ system prompt + 이미지 + 태그를 /v1/chat/completions로 전송
                                      │
                                      ├─ 캡션이 2,048바이트 초과 시 Gemma에 재요청
                                      └─ 통과한 캡션을 captions/image.webp.txt로 저장
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
- 원본의 append 저장 대신 기존 출력은 기본적으로 건너뛰고, `--overwrite`를 지정한 경우에만 교체합니다.
- 정상적인 비어 있지 않은 응답을 받은 뒤 임시 파일과 `os.replace()`를 사용해 원자적으로 저장합니다.
- 출력 경로가 원본 이미지와 겹치거나 여러 이미지가 같은 출력 경로를 사용하면 작업 시작 전에 중단합니다.
- 한 이미지가 실패해도 다음 이미지를 계속 처리하고 마지막에 실패 개수를 반환합니다.
- 매 항목마다 현재 태그, 완료 개수, 진행률, 경과 시간, 예상 총 시간과 예상 남은 시간을 표시합니다.
- 실행 시간을 제한하면 완료 위치를 진행 상태 파일에 기록하고 다음 실행에서 남은 항목을 이어서 처리합니다.
- Gemma가 생성한 캡션의 저장 크기가 2,048바이트를 초과하면 같은 이미지와 태그로 캡션을 다시 생성합니다.

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

이미지 처리는 Pillow를 사용하고, JPEG XL 디코딩은 [`pillow-jxl-plugin`](https://github.com/Isotr0py/pillow-jpegxl-plugin)이 Pillow에 등록합니다. `requirements.txt`를 설치하면 두 패키지가 함께 설치됩니다. HTTP 요청에는 Python 표준 라이브러리를 사용합니다.

## 입력 폴더 준비

각 이미지와 같은 폴더에 같은 stem의 태그 파일을 둡니다.

```text
/path/to/images/
├── 001.webp
├── 001.txt
├── 002.png
├── 002.txt
├── 003.jxl
└── 003.txt
```

예를 들어 `001.webp`의 입력 태그는 `001.txt`에서 읽습니다. 태그 파일은 기본적으로 UTF-8 또는 UTF-8 BOM 형식으로 읽습니다. `#`으로 시작하는 줄은 무시합니다.

처리할 때는 읽을 수 있는 현재 태그 내용과 원본 태그 파일의 바이트 크기가 터미널에 표시됩니다. 제어 문자가 터미널 동작을 바꾸지 않도록 내용은 JSON 문자열처럼 따옴표와 이스케이프를 사용하며, 매우 긴 내용은 앞 2,048문자까지만 보여줍니다.

## 진행률과 예상 시간

각 이미지 처리가 끝날 때 다음과 같은 상태가 표시됩니다.

```text
진행률: 12/100 (12.0%) | 경과 00:04:30 | 예상 총 00:37:30 | 예상 남은 시간 00:33:00
```

예상 시간은 지금까지 처리가 끝난 항목의 평균 시간을 기준으로 계산합니다. 첫 항목이 끝나기 전에는 계산 중으로 표시되며, API 응답 속도나 긴 캡션 재생성 여부에 따라 계속 보정됩니다.

## 시간 제한 실행과 이어서 처리

`--time-limit`으로 한 번의 실행에서 캡션을 생성할 시간을 정할 수 있습니다. 숫자만 지정하면 초 단위이며, `s`, `m`, `h` 단위와 복합 형식을 사용할 수 있습니다.

```bash
# 30분 동안 처리
python3 captionor.py "/path/to/images" --time-limit 30m

# 다음 실행에서 남은 항목을 다시 30분 동안 처리
python3 captionor.py "/path/to/images" --time-limit 30m

# 1시간 30분, 2시간, 1800초도 지정 가능
python3 captionor.py "/path/to/images" --time-limit 1h30m
python3 captionor.py "/path/to/images" --time-limit 2h
python3 captionor.py "/path/to/images" --time-limit 1800
```

기본 진행 상태 파일은 출력 폴더의 `.captionor-progress.json`입니다. 사람이 읽을 수 있는 JSON 형식이며 캡션 파일과 마찬가지로 임시 파일을 거쳐 원자적으로 갱신됩니다. 진행 상태가 있으면 다음 실행에서 `--time-limit`을 생략해도 자동으로 읽고 남은 항목을 이어서 처리합니다.

안전한 재개를 위해 입력·출력·태그 경로, 재귀 처리 여부, 파일명 모드와 함께 API 주소·모델, system prompt, 생성 옵션, 이미지 전처리, 캡션 제한 설정의 지문을 기록합니다. 이 조건이 달라지면 서로 다른 설정의 캡션이 한 작업에 섞이지 않도록 재개를 거부합니다. 설정을 바꿔 새로 생성하려는 경우 `--reset-progress`를 사용하세요.

```bash
# 이전 진행 상태를 읽고 시간 제한 없이 나머지를 끝까지 처리
python3 captionor.py "/path/to/images"

# 설정 파일의 session.time_limit_seconds를 이번 실행에서만 해제
python3 captionor.py "/path/to/images" --no-time-limit

# 진행 상태 파일을 다른 위치에 저장하고 이후에도 같은 파일로 이어서 처리
python3 captionor.py "/path/to/images" \
  --time-limit 30m \
  --progress-file "/path/to/state/my-caption-progress.json"
```

시간 제한은 새 항목을 시작하기 직전에 확인합니다. 제한 시간에 도달했더라도 이미 시작한 이미지의 API 요청과 안전한 파일 저장은 끝낸 뒤 중단하므로 실제 실행 시간은 지정한 값보다 길어질 수 있습니다. 진행 중인 항목이 정상적으로 저장된 뒤에만 완료 위치가 갱신됩니다.

API 오류나 태그 누락으로 캡션을 만들지 못한 항목은 완료로 기록하지 않으며 다음 실행에서도 처리 대상으로 남습니다. 반대로 캡션을 생성했거나 비어 있지 않은 기존 출력이 확인된 항목은 완료로 기록합니다. 진행 상태에는 완료로 남아 있어도 해당 출력 파일이 사라졌거나 비어 있으면 다시 처리합니다.

`--overwrite`를 사용한 실행도 진행 상태에 완료로 기록된 항목은 다시 덮어쓰지 않고 그 다음 항목부터 이어집니다. 모든 항목을 처음부터 다시 생성하려면 `--reset-progress`를 함께 사용합니다. `--reset-progress`는 진행 상태만 초기화하며 기존 캡션 파일 자체를 삭제하지 않습니다. 지정한 기존 파일이 Captionor 진행 상태로 확인되지 않거나 JSON이 손상되어 있으면 임의 파일을 지우지 않도록 초기화를 거부하므로, 파일을 직접 확인해 옮기거나 다른 `--progress-file`을 지정하세요.

```bash
python3 captionor.py "/path/to/images" --overwrite --reset-progress
```

`--dry-run`은 기존 진행 상태를 바탕으로 작업 예정 목록을 보여줄 수 있지만 진행 상태 파일을 만들거나 갱신하거나 삭제하지 않습니다.

같은 진행 상태 파일을 사용하는 Captionor 프로세스를 동시에 실행하면 중복 생성이나 진행 기록 유실이 생길 수 있으므로 한 번에 하나만 실행하세요. `--limit`을 함께 쓸 때 맨 앞의 실패·태그 누락 항목이 계속 남아 있으면 다음 실행에서도 먼저 선택되므로, 태그나 오류를 해결하거나 제한값을 늘려야 뒤 항목도 처리됩니다.

## 2KB 초과 Gemma 캡션 자동 재생성

크기 제한 대상은 입력 태그 파일이 아니라 **Gemma API가 새로 생성한 자연어 캡션**입니다. 캡션과 마지막 줄바꿈을 UTF-8로 인코딩한 실제 저장 크기가 2,048바이트 이하면 저장하고, 2,049바이트부터 해당 결과를 버리고 Gemma에 새 캡션을 요청합니다.

재요청에서도 원래 이미지, system prompt, 입력 태그를 그대로 사용합니다. 고정 seed로 같은 장문이 반복되는 것을 줄이기 위해 다음 시도의 seed를 변경하고, 원래 태그와 분리된 추가 지시문으로 더 짧은 캡션을 요청합니다. 입력 태그 파일은 크기와 관계없이 읽기 전용이며 수정하거나 백업하지 않습니다.

- 기본값은 최초 요청을 포함해 총 2회의 캡션 생성 시도입니다. 각 생성 시도 안의 일시적 HTTP 오류 재시도는 `api.max_retries`에 따라 별도로 수행됩니다.
- 각 Gemma 응답 내용과 UTF-8 저장 바이트 크기를 터미널에 표시합니다. 긴 응답은 앞 2,048문자까지만 표시합니다.
- 제한을 통과한 결과가 생긴 경우에만 출력 파일을 원자적으로 저장합니다.
- 모든 생성 결과가 2,048바이트를 초과하거나 후속 API 요청이 실패하면 해당 이미지를 실패 처리합니다. `--overwrite`를 사용했더라도 기존 캡션은 그대로 보존됩니다.
- `--dry-run`은 API를 호출하지 않으므로 캡션 크기를 검사하지 않고 생성 예정 작업만 보여줍니다.

같은 폴더에 `image.jpg`와 `image.png`처럼 stem이 같은 이미지가 있으면 기본 `stem` 모드에서 하나의 `image.txt`를 읽기 전용으로 공유합니다. 각각 `image.jpg.txt`, `image.png.txt`를 사용하려면 `files.tag_filename_mode`을 `image_name`으로 바꿉니다.

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
├── 002.png.txt
└── 003.jxl.txt
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

# 30분 동안 처리하고 다음 실행에서 자동으로 이어서 처리
python3 captionor.py "/path/to/images" --time-limit 30m

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
| `files.image_extensions` | JPG, PNG, WebP, BMP, GIF, TIFF, JXL | 처리할 이미지 확장자 |
| `files.tag_filename_mode` | `stem` | 입력 태그를 `image.txt`에서 읽음 |
| `files.output_filename_mode` | `image_name` | 출력을 `image.webp.txt`로 저장 |
| `caption.max_output_bytes` | `2048` | 저장할 Gemma 캡션의 UTF-8 최대 바이트 수 |
| `caption.max_attempts` | `2` | 크기 초과 시 최초 요청을 포함한 최대 캡션 생성 횟수 |
| `session.time_limit_seconds` | `0` | 한 실행의 시간 제한(초). `0`이면 제한 없음 |
| `session.progress_filename` | `.captionor-progress.json` | 출력 폴더에 둘 기본 진행 상태 파일명 |
| `request_options` | 워크플로 값 | API 요청 최상위 생성 옵션 |

`request_options.repetition_penalty`는 원본 워크플로의 필드명을 그대로 보존한 서버 확장 옵션입니다. 사용하는 서버가 `repeat_penalty`만 지원한다면 해당 키 이름을 바꿔야 합니다.

일반적인 `image.txt` 출력명을 원하면 `files.output_filename_mode`을 `stem`으로 바꿀 수 있습니다. 이 경우 입력 태그 폴더와 출력 폴더는 반드시 다르게 두어야 하며, 프로그램도 동일 경로 충돌을 거부합니다.

## 문제 해결

- `API 서버에 연결할 수 없습니다`가 나오면 먼저 8080 포트의 비전 API 서버를 실행합니다.
- `model not found`가 나오면 `/v1/models`의 모델 ID를 `api.model` 또는 `--model`에 사용합니다.
- 알 수 없는 `repetition_penalty` 필드 오류가 나오면 사용하는 서버 문서에 맞춰 `repeat_penalty` 등으로 키를 바꿉니다.
- context 또는 token 제한 오류가 나오면 `request_options.max_tokens`를 줄입니다.
- JPEG XL 파일에서 `pillow-jxl-plugin` 설치 안내가 나오면 가상환경을 활성화한 뒤 `python3 -m pip install -r requirements.txt`를 다시 실행합니다.
- Gemma 캡션이 계속 2,048바이트를 넘으면 `caption.max_attempts`를 늘리거나 system prompt에서 원하는 캡션 길이를 더 명확히 지정합니다.
- 다른 `--progress-file`을 사용했던 작업을 이어가려면 이전과 같은 경로를 다시 지정합니다. 기본 경로로 돌아가면 해당 출력 폴더의 `.captionor-progress.json`을 사용합니다.
- 진행 상태를 무시하고 처음부터 새 작업으로 시작하려면 `--reset-progress`를 지정합니다. 기존 캡션까지 다시 생성하려면 `--overwrite`도 함께 지정해야 합니다.

## 테스트

테스트는 실제 모델 없이 로컬 모의 API로 요청 본문, 인증 헤더, HTTP 재시도, 진행률·예상 시간, 시간 제한과 진행 상태 재개, Gemma 캡션의 2KB 경계와 UTF-8 바이트 판정, 입력 태그 불변성, JPEG XL을 포함한 이미지 인코딩과 원자 저장을 검증합니다.

```bash
python3 -m unittest discover -s tests -v
```

원본 [자연어 캡션 뉴뉴 수정.json](./자연어%20캡션%20뉴뉴%20수정.json)은 참고용으로 그대로 보존되어 있습니다.

변환 동작은 [comfyui-openai-api v2.0.1의 요청 구성](https://github.com/hekmon/comfyui-openai-api/blob/v2.0.1/completions.py), [ComfyUI의 ImageScaleToMaxDimension 구현](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_images.py), [WAS Node Suite의 배치 이미지·텍스트 로더](https://github.com/WASasquatch/was-node-suite-comfyui/blob/ea935d1044ae5a26efa54ebeb18fe9020af49a45/WAS_Node_Suite.py)를 기준으로 확인했습니다.
