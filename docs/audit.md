# 저장소 품질 및 보안 감사

2026년 10월 13일의 로컬 작업 트리를 대상으로 Python 모듈, Vue 프런트엔드,
설정, 의존성, 테스트와 검사 스크립트를 검토했다. 확인된 수치 오류, 입력 처리 오류,
불필요한 대형 메모리 할당을 수정하고 회귀 테스트를 추가했다. 공개 취약점의 등급과
이 애플리케이션에서 실제로 확인된 노출 경로는 아래에서 구분한다.

검증 환경은 Windows, 저장소 `.venv`, PyTorch 2.14.0 CPU 빌드다.
CUDA, 실제 WebGL 렌더링 성능, 장시간 학습의 수렴 및 측정된 3D 정답에 대한
생성 품질은 이번 결과로 보장하지 않는다. 기존 가중치와 실험 결과는 변경하지 않았다.

## 심각도별 수정 결과

### 높은 등급의 의존성 취약점

`npm audit`가 3개 패키지 항목에서 high를 보고했다. 서로 독립적인 취약점은 2건이다.

| 취약점 | 수정 | 이 저장소의 노출 범위 |
| --- | --- | --- |
| Vue server renderer의 SSR XSS | Vue 및 관련 compiler/runtime/renderer 3.5.41 → 3.5.43 | 현재 앱은 클라이언트 렌더링이다. 서버 측 HTML 렌더링을 통한 실제 악용은 확인하지 않았다. |
| source-map-js의 비신뢰 indexed source map 처리 DoS | 1.2.1 → 1.2.2 | 빌드 의존성이다. 앱이 외부 source map을 받아 처리하는 경로는 확인하지 않았다. |

근거는 [Vue advisory](https://github.com/advisories/GHSA-g2v6-rqmx-r4w6),
[source-map-js advisory](https://github.com/advisories/GHSA-68fv-2mgg-jv7q)와
[renderer lock](../frontend/package-lock.json#L1126),
[source-map-js lock](../frontend/package-lock.json#L1878),
[Vue lock](../frontend/package-lock.json#L2037)이다.
`package.json`의 버전 범위 안에서 lock을 갱신했고, 재감사 결과는 0건이다.

### 중간 심각도

| 문제와 재현 조건 | 원인과 수정 | 코드 및 회귀 근거 |
| --- | --- | --- |
| PNG에 선언된 치수보다 압축 해제 결과가 훨씬 크면 끝까지 메모리에 적재 | `Response.arrayBuffer()`로 전량 수집한 뒤 검사하던 처리를, 선언된 행 길이까지만 읽고 초과하면 취소하는 스트림 처리로 변경 | [image-labels.js:57](../frontend/src/image-labels.js#L57), [image-labels.test.js:53](../frontend/test/image-labels.test.js#L53) |
| FP16 채널 정규화가 유한한 큰 입력에서 0으로 붕괴 | `[-1000,1000]`의 분산이 FP16 범위를 넘었다. 평균·분산·정규화를 FP32에서 계산한 뒤 원래 dtype으로 복원 | [denoiser.py:36](../src/model/denoiser.py#L36), [test_models.py:12](../tests/model/test_models.py#L12) |
| 작은 beta의 diffusion posterior가 예상값 약 1 대신 `7.9754e28` 출력 | `alpha_bar`가 1로 반올림되어 차이가 사라졌다. FP64와 `expm1`로 계수를 미리 계산하고, 반정밀도 잡음의 제곱근은 FP32 이상에서 계산 | [diffusion.py:40](../src/model/diffusion.py#L40), [test_diffusion.py:32](../tests/model/test_diffusion.py#L32) |
| preserved base와 일치하는 배치 차원 앵커가 충돌 또는 shape 오류로 거부 | `[1,H,W]` 라벨과 `[1,C,H,W]` 비율을 별도로 잘못 해석했다. 교집합을 자른 뒤 기존 `anchor_probabilities` 변환을 재사용 | [volume_condition.py:182](../src/predict/volume_condition.py#L182), [test_volume_condition.py:13](../tests/predict/test_volume_condition.py#L13) |
| 대형 볼륨의 phase 비율 평가가 전체 `int64` 복사본 생성 | 기존 청크 순회를 재사용해 청크별 histogram을 누적. 비연속 텐서도 전체 flatten 복사를 하지 않음 | [label.py:8](../src/evaluate/label.py#L8), [label.py:80](../src/evaluate/label.py#L80), [test_label.py:98](../tests/evaluate/test_label.py#L98) |
| HR 미리보기에서 단면 3개를 보여주기 위해 HR 크기 전체 nearest 볼륨 생성 | 표시할 축별 단면만 골라 2D로 확대. 비정수·직육면체 크기에서도 기존 3D nearest의 중앙 단면과 동일함을 비교 | [06_check_hr.py:46](../scripts/06_check_hr.py#L46), [test_hr_preview.py:9](../tests/test_hr_preview.py#L9) |

PNG 회귀의 수정 전 재현은 1×1 선언 이미지에서 64바이트짜리 출력 100개를 모두
소비한 후 실패했다. 수정 후에는 끝까지 읽기 전에 취소하며 정상 grayscale 및 indexed
PNG도 유지한다. 이는 사용자가 선택한 이미지로 브라우저 자원을 소모시키는 경로이며,
임의 코드 실행을 입증한 것은 아니다.

diffusion 회귀는 FP32/FP16/BF16에서 검사한다. 독립 검토에서 `sample_pair`의
작은 beta 잡음도 dtype 변환 전에 제곱근을 구해야 함을 추가로 확인해 반영했다.
기본 학습 설정에서 이 극단적인 beta 문제가 발생했다고 주장하지 않는다.
파생 diffusion buffer는 비영속으로 두었고, 프로젝트의 LR/SR 로더는 저장된
설정에서 diffusion을 재조립한다. 기존 모델 및 학습 체크포인트 키를 보존했다.

phase 비율의 기본 청크 상한은 1,048,576개 원소다. 예를 들어 `1024³` uint8
볼륨을 한꺼번에 int64로 바꾸면 그 복사본만 8 GiB지만, 수정 후 이 변환은 청크당
최대 8 MiB다. 이는 변환 버퍼의 계산상 크기이며 전체 프로세스 peak RAM 실측은 아니다.
HR 미리보기의 추가 보간 메모리는 전체 부피 대신 표시할 단면 크기에 비례한다.

Python 의존성 감사에서도 Windows 관련 moderate 취약점 2건을 수정했다.

| 패키지 | 발견 및 수정 | 적용 범위 |
| --- | --- | --- |
| Werkzeug | 3.1.8 → 3.1.9; Windows 특수 장치명 처리로 파일 읽기가 멈출 수 있음 | TensorBoard의 전이 의존성. [requirements.txt:12](../requirements.txt#L12)에 안전한 최소 버전을 명시. FastAPI 앱이 `send_from_directory`를 호출하는 경로는 없음. |
| uv | `.venv`의 0.12.17 → 0.13.0; 악성 wheel의 Windows 경로 이탈 쓰기 | 설치 도구 환경의 문제. 프로젝트 런타임 의존성으로 추가하지 않았으며 전역 설치는 변경하지 않음. |

공식 근거는 [Werkzeug advisory](https://github.com/pallets/werkzeug/security/advisories/GHSA-g6x2-hccm-hh4m)와
[uv advisory](https://github.com/astral-sh/uv/security/advisories/GHSA-2cv4-cqwr-gwf7)다.
기존 PyTorch 빌드는 교체하지 않았다.

### 낮은 심각도

| 문제 | 수정 및 근거 |
| --- | --- |
| `vf=(1e308,1e308,1e308)`처럼 각 항은 유한하지만 합이 무한대인 경우 정규화 결과가 모두 0 | 최대값으로 먼저 스케일링한 뒤 합으로 나눔. [generator.py:227](../src/predict/generator.py#L227), [test_generate.py:372](../tests/predict/test_generate.py#L372) |
| HTTP seed 스키마가 uint64 범위 밖 값이나 bool 등 변환 가능한 입력을 허용 | 엄격한 uint64 정수 검증으로 추론 전 차단. [schema.py:72](../backend/src/schema.py#L72), [test_app.py:153](../tests/backend/test_app.py#L153) |

큰 seed에 대해 기존 서버도 PyTorch의 `ValueError`를 잡아 422를 반환했다.
따라서 seed 수정은 기존 500 오류의 수정이 아니라 조기 입력 검증이다.

## 모듈 구조와 의존성

초기 추적 대상 Python 파일 198개에 대해 AST 구문 및 내부 절대 import 관계를
조사했다. 그 그래프에서는 순환 import를 발견하지 않았다. 동적 import와 검사
스크립트의 `sys.path` 조작까지 포함한 모든 런타임 의존성을 증명하는 검사는 아니다.

| 영역 | 구성과 책임 | 주요 연결 및 검토 결과 |
| --- | --- | --- |
| 학습 CLI | `run_train_1st.py`, `run_train_2nd.py` | 인수 → `train/run/low_res.py`, `sr.py`. `--steps`는 총 목표 단계이며 저장 설정을 통한 재개 경로 유지. |
| `src/config` | `schema`, `defaults`, `values`, `files`, `data`, `train`, `generation` | YAML → 검증·정규화 → build/API. LR/SR, 생성 설정, source 경로와 numeric 경계 검사. |
| `src/data` | `source`, `provenance`, `dataset`, `loader`, `split`, `augment`, `slice`, `bank` | 파일 발견·지문 → 라벨 crop/좌표 → 배치·평면 샘플·SR bank. 데이터 dict와 높이 단위, 분할 배제, 비율 보존 확인. |
| `src/prepare` | `resize`, `height`, `profile` | 라벨 → phase 채널·보존적 비율 resize, 높이·프로필 조건. 라벨 보간과 비율 보간의 경계 유지. |
| `src/model` | `layers`, `pyramid`, `denoiser`, `critic`, `diffusion` | 조건 임베딩 → 3D denoiser 및 2D critic. 채널 정규화와 diffusion 수치 오류 수정. |
| `src/build` | `data`, `model`, `trainer`, `sr`, `predict` | 설정을 데이터·모델·트레이너/추론기로 조립. 학습/HTTP 핸들러에 모델 로직을 복제하지 않음. |
| `src/train` | `trainer`, `step`, `batch`, `conditioning`, `coarse`, `anchor_bank`, `ema`, `metrics` | 조건·replay → critic/generator update → EMA·관측값. measured 2D와 generated replay의 역할을 구분. |
| `src/train/loss` | `gan`, `denoiser`, `sr`, `anchor`, `anchor_continuity`, `anchor_neighborhood`, `transition`, `volume_fraction`, `spatial_profile`, `group_statistics` | adversarial 및 통계·앵커·LR/SR 일관성 목적함수. gradient와 통계 집계 경계를 기존 테스트로 검증. |
| 학습 수명주기 | `train/run/{loop,low_res,sr,source,bank}`, `checkpoint`, `state`, `relocate` | 로그·완료 단계·checkpoint·고정 LR source 검증. source hash, 데이터 fingerprint, 아카이브 보존과 재개 경계 확인. |
| 공통 계약 | `api`, `anchor`, `plane`, `phase`, `random`, `storage` | 공개 LR/SR/앵커 API, `(z,y,x)`와 평면 규약, RNG 범위, TIFF/텐서 저장. `weights_only=True`, strict 모델 로딩, 저장 실패 처리를 확인. |
| LR 추론 | `predict/{inference,generator,volume_condition,convert,memory}` | 요청 → 조건 검증 → 직접/타일 생성 → CPU 결과. VF 정규화와 preserved base 앵커 검증 수정. |
| 타일링 | `predict/tiling/{layout,state,sampler,generator,fusion}` | tile plan, global state, 순환 z slab 합성, 청크 CPU 출력. H/W 전체 폭을 갖는 buffer와 메모리 예산 확인. |
| SR 추론 | `predict/sr/{inference,tiled,memory,extension}` | LR 비율 → coarse 지역 보간 → 공통 sampler, HR 확장·보존. 정수 scale 및 좌표 정렬, 비정수 전체 추론 경로 확인. |
| 평가 | `evaluate/{label,image,volume,connectivity,structure,seam,anchor,profile,fid,kid,tortuosity}` | 라벨 통계·연결성·경계·프로필, Inception 기반 2D 지표와 수송 solver. phase 비율 메모리 수정; 서로 다른 지표를 3D 정답 검증으로 해석하지 않음. |
| `backend` | `run`, `config`, `schema`, `middleware`, `app`, `response` | 요청 크기·스키마 → 메모리 preflight·동시성 슬롯 → raw/TIFF 응답 및 정리. `X-Volume-Shape`와 uint8 byte 길이 계약 유지. |
| `frontend` | `App`, `Sidebar`, `CropEditor`, `VolumeViewer`, `api`, `input-state`, `use-generation`, `image-labels`, `volume-data`, Vite proxy | 라벨 이미지·crop → 요청 gate/취소 → raw 검증 → VTK. PNG 제한 수정; 요청 경쟁, Z 좌표, VTK 자원 해제 테스트 확인. |
| `simul` | `run`, `geometry`, `export` | geometry 검증 → 입자 배치 → 3D TIFF/축별 palette PNG. 기존 비어 있지 않은 출력 디렉터리 보호 확인. |
| `scripts/common` 및 01–06 | CLI·진단·TensorBoard 집계와 데이터/생성/앵커/확장/SR 검사 | 공개 API 및 공통 변환 재사용. HR 미리보기 메모리 수정. headless 검사와 저장 산출물 경로 확인. |
| `scripts/experiments` | `training_stability`, `critic_inputs`, `prediction_combinations` | 실험 변형·재개·조건 조합 검사. 짧은 실행 증거와 생성 품질 증거의 구분 유지. |
| `scripts/paper` | `provenance`, `evaluation_io`, `evaluate_*`, `make_*` | 입력 hash/manifest, 경로 충돌 확인, 구조·연장·앵커 평가와 그림 생성. 기존 연구 결과를 수정하지 않음. |
| 테스트·구성·문서 | `tests`, `frontend/test`, `config`, `pyproject.toml`, manifest, README 및 docs | 기존 사용법과 호출자, 테스트 fixture, Ruff 규칙을 함께 확인. 설정 프리셋이나 실험 기대값을 통과 목적으로 바꾸지 않음. |

핵심 외부 의존성은 수치·학습의 PyTorch/NumPy/SciPy, 라벨 I/O의 Pillow/tifffile,
서버의 FastAPI/Starlette/AnyIO/Uvicorn, 평가의 torchmetrics/torch-fidelity/TauFactor,
로그·GUI의 TensorBoard/Matplotlib/napari, 웹의 Vue/vtk.js/Vite다.
선택적인 지표·뷰어가 필요 없는 경로에서는 해당 무거운 import를 피하는 기존 경계를 유지했다.

## 코드 품질 및 남은 성능 관찰

기존 모듈 경계를 유지한 국소 수정이다. 공통 앵커 변환과 라벨 청크 처리를 재사용했고,
저장 가중치 구조, phase 비율, 좌표와 raw HTTP 응답 형식은 유지했다.
`normalize_train_config`, `validate_training_values`, `estimate_sr_memory`,
`Denoiser3D.compute_logits`는 분기 수가 많은 핵심 지점이지만, 크기만으로 결함을
판정하거나 추상화를 추가하지 않았다.

아래 항목은 코드에서 확인한 비용 또는 운영상 제약이며, 이번에 실측으로 순위를
확정한 병목이나 재현된 오류로 분류하지 않았다.

- [dataset.py:50](../src/data/dataset.py#L50): crop마다 원본 이미지를 다시 디코딩한다.
  캐시는 I/O를 줄일 수 있지만 메모리·worker 수·데이터 변경 감지와 함께 판단해야 한다.
- [train/run/bank.py](../src/train/run/bank.py): CPU 샘플 리스트와 `torch.stack` 결과가
  겹치는 순간 bank 저장량의 약 두 배가 필요하다. `generate_probs()`가 CPU로 반환하므로
  bank 전체가 GPU에 누적된다는 의심은 배제했다.
- [anchor_neighborhood.py](../src/train/loss/anchor_neighborhood.py): 평면·거리·영역을
  순회하는 손실의 비용은 실제 설정별 profile이 필요하다.
- [06_check_hr.py](../scripts/06_check_hr.py): HR coarse-error 계산에는 여전히 전체
  phase 채널 변환이 있다. 미리보기 개선이 모든 진단 메모리를 제한하는 것은 아니다.
- [VolumeViewer.vue:57](../frontend/src/components/VolumeViewer.vue#L57): 고정 ray sampling과
  [api.js:43](../frontend/src/api.js#L43)의 항상 요청하는 metrics는 큰 볼륨의 응답 시간을
  늘릴 수 있다. VTK 청크는 빌드 결과 약 910 kB, gzip 약 243 kB이며 지연 로딩된다.
- [state.py:73](../src/train/state.py#L73): 학습 재개는 모델·optimizer·scaler·진행 상태를
  복구하지만 전체 RNG 및 DataLoader 진행 위치를 저장하지 않는다. 중단 없는 학습과의
  bitwise 동일 궤적은 현재 계약이 아니다.
- [requirements.txt](../requirements.txt)는 대부분 버전을 고정하지 않아 전체 환경을
  재현하는 lock이 아니다. CUDA별 PyTorch 배포를 임의로 고정하지 않았다.

## 검증과 한계

코드를 고정한 뒤 실행한 최종 전체 테스트는 실패 없이 완료됐다.

| 검증 | 최종 결과 |
| --- | --- |
| 전체 Python pytest | **1,353 통과, 16 건너뜀, 9 subtests 통과**, 331.34초. 건너뛴 16개는 모두 CUDA 부재 때문 |
| 오류 경로 및 메모리 회귀 재검증 | 63 통과, CUDA 1 건너뜀. 학습 CLI의 기존 60초 제한 유지 |
| 프런트엔드 `npm test` | 30 통과 |
| 프런트엔드 `npm run build` | 성공, 235개 모듈. VTK 청크 크기 경고는 남음 |
| 전체 Python 대상 Ruff | 통과 |
| `git diff --check` | 통과 |
| `pip check` | 의존성 충돌 없음 |
| `npm audit` | 알려진 취약점 0건 |
| `.venv`의 `pip-audit` | 177개 패키지, 건너뜀 0개, 알려진 취약점 0건 |

전체 Python 검증 명령은 저장소 루트에서 다음과 같다. 스레드 수 제한은 이 호스트의
메모리 경합을 줄이기 위한 설정이며 테스트 timeout이나 기대값을 변경하지 않는다.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& .\.venv\Scripts\python.exe -m pytest -q -ra --durations=10 -o faulthandler_timeout=300
& .\.venv\Scripts\python.exe -m ruff check src backend scripts simul tests run_train_1st.py run_train_2nd.py
```

[최종 pytest 로그](../run/checks/audit-20261013-01a133bb/diffusion-gan3d-verified-pytest.log),
[오류 경로 재검증](../run/checks/audit-20261013-01a133bb/diffusion-gan3d-recheck.log),
[Python 의존성 감사 결과](../run/checks/audit-20261013-01a133bb/diffusion-gan3d-final-pip-audit.json)를
로컬 `run/checks/audit-20261013-01a133bb/`에 보존했다. 이 디렉터리는 기존 규칙대로
Git 추적 대상이 아니다. 같은 디렉터리의 `diffusion-gan3d-final-pytest.log`는
아래에 설명한 **첫 통합 실행의 실패 기록**이며 최종 통과 로그와 구분해야 한다.

pytest 경고 5건은 Starlette/httpx 및 AnyIO의 폐기 예정 API 경고 2건과
KID의 feature buffer 메모리 경고 3건이다. 가장 오래 걸린 테스트는 CPU에서
4배 SR 타일 생성·coverage를 검사한 95.56초의 테스트다. 이는 이 호스트의 테스트
측정값이며 운영 워크로드의 병목 순위나 GPU 성능 측정값은 아니다.

학습·데이터, 추론, 웹·보안 영역은 병렬 에이전트가 분석했고, 별도 검토자가
최종 diff의 정확성과 회귀 가능성을 독립적으로 확인했다. 보고서도 별도로 검토해
코드·테스트 연결과 검증 범위가 일치하는지 확인했다. 독립 정적 검토와 위의 실행
검증 결과는 서로 구분한다.

수정 전 실패를 실제로 확인한 항목은 preserved base 앵커(5 실패), 큰 VF(1 실패),
phase 비율 전체 할당(2 실패), HR 미리보기 전체 할당(1 실패)이다.
정규화와 diffusion은 작은 수치 재현으로 원인을 확인했다. PNG는 원본 HEAD의
스트림 소비 횟수와 취소 여부를 비교했다. 이후 회귀 테스트를 통해 수정 결과를 검사한다.

검증 중 오류와 조치:

- 처음의 Python 감사는 `--disable-pip`를 설치된 환경 감사에 사용해 인수 오류로
  실패했다. 해당 옵션을 제거하고 감사한 뒤, 패키지를 수정하고 다시 감사했다.
- 병렬 PyTorch 테스트 실행 중 실제 가용 RAM 약 243 MB, RAM 사용률 97.1%,
  CPU 100%, 초당 page-in 약 5,029를 관측했다. 중복 baseline만 종료하고 단일 전체
  테스트로 전환했다. 중단된 실행은 통과 결과로 계산하지 않는다.
- 독립 정적 검토에서 작은 beta의 `sample_pair` 반정밀도 경로 누락을 발견해 수정했다.
  검토자가 테스트를 실행했다고 간주하지 않으며, 실행 결과는 별도로 기록한다.
- 첫 통합 실행은 1,351 통과, 2 실패, 16 건너뜀으로 끝났다. 수정 전에 import된
  diffusion의 FP16 Markov 잡음 테스트와 자원 부족 시 학습 CLI 하위 프로세스의
  60초 timeout이 실패 원인이었다. 현재 코드의 해당 경로와 메모리 회귀를 다시 실행해
  63 통과, CUDA 1 건너뜀을 확인했다. timeout이나 기대값은 완화하지 않았다.

의존성 감사 0건은 조회 시점의 알려진 취약점 데이터베이스 결과다. 새로운 취약점,
인증 없는 도구의 공개망 운영, 실제 GPU 메모리 사용이나 학습 수렴까지 증명하지 않는다.
