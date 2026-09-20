# 수동 확인 스크립트

프로젝트 루트에서 실행합니다. 보통 `--weight`만 지정하면 됩니다.
`generator.pt` 파일과 그 파일이 들어 있는 실행 폴더를 모두 받습니다.

| 번호 | 확인할 것 | 웨이트 |
|---|---|---|
| 01 | 실제 학습 데이터의 crop | LR 또는 SR. 생략하면 현재 LR 학습 설정 사용 |
| 02 | 기본 LR 생성 결과 | LR |
| 03 | 여러 앵커 평면의 영향 | LR |
| 04 | 큰 영역을 타일로 생성한 결과와 이음새 | LR |
| 05 | 한쪽 경계 평면에서 이어지는 단면 | LR |
| 06 | LR·최근접 확대·SR 결과 비교 | SR. LR 웨이트는 저장된 학습 설정에서 찾음 |

```powershell
.venv\Scripts\python.exe scripts/01_check_dataset.py --weight "run/my-lr-run"
.venv\Scripts\python.exe scripts/02_check_generated.py --weight "run/my-lr-run"
.venv\Scripts\python.exe scripts/03_check_anchor.py --weight "run/my-lr-run"
.venv\Scripts\python.exe scripts/04_check_scale_up.py --weight "run/my-lr-run"
.venv\Scripts\python.exe scripts/05_check_continuation.py --weight "run/my-lr-run"
.venv\Scripts\python.exe scripts/06_check_hr.py --weight "run/my-sr-run"
```

기본 실행은 결과를 저장하고 비교 화면을 엽니다. 저장 위치는 실행할 때마다
`run/checks/<시간>_<스크립트 이름>/`에 새로 만들어 터미널에 표시합니다.

- `01`: 학습 crop 미리보기 PNG.
- `02~05`: `volume.tiff`, xy/xz/yz 미리보기 `volume.png`, 실행 옵션 `volume.json`.
  `05`는 경계에서 멀어지는 단면 비교 PNG도 저장합니다.
- `06`: `lr.tiff`, `lr_probs.pt`(fraction 입력일 때), `hr.tiff`, `comparison.png`, `report.json`.

필요할 때만 아래 옵션을 추가합니다.

| 옵션 | 사용법 |
|---|---|
| `--no-view` | 창을 열지 않고 저장만 하기 |
| `--napari` | 2D 비교 대신 3D로 보기. `02~06`에서 사용 |
| `--device cpu` | GPU 없이 실행. `02~06`에서 사용 |
| `--seed 1` | 다른 샘플 확인. 기본값은 `0` |
| `--domain 1` | 다른 학습 도메인 선택. 기본값은 `0` |
| `--out 경로` | `01`은 PNG, `02~05`는 TIFF, `06`은 결과 폴더 지정 |
| `--help` | 해당 스크립트의 전체 옵션과 실행 예시 보기 |

`03`과 `05`는 실제 3D 정답 대신 생성한 참조 볼륨에서 앵커를 가져옵니다.
실제 경계 이미지를 확인하려면 `05`에 `--anchor image.png`를 추가합니다.
`04`는 기본 2×2×2 타일입니다. 크기를 바꾸려면 `--blocks D H W`를 지정합니다.

`02~05`의 guidance 기본값은 `config/gen.yaml`에서 읽습니다.
`06`의 SR guidance 기본값은 `1.0`이고, LR guidance는 `config/gen.yaml`을 따릅니다.
SR에 기록된 LR 파일을 옮겼다면 `06`에 `--lr-weight "새 LR 경로"`를 추가합니다.

`scripts/common/`은 공통 코드, `scripts/experiments/`는 학습·추론 조합 실험,
`scripts/paper/`는 논문 재현용입니다. 평소 결과 확인은 위 번호 스크립트를 사용하면 됩니다.
