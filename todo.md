# TODO

## 1. 재학습 (LR → SR)

- [ ] default 샘플을 LR부터 SR까지 끝까지 학습한다 (`commands/train_default.command`).
  - 이번 학습에 처음 적용되는 것: critic `input_mode: pair` 기본값, 새 LR 선택 기준(구조 오차 최저, cluster error 포함), LR 10000 step, burn-in 스케줄(학습률·β2·R1/R2·EMA, LR 5000 / SR 2500 step), SR group_statistics 램프 2500.
  - 지난 run(`run/20261027-0703-default-lr`)은 single critic, 고정 학습률로 10269 step에서 멈췄다.

## 2. 학습 결과 확인

- [ ] `selection/scores.yaml`에서 best step 주변의 `two_point_mae`, `chord_tv`, `cluster_error` 추이를 본다.
  - 지난 run에서는 cluster error가 step 4500 이후 0.13~0.17에서 멈췄고, two-point 오차는 계속 커졌다.
- [ ] 지난 run과 같은 지표로 비교한다. 비교 대상: 지난 run의 step 4500(structure 1.96)과 10269(2.59).
  - pair critic과 burn-in이 함께 바뀌었으니, 어느 쪽 효과인지 나누려면 burn-in을 끈 학습(`burn_in`: `lr_scale: 1.0`, `beta2: 0.9`, `regularization_scale: 1.0`)을 하나 더 돌린다.
- [ ] 학습 후반에 two-point 오차가 커지는 원인을 확인한다. critic이 학습 데이터를 외우는 것으로 의심된다.

## 3. SR 추론 타일 크기 (SR 모델이 생긴 뒤)

- [ ] 같은 LR 입력으로 SR 타일 64(현재 기본값)와 128(학습 크기)의 결과를 비교한다.
  - SR은 HR 128³로 학습하는데, 추론 기본 타일은 HR 64(여백 포함 80)다(`src/predict/sr/inference.py` `default_tiling`).
  - 차이가 있으면 기본 타일을 학습 크기에 맞춘다. 메모리 사용량도 같이 본다.

## 4. 검토 후 결정할 것

- [ ] **SR 단계에도 앵커 DPS 적용** (앵커 품질 개선)
  - 지금은 앵커를 LR 64로 줄여서 맞추기 때문에, 세부 모양은 SR이 새로 그린다.
  - SR refine 경로(`refine`, `refine_probs`, `extend_hr`, `backend/src/pipeline.py`)에서 `TileSampler`에 `tile_guidance`를 넘기는 방식이다.
  - 비용: SR 타일에서 역전파가 필요해 메모리와 시간이 크게 는다.
  - LR에서는 DPS가 앵커 이음새를 만들지 않음을 확인했다(앵커 평면 양옆 변화량 0.82~0.85배, 앵커 없는 볼륨은 1.02~1.09배).
- [ ] **SR 디퓨전 단계를 10단계에서 2~4단계로 줄이기** (속도)
  - SR 재학습이 필요하다. 품질이 유지되는지 실험해야 한다.

## 보류

- latent를 거의 무시하는 문제: 다양성은 초기 노이즈와 posterior 노이즈에서만 나온다. 결과물에 바로 드러나지 않으니 다양성이 부족해 보일 때 다시 본다.
- `train.initial_weights`가 best step generator와 최신 step critic을 짝지어 쓰는 문제.
- `PAPER.md`가 아직 예전 앵커 방식(학습형 앵커)을 설명한다.
