# TODO

## 1. 학습 후반 크리틱 과적합 (재학습 필요)

- [ ] `default` 샘플 LR을 다시 학습해 R1/R2 감쇠 완화(burn-in 뒤 0.5, 191a7ca)의 효과를 본다.
  - `run/20261027-1522-default-lr`(0.3)에서는 real−fake 점수 차이가 끝까지 벌어졌고(1k 0.4 → 9k 2.8), two-point 오차는 3000 step 0.007에서 8500 step 0.031로 커졌다.
  - 학습 데이터가 이미지 한 장이라 근본 원인은 데이터 쪽으로 본다. 크리틱 크기는 생성기와 비슷하다(LR 2.24M 대 1.87M).
- [ ] 효과를 나누려면 burn-in을 끈 학습(`burn_in`: `steps: 0`, `lr_scale: 1.0`, `beta2: 0.9`, `regularization_scale: 1.0`)을 하나 더 돌린다.

## 2. 검토 후 결정할 것

- [ ] **SR 단계에도 앵커 DPS 적용** (앵커 품질 개선)
  - 지금은 앵커를 LR 64로 줄여서 맞추기 때문에, 세부 모양은 SR이 새로 그린다.
  - SR refine 경로(`refine`, `refine_probs`, `extend_hr`, `backend/src/pipeline.py`)에서 `TileSampler`에 `tile_guidance`를 넘기는 방식이다.
  - 비용: SR 타일에서 역전파가 필요해 메모리와 시간이 크게 는다.
- [ ] **SR 디퓨전 단계를 10단계에서 2~4단계로 줄이기** (속도)
  - SR 재학습이 필요하다. 품질이 유지되는지 실험해야 한다.

## 3. 논문

- [ ] SR 학습이 끝나면 `paper/README.md`의 SR 절과 결론을 쓰고, README 그림을 HR 볼륨으로 바꾼다.
  - LR 부분은 `paper/scripts/evaluate_lr.py` → `make_figures.py`로 다시 만든다.

## 보류

- latent를 거의 무시하는 문제: 다양성은 초기 노이즈와 posterior 노이즈에서만 나온다. 결과물에 바로 드러나지 않으니 다양성이 부족해 보일 때 다시 본다.
- `train.initial_weights`가 best step generator와 최신 step critic을 짝지어 쓰는 문제.
