# 4차 학습 전략 — Input-Folded VecNormalize + 마르코프 BC 워밍업 + On-Policy PPO

**작성 일시**: 2026-11-26  
**목표**: 1~3차 SAC 학습의 4가지 구조적 실패 원인을 제거하고, CPU 환경에서 1회 약 12~15분 내로 공식 Seed 0~9 **8/10 이상** 달성 및 비교실험 A(`base` vs `sym`), 비교실험 B(`noDR` vs `DR`), 제출용 영상 완결.

---

## 1. 1~3차 학습(`v1~v3`) 실패 원인 재진단

| 차수 | 알고리즘 / 설정 | 스텝 / 시간 | 공식 Seed 0~9 | 핵심 한계 |
| :--- | :--- | :---: | :---: | :--- |
| `v1` | SAC (raw obs) | 1.5M / 4.6h | 0 / 10 | 호버링 보상 편법 + 135배 관측 스케일 불균형 |
| `v2` | SAC + raw 데모 | 1.5M / 4.6h | 0 / 10 | 오프라인-온라인 OOD로 `ent_coef=50.2`, `critic_loss=9.49e+07` 폭발 |
| `v3` | StableSAC + 데모 혼합 | 600k / 53m | 0 / 10 | Q-폭발은 해결했으나 관측 불균형·보상 역전·그립 지속 부족 지속 |

### 정밀 진단으로 밝혀진 4가지 구조적 원인

1. **관측 벡터의 135배 스케일 불균형 (`observation.py`)**
   - `cube_xyz - grasp_point`의 표준편차는 `0.00408`인 반면 `joint_q` 및 `sin/cos`, 플래그는 `0.5504`로 **135배 차이**(조건수 약 18,225)가 존재함.
   - 입력 정규화 없이 MLP 첫 층에 넣으면 4 mm 정밀도의 파지 오차 신호가 묻힘.
   - **사전 프로브 검증**: 동일 시연 BC 학습 시 정규화 적용 전후 MSE가 **`0.1661` (0/10) → `0.0103` (16배 감소)**으로 급감함.
2. **보상 함수의 수학적 역전 (`timeout_not_released` 원인)**
   - 기존 설정($\gamma=0.98$, `place=1.80`, `release=2.20`, `hold=0.80`, `success_bonus=20.0`)에서는:
     - 안 놓고 30초까지 버틸 때 기대 가치: $V = \frac{1.80}{1 - 0.98} = 90.0$
     - 즉시 열고 2초 유지 후 성공 종료(`done=True`) 시 가치: $V \approx 85.1 < 90.0$
   - 즉, 성공 종료보다 버티기가 더 유리한 보상 역전이 존재했으며, 목표 경계 직전 바닥 접촉 시 `carry(1.40) → grasp(0.60)` 급락으로 인한 재리프트 낙하(`dropped`)가 발생함.
3. **기존 스크립트 시연(`scripted_check.py`)의 비마르코프 모순**
   - `t > 30`, `t > 35` 타이머 기반 상태기계는 동일 관측에서 모순된 `dz` 행동을 출력해 지도학습 및 리플레이 학습을 방해함.
   - 상태 기반 순수 마르코프 전문가로 교체 + 입력 정규화 BC만 적용한 **사전 프로브(28개 시연, 60초, RL 0스텝)**에서 즉시 **공식 Seed 0~9 `2/10` (`+7.8, +8.4 mm`), `mean_max_stage=3.20`, 파지 `9/10`**을 달성함.
4. **계획서 원안(PPO) 복귀의 이점**
   - 계획서 §3에 명시된 On-policy **PPO**는 Critic Q-폭발이 원천 차단되며, CPU 8병렬 환경에서 **~700~1,000 FPS**로 구동되어 40만 스텝이 약 10~15분 만에 완료됨.

---

## 2. 4차 개선(`v4`) 핵심 설계

### 2.1 `VecNormalize` 학습 + 저장 시 1층 대수적 병합 (Sim2Real 100% 호환)
- 학습 시 `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=100.0)`로 135배 스케일 불균형을 제거.
- `policy_best.zip` / `policy_final.zip` 저장 시 표준 `PPO("MlpPolicy")`의 `policy_net[0]` 및 `value_net[0]` 가중치·편향에 정규화 통계 $(\mu, \sigma)$를 수학적으로 병합:
  $$W' = W \cdot \mathrm{diag}(1/\sigma), \qquad b' = b - W(\mu/\sigma)$$
- 저장 직후 원본 가중치를 복원하여 학습은 정규화 공간을 유지하고, 저장된 `.zip`은 `evaluate.py`, `record_video.py`, `tools/check_runner_parity.py`, `real/policy_runner.py`에서 **수정 없이** 표준 `PPO.load()`로 원 관측 입력을 받아 오차 `< 1e-5`로 추론함.

### 2.2 보상 수식 정상화 (`config/task.yaml`, `pickplace/reward.py`)
- **밀집(Dense) 보상 10배 축소 + `gamma=0.995` + `success_bonus=100.0`**:
  - `reach_scale=0.02, grasp=0.04, lift=0.06, carry=0.08, place=0.10, release=0.14, hold=0.06`, **`success_bonus=100.0`**.
  - 초기 PPO 검증(`t=0..80k`)에서 밀집 보상이 클수록 긴 실패 에피소드(240점)가 빠른 성공을 압도하는 현상을 확인하여, 스텝당 보상을 1/10로 줄임으로써 성공 시 총수익(`+108` 비할인 / `+45~84` 할인)이 버티기 최대치(`+35~50` 비할인 / `+15~18` 할인)를 압도하도록 교정.
- **`place_near` 브리지 추가 (`reward.py`)**:
  - 들어올린 뒤 목표 근처 바닥에 닿았을 때(`grasped & on_floor & ever_lifted & not inside_goal`) 보상이 `grasp`로 추락하지 않고 `carry` 거리 비례 보상을 유지해 급격한 재리프트 낙하를 방지.

### 2.3 마르코프 80회 BC 워밍업 + 저분산(`std=0.061`) Critic 워밍업 + 완만 PPO
1. **시연 수집**: 분리 시드(`20000+`)에서 마르코프 전문가 성공 궤적 **80회** 수집(행동/할인수익 MC Return 계산).
2. **BC 워밍업**: `obs_rms`·`ret_rms` 초기화 후 Actor(행동 MSE) + Critic(정규화 수익 MSE) 동시 120에폭 학습, 초기 탐색 표준편차 **`log_std = -2.8`($\sigma \approx 0.061$)** 설정 (실측 진단에서 $\sigma=0.20$일 때 확률적 성공률이 7.7%로 붕괴하던 반면 $\sigma=0.05$에서는 4/10이 온전히 보존됨을 확인).
3. **Critic 워밍업 (2 Rollout)**: 초반 2번의 Rollout(16,384 스텝) 동안 Actor(`policy_net`, `action_net`, `log_std`)를 동결해 Value Network를 On-policy 분포에 정합시킨 뒤 Actor 해동.
4. **`AnchoredPPO` 안정화 본학습 (`240,000` 스텝, `rl.py`)**:
   - **BC 앵커 손실(`bc_anchor_loss`)**: Actor 해동 후 매 PPO 미니배치에 시연 행동 MSE(`0.50 → 0.10` 선형 감쇠)를 더해 4 mm 파지 정밀도 망각(Catastrophic Forgetting) 원천 차단.
   - **음수 어드밴티지 감쇠(`adv_neg_scale = 0.25`)**: 후반 낙하 실패가 앞선 접근·파지·운반 가중치까지 덮어쓰는 신용 할당 붕괴 방지.
   - **정규화 통계 고정(`obs_rms.count = 1e9`), 미세 학습률(`1.5e-5 → 3e-6`), `std ∈ [0.03, 0.10]` 클램핑** (`n_envs=8, n_steps=1024, batch_size=256, n_epochs=4, clip_range=0.15, target_kl=0.015`).

---

## 3. 검증 및 실험 로드맵

1. **Step 0 (게이트)**: 단위 테스트(`test_success_rules.py` 6/6) + Fold 정밀도(`2.33e-09`) + `check_runner_parity` (`0.00e+00`, `PARITY OK`).
2. **Step 1 (`A_sym_v4`)**: 240k 스텝 학습 및 공식 10회 + 추가 100회 평가 (목표: 공식 **8/10 이상**).
3. **Step 2 (`A_base_v4` & 비교 A)**: 완전히 동일한 조건(`exp_A_base.yaml`) 학습·평가 후 `reports/compare_A_v4/` 생성.
4. **Step 3 (`B_dr_v4` & 비교 B)**: Domain Randomization(`exp_B_dr.yaml`) 학습 + `eval_noise.yaml` 노이즈 평가 후 `reports/compare_B_v4/` 생성.
5. **Step 4 (영상·백업)**: `A_sym_v4` Seed 0~9 영상 녹화(`videos/final_sim_10trials.mp4`), 패리티 검증 및 GitHub 푸시.

---

## 4. 최종 실험 결과 (`v4`, 동일 조건 공정 비교)

### 4.1 비교 A: 90° 대칭 관측·보상 효과 (공칭 환경, `reports/compare_A_v4/`)

| 방법 | 학습 run | 공식 10회 | 추가 무작위 (100회) | 성공 여유 평균/최소 [mm] | 평균 성공 시간 [s] | 종료 원인 |
|---|---|---|---|---|---|---|
| base | A_base_v4 | 4/10 | 22.0% | 6.2 / 0.0 | 14.7 | {'dropped': 31, 'success': 26, 'outside_goal': 19, 'out_of_area': 7, 'timeout_not_released': 4, 'timeout_carrying': 18, 'lift_fail': 3, 'grasp_fail': 2} |
| sym | A_sym_v4 | 6/10 | 57.0% | 7.1 / 0.6 | 14.9 | {'success': 63, 'dropped': 24, 'timeout_not_released': 6, 'outside_goal': 13, 'out_of_area': 1, 'timeout_carrying': 3} |

- **핵심 결론**: 90° 대칭 yaw 표현(`sin 4θ, cos 4θ`) 및 잡기/놓기 정렬 보상을 적용한 `A_sym_v4`는 동일한 80회 시연 + `AnchoredPPO` 조건에서 `A_base_v4` 대비 추가 무작위 100회 성공률을 **22.0% → 57.0% (2.59배, +35.0%p)** 로 끌어올렸으며, 파지/리프트 실패를 **0건**으로 제거하고 배치 여유도(평균 `7.1 mm`)를 개선함.

### 4.2 비교 B: Domain Randomization 효과 (노이즈 시험 환경 `eval_noise.yaml`, `reports/compare_B_v4/`)

| 방법 | 학습 run | 공식 10회 | 추가 무작위 (100회) | 성공 여유 평균/최소 [mm] | 평균 성공 시간 [s] | 종료 원인 |
|---|---|---|---|---|---|---|
| noDR | A_sym_v4 | 2/10 | 26.0% | 6.5 / 0.6 | 13.6 | {'dropped': 34, 'success': 28, 'outside_goal': 5, 'timeout_not_released': 21, 'timeout_carrying': 14, 'lift_fail': 3, 'out_of_area': 4, 'grasp_fail': 1} |
| DR | B_dr_v4 | 2/10 | 6.0% | 6.4 / 3.8 | 9.8 | {'dropped': 60, 'grasp_fail': 24, 'success': 8, 'out_of_area': 5, 'outside_goal': 10, 'timeout_carrying': 1, 'timeout_not_released': 2} |

- **공칭 환경에서의 `B_dr_v4`**: 공식 0/10, 추가 11/100 (11.0%).
- **분석**: 가림(`occlusion_radius=0.02`), 카메라 지연(`[0,2]` 스텝), 관측 편향/검출 실패(`dropout=0.05`)가 동시에 켜진 상태에서 참값 기반 전문가 행동을 모방(BC)할 때 지연된/가려진 관측과의 불일치가 발생하여, `B_dr_v4`는 BC 초기 성능(`6.7%`) 자체가 낮았음(다만 성공 시 평균 시간 `9.8 s`, 최소 여유 `+3.8 mm`로 빠르고 `timeout_not_released` 빈도가 낮음). 반면 공칭 환경에서 정밀하게 학습된 `A_sym_v4`는 가혹한 미지 노이즈 환경에서도 `26.0%`의 일반화 성능을 유지함. (실습실 1차 실측 후 DAgger/미세조정 또는 실측 노이즈 범위에 맞춘 추가 DR 튜닝 여지 확인됨.)
