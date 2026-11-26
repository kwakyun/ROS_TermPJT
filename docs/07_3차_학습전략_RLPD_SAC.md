# 3차 학습 전략 — 안정화 RLPD-SAC + 보상 브리지

> `A_sym_v2`의 150만 스텝 로그와 롤아웃을 정밀 진단하여 규명한 **3가지 실패 원인(Q-값 폭주, 보상 절벽 2종, CPU 연산 병목)**을 해결하고, CPU 환경에서 40분 내외로 8/10 이상 성공률을 달성하기 위한 3차 학습(`v3`) 전략 문서입니다.

---

## 1. 2차 학습(`A_sym_v2`) 실패 원인 진단

### 1.1 Critic Q-값 폭주 및 엔트로피 발산 (`logs_A_sym_v2.txt`)
| 스텝 (`t`) | `ep_rew_mean` | `critic_loss` | `actor_loss` | `ent_coef` | 진단 |
|---:|---:|---:|---:|---:|---|
| 14,400 | 33.7 | `2.14` | `-16.5` | `0.72` | 정상 |
| 96,000 | 46.6 | `8.83` | `-32.2` | `0.023` | 정상 |
| 177,600 | 47.2 | `946` | `-123` | `0.131` | OOD Q-값 발산 시작 |
| 340,800 | 37.2 | `2.04e+05` | `-3,990` | `4.65` | 폭주 |
| 504,000 | 50.1 | `4.72e+07` | `-28,000` | `28.0` | 신경망 붕괴 |
| 984,000 | 61.6 | `9.49e+07` | `-43,100` | `42.5` (최대 `50.2`) | 사실상 랜덤 정책화 |

- 스크립트 시연 행동의 `±1.0` 포화 경계 및 오프라인-온라인 분포 불일치로 정책 엔트로피가 급감하고, `ent_coef: auto`가 이를 보상하려 `ent_coef`를 50.2까지 증폭시켜 Q-값이 폭발했습니다.

### 1.2 보상 함수의 구조적 절벽 2종 (`A_sym_v2/policy_best.zip` 롤아웃)
1. **원거리 닫힌 그리퍼 접근 (`close = 598/600` 스텝)**: `reach` 보상이 그리퍼 개폐 여부를 구분하지 않아 미리 닫힌 채 큐브 앞 21~23 mm에서 멈춤.
2. **목표 안착 후 해제(Release) 보상 절벽 (Seed 4 `timeout_not_released`)**: 목표 안착(`place = 2.0`) 후 그리퍼를 열기 시작하면 접촉이 떨어지며 `grasped = False`가 되어 `place`가 즉시 `0.0`으로 하락하고, 열림 판정 각도(`0.33 rad`)까지 10스텝 동안 보상이 `2.0 → 0.25`로 추락하여 그리퍼를 여는 동작을 회피함.

---

## 2. 3차 개선(`v3`) 핵심 설계

1. **RLPD(Reinforcement Learning with Prior Data) 안정화 (`pickplace/rl.py`)**
   - `LayerNormCritic`: Critic 은닉층마다 `LayerNorm`을 삽입해 OOD Q-값 과대평가 억제.
   - `StableSAC`: `ent_coef`를 `[0.005, 0.20]`으로 클램프하고 Actor/Critic 그래디언트를 `max_norm = 10.0`으로 클리핑. (`SAC.__init__` 시그니처를 변경하지 않아 기존 `evaluate.py`와 `real/policy_runner.py`의 `SAC.load()`와 100% 호환)
   - `DemoMixReplayBuffer`: 시연 40회(시드 `20000+`, 행동 `±0.85` 클립)를 별도 버퍼에 두고 학습 배치의 25%로 혼합하되 학습 70% 시점까지 `25% → 0%`로 선형 감쇠.

2. **보상 브리지(Reward Bridge) 2종 (`pickplace/env.py`, `pickplace/reward.py`)**
   - `closed_far`: 큐브에서 먼 상태(`d > 2.5 cm`, 미파지)에서 그리퍼를 닫으면 `-0.05` 감점; 근접(`d < 1.5 cm`) 닫기 시 `pre_grasp` 중간 보상 부여.
   - `opening`: 목표 안착 후(`ever_lifted & on_floor & inside_goal`) 그리퍼 열기 명령 시, 열림 진행률(`gripper_open_frac`)에 비례해 `place → release`로 연속 보간.
   - 성공 판정기(`SuccessTracker`), 관측(`observation.py`), 제어(`controller.py`)는 변경하지 않아 규정 및 sim2real 패리티를 그대로 유지.

3. **CPU 연산 6배 고속화 (`config/task.yaml`)**
   - `net_arch: [256, 256]`, `batch_size: 256`, `torch_threads: 2`, `total_timesteps: 600000` (CPU MLP forward+backward 벤치마크 `185 → 1,154 it/s`).

---

## 3. 검증 및 게이트 기준

- **Smoke 테스트 (`smoke_v3`)**: `critic_loss < 20`, `0.005 <= ent_coef <= 0.20`, `SAC.load()` 및 `check_runner_parity` 통과.
- **20만 스텝 조기 게이트 (`A_sym_v3`, 약 15~20분)**: `critic_loss < 10`, `mean_max_stage >= 2.5` 또는 `success_rate >= 0.20` 확인 후 60만 스텝 완주.
- **비교 A 완결**: `A_sym_v3` 목표 달성(공식 Seed 0~9 8/10 이상) 후 동일 설정으로 `A_base_v3` 학습 및 `reports/compare_A_v3/` 생성.
- **Plan B (20만 스텝 게이트 미달 시)**: 30회 시연 BC 워밍업 + 온폴리시 PPO 파인튜닝(60만 스텝)으로 즉시 전환.
