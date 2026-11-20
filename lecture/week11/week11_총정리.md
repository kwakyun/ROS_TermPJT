# [11주차 총정리] SAC로 로봇 정책 학습하기 — 두 큐브 접촉 과제

> 범위: `lecture.md`, `practice.md`, `week11_SAC_reward_shaping.pptx`, `source_code/` 전체 (`reward.py`, `train.py`, `evaluate.py`, `env.py`, `observation.py`, `inspect_environment.py`, `visualize.py`, `visualization.py`, `config.yaml`, `models/`)
> 한 줄 요약: **10주차에 만든 "Gymnasium 환경 + 무작위 Agent" 구조에서 Agent를 SAC 정책으로 바꾼다. 보상을 직접 작성하고(TODO 1), Stable-Baselines3로 모델 생성·학습·저장(TODO 2~4), 불러오기·결정적 평가(TODO 5~6)까지 한 바퀴를 돈다.**

---

## 목차

1. [전체 흐름 한눈에 보기](#1-전체-흐름-한눈에-보기)
2. [핵심 개념](#2-핵심-개념)
3. [Stable-Baselines3 API](#3-stable-baselines3-api)
4. [이번 환경의 설계 명세](#4-이번-환경의-설계-명세)
5. [보상 설계](#5-보상-설계)
6. [코드 구조와 호출 흐름](#6-코드-구조와-호출-흐름)
7. [파일별 코드 설명](#7-파일별-코드-설명)
8. [TODO 1~6 정답과 해설](#8-todo-16-정답과-해설)
9. [실습 진행 순서](#9-실습-진행-순서)
10. [검증 기준값](#10-검증-기준값)
11. [자주 틀리는 부분과 문제 해결](#11-자주-틀리는-부분과-문제-해결)
12. [복습 질문](#12-복습-질문)
13. [10 → 11 → 12주차 연결](#13-10--11--12주차-연결)

---

## 1. 전체 흐름 한눈에 보기

### 1.1 수업의 위치

```text
9주차 : 사람이 관절 목표를 정해 MuJoCo/로봇에 전달
10주차: MuJoCo 장면을 Gymnasium 환경으로 감싸기 → Random Agent로 실행 확인
11주차: 같은 환경 구조에 보상을 넣고 SAC 정책 학습·저장·평가 (고정 배치)   ← 지금
12주차: 같은 접촉 과제의 초기 배치 다양화 + 보상 확장
```

### 1.2 이번 주에 답할 수 있어야 하는 질문

| 질문 | 짧은 답 |
|---|---|
| Policy는 무엇을 받아 무엇을 내는가? | Observation 15개 → Action `[ax, ay, az]` 3개 |
| Actor·Critic·Replay Buffer의 역할은? | Actor = 행동 선택, Critic = 그 행동의 가치(Q값) 추정, Buffer = 경험 저장·재사용 |
| 보상식을 바꾸면 무엇이 달라지나? | Agent가 경험을 "좋다/나쁘다"로 평가하는 기준이 바뀌고, 그 결과 학습되는 행동이 바뀐다 |
| `learn/save/load/predict`는 언제 쓰나? | 학습 실행 / 정책 보관 / 정책 복원 / 평가 행동 계산 |
| 학습 Return과 평가 성공률의 차이는? | Return = 한 Episode의 보상 합(학습 신호), 성공률 = 실제 과제 달성 비율. 둘은 따로 확인한다 |

### 1.3 실습 전체 흐름

```text
이전 분반 ~/week11 삭제 → 새 venv → 파일 배치 → torch(CPU)·requirements 설치
→ 초기 환경 확인(reset 예제, visualize)
→ TODO 1~6 작성
→ Source 검사 → inspect_environment(API·한 Step) → Reward 검사
→ 확인 학습 5,200 Step(smoke) → 저장·평가 연결 확인
→ 본 학습 60,000 Step(baseline) → 40 Episode 평가 → Viewer로 정책 관찰
→ 결과 기록·제출 → deactivate → ~/week11 전체 삭제
```

### 1.4 한 장으로 보는 데이터 흐름

```text
                  ┌──────────── 학습 (train.py) ────────────┐
config.yaml ──▶ SAC("MlpPolicy", env, ...)                        TODO 2
                  │
                  ▼
            model.learn(total_timesteps)                          TODO 3
              ├─ 처음 5,000 Step: 무작위 Action으로 경험 수집
              ├─ 이후: Actor가 확률적 Action 선택 → env.step()
              │        └─ env가 reward.py로 보상 계산             TODO 1
              ├─ 경험 (obs, action, reward, next_obs, done) → Replay Buffer
              └─ Step마다 Buffer에서 256개 추출 → Critic·Actor 갱신
                  │
                  ▼
            model.save("policy_final")  →  policy_final.zip       TODO 4

                  ┌──────────── 평가 (evaluate.py) ─────────┐
            SAC.load(policy_final.zip, env, device="cpu")         TODO 5
                  │
                  ▼
            while not 종료:
                action, _ = model.predict(obs, deterministic=True) TODO 6
                obs, ... = env.step(action)
                  │
                  ▼
            evaluation_summary.json / evaluation_episodes.csv
```

---

## 2. 핵심 개념

### 2.1 환경 실행에서 정책 학습으로

10주차의 Agent는 `action_space.sample()`로 매번 무작위 행동을 뽑았고, 결과를 기억하지 않았다.
이번에는 **행동의 결과를 경험으로 저장하고, 그 경험으로 행동 선택 방법을 바꾼다.**

- 지도학습과의 차이: 사람이 "정답 Action"을 주지 않는다. 환경이 행동 결과의 **보상**만 계산해 준다.
- 목표: 앞으로 받을 **할인 누적 보상**이 커지도록 정책을 개선한다.

### 2.2 Policy (정책)

- 현재 Observation을 받아 Action을 선택하는 함수. 이번에는 신경망(MLP)으로 표현한다.
- 입력 15개 실수 → 출력 3개 실수. 가중치가 바뀌면 같은 관측에서도 행동이 달라진다.

### 2.3 SAC (Soft Actor-Critic)

- **연속 Action**을 다루는 강화학습 알고리즘. XYZ 이동량처럼 크기·방향을 실수로 표현하는 문제에 맞다.
- **확률적 정책**: 학습 중에는 행동 분포에서 표본을 뽑아 여러 행동을 시도한다.
- **엔트로피**: 정책이 행동을 얼마나 다양하게 고르는지 나타내는 양. SAC는 보상과 엔트로피를 함께 고려해 탐색을 유지한다.
- **엔트로피 계수(`ent_coef`)**: 보상과 행동 다양성 사이의 비중. `auto_0.1`은 0.1에서 시작해 자동 조정한다.

SAC를 이루는 네 가지를 묶어 기억한다.

| 구성 요소 | 입력 → 출력 | 역할 |
|---|---|---|
| **Actor** | Observation → 행동 분포 → Action | 행동을 선택한다 |
| **Critic (2개)** | Observation + Action → Q값 | 그 행동 이후의 할인 누적 보상을 추정한다 |
| **Replay Buffer** | 경험 저장 → Mini-batch | 과거 경험을 다시 꺼내 학습에 쓴다 |
| **엔트로피 항** | — | 탐색(행동 다양성)을 유지한다 |

### 2.4 Actor와 Critic

- **Actor**는 Critic이 추정한 가치와 엔트로피를 고려해 갱신된다. → "가치가 높다고 평가되는 행동"을 더 자주 고르게 된다.
- **Critic**은 Q값 추정과 학습 목표 사이의 오차로 갱신된다. SAC는 Q함수를 **두 개** 쓰고, 목표값 계산에 **작은 쪽**을 사용해 과대 추정을 줄인다.
- **목표 네트워크와 `tau`**: 목표 Critic은 `0.995 × 기존 목표 + 0.005 × 현재 Critic`으로 천천히 따라간다.
- `critic_loss`가 작다고 과제를 잘하는 것은 아니다. 손실과 성공률은 **서로 다른 측정값**이다.

### 2.5 Replay Buffer와 Off-policy

한 번의 상호작용에서 얻는 경험:

```text
(현재 Observation, Action, Reward, 다음 Observation, 종료 정보)
```

- Buffer는 최대 60,000개를 보관하고, 가득 차면 오래된 것부터 덮어쓴다.
- 갱신할 때마다 256개를 **무작위로** 뽑는다(Mini-batch).
- 성공·실패 Step뿐 아니라 "팔만 움직이고 큐브는 그대로인" Step도 모두 경험이다.
- **Off-policy**: 과거 정책(초기 무작위 행동 포함)이 모은 경험도 다시 사용할 수 있는 학습 방식.

### 2.6 `learning_starts`와 갱신 횟수

- 처음 **5,000 Step**은 무작위 행동으로 경험만 모으고, 그 뒤부터 신경망을 갱신한다.
- `train_freq=1`, `gradient_steps=1`이므로 이후에는 **환경 Step 1회당 갱신 1회**다.

```text
갱신 횟수 = 전체 Step − learning_starts
 5,200 Step →    200회   (확인 학습이 5,200인 이유: 갱신이 실제로 일어나는지 보려고)
60,000 Step → 55,000회
```

### 2.7 학습과 평가의 구분

| | 학습 (`learn`) | 평가 (`predict`) |
|---|---|---|
| 행동 선택 | 확률적(탐색) | 결정적 `deterministic=True` |
| 가중치 | 갱신한다 | 고정한다 |
| 반복문 주체 | SB3 내부 | 학생 코드 |
| 주로 보는 값 | `ep_rew_mean`, 최근 성공률, loss | 성공률, 평균 Step, 최종 거리 |

### 2.8 Reward · Return · 성공률

- **Reward**: 한 Step의 값.
- **Return**: 한 Episode의 보상 합. 로그의 Episode Return은 **할인하지 않은** 합계이고, 학습 목표에는 할인율 `gamma`가 쓰인다.
- **성공률**: 성공 Episode 수 ÷ 전체 평가 Episode 수.
- Return이 올라도 실제 접촉 성공이 없을 수 있다(5.4절). 그래서 둘을 함께 읽는다.

### 2.9 주요 하이퍼파라미터 (`config.yaml`의 `training`)

| 설정 | 값 | 의미 |
|---|---|---|
| `total_timesteps` | 60000 | 전체 환경 상호작용 **예산** (Episode 수가 아니다) |
| `learning_rate` | 0.0003 | 한 번의 갱신으로 가중치를 바꾸는 크기 |
| `buffer_size` | 60000 | Buffer의 **저장 용량** |
| `learning_starts` | 5000 | 갱신 전 무작위 수집 Step |
| `batch_size` | 256 | 한 번의 갱신에 뽑는 경험 수 |
| `gamma` | 0.99 | 미래 보상 할인율. k Step 뒤 보상에 `gamma**k`가 곱해진다 |
| `tau` | 0.005 | 목표 Critic의 완만한 갱신 비율 |
| `ent_coef` | `auto_0.1` | 엔트로피 계수 자동 조정(초기 0.1) |
| `train_freq`, `gradient_steps` | 1, 1 | Step 1회마다 갱신 1회 |
| `network_architecture` | `[128, 128]` | Actor·Critic 각각 128 뉴런 은닉층 2개 |
| `seed` | 12 | 모델 초기화·무작위 행동의 난수 기준 |

> `buffer_size=60000`과 `total_timesteps=60000`은 숫자만 같을 뿐 **용량**과 **예산**으로 역할이 다르다.

---

## 3. Stable-Baselines3 API

### 3.1 역할 분담

| 구성 요소 | 담당 |
|---|---|
| MuJoCo | 로봇·큐브의 운동과 접촉 계산 |
| Gymnasium 환경 (`env.py`) | Space, `reset`·`step`, 보상·종료 조건 제공 |
| Stable-Baselines3 (SB3) | 경험 수집, Replay Buffer 관리, Actor·Critic 갱신 |
| 학생 코드 | 모델 생성 → 학습 요청 → 저장 → 불러오기 → 평가 연결 |

### 3.2 기본 API 다섯 가지

| API | 역할 | 반환값 |
|---|---|---|
| `SAC("MlpPolicy", env, ...)` | 정책 종류·환경·설정으로 모델 생성 | SAC 모델 객체 |
| `model.learn(total_timesteps=N)` | 최대 N Step 상호작용하며 학습 | 학습된 모델 자신 |
| `model.save("policy")` | 가중치·모델 설정 저장 (`.zip` 자동 부착) | `None` |
| `SAC.load("policy.zip", env=env)` | 저장한 모델 복원 + 환경 연결 | 복원한 모델 객체 |
| `model.predict(obs, deterministic=True)` | 현재 관측으로 평가 행동 계산 | `(action, state)` |

- `MlpPolicy`: 실수 벡터를 입력받는 다층 퍼셉트론 정책. 입력 15·출력 3의 크기는 환경의 **Space에서 자동으로** 읽는다.
- `load()`는 인스턴스가 아니라 **클래스**에서 호출하고, 반환값을 변수에 받아야 한다.
- `predict()`의 두 번째 반환값 `state`는 순환 정책용이며 MLP에서는 `None`이다.
- `save()`에는 Replay Buffer가 들어가지 않는다. 필요하면 `save_replay_buffer()`를 따로 쓴다.

### 3.3 Monitor와 Callback (제공 코드)

- **`Monitor`**: 환경을 감싸 Episode 결과를 CSV로 기록하는 Wrapper. 기본 열 `r`(Return)·`l`(길이)·`t`(경과 시간)에 더해 `info_keywords`로 지정한 `is_success`, `scenario`, `failure_reason`을 기록한다.
- **Callback**: 학습 중 SB3가 Step마다 호출하는 코드. `_on_step()`이 `True`면 계속, `False`면 **조기 종료**.
- **`CallbackList([...])`**: 여러 콜백을 묶어 `learn(callback=...)`에 한 번에 연결한다.

| 콜백 | 하는 일 |
|---|---|
| `ProgressCallback` | 5,000 Step마다 `Progress n/total`, 경과 시간, ETA 출력 |
| `ValidationCallback` | 최근 학습 성공률이 99% 이상이면 별도 20 Episode 검증 → `policy_best` 저장, 검증도 99% 이상이면 `policy_converged` 저장 후 조기 종료 |
| `CheckpointCallback` | 10,000 Step마다 `checkpoints/sac_push_*.zip` 저장 |

### 3.4 CLI 옵션과 API 인자의 연결

| CLI 옵션 | 연결되는 곳 | 기본값 |
|---|---|---|
| train `--steps` | `learn(total_timesteps=...)` | 60000 |
| train `--seed` | `SAC(seed=...)` | 12 |
| train `--device` | `SAC(device=...)` | `auto` (실습은 `cpu`) |
| train `--threads` | `torch.set_num_threads()` | 1 |
| train `--run-name` | 저장 폴더 `outputs/<이름>/` | 현재 시각 |
| evaluate `--run` | 정책이 있는 실행 폴더 | `latest` |
| evaluate `--model` | `policy_final` / `policy_best` 선택 | `final` |
| evaluate `--policy` | ZIP 직접 경로 (있으면 우선) | 없음 |
| evaluate `--episodes-per-scenario` | 평가 회차 수 | 40 |
| evaluate `--seed` | 평가 `reset(seed=...)` 시작값, 회차마다 +1 | 12000 |

### 3.5 가장 작은 전체 예제 (Pendulum)

```python
import gymnasium as gym
from stable_baselines3 import SAC

environment = gym.make("Pendulum-v1")
try:
    model = SAC("MlpPolicy", environment, learning_rate=0.0003,
                learning_starts=100, seed=12, device="cpu")   # 생성
    model.learn(total_timesteps=200)                          # 학습
    model.save("pendulum_policy")                             # 저장

    loaded_model = SAC.load("pendulum_policy.zip", env=environment, device="cpu")  # 복원
    observation, info = environment.reset(seed=12)
    terminated = truncated = False
    while not (terminated or truncated):
        action, state = loaded_model.predict(observation, deterministic=True)      # 예측
        observation, reward, terminated, truncated, info = environment.step(action)
finally:
    environment.close()
```

로봇 실습의 TODO 2~6은 이 다섯 줄을 제공 환경과 `config.yaml` 값으로 바꿔 쓰는 것과 같다.

---

## 4. 이번 환경의 설계 명세

좌표는 모두 로봇 `base_link` 기준, 길이 단위는 m, 각도 단위는 rad이다.

### 4.1 장면 (`models/push_scene.xml`)

| 항목 | 값 |
|---|---|
| 초록 큐브(`object_cube`) 시작 XY | `(0.310, 0.030)` |
| 파란 큐브(`target_cube`) 시작 XY | `(0.370, 0.000)` |
| 큐브 | 한 변 25 mm, 11 g, `freejoint`(자유 물체), 초기 yaw 0 |
| 초기 EEF XYZ | 약 `(0.225, 0, 0.015)` |
| 초기 손끝 XYZ | 약 `(0.253904, -0.001521, 0.034919)` |
| 작업면 Z | 약 `-0.0048496` |

- 배치가 **고정**이다. `config.yaml`의 spawn 범위가 `[0.31, 0.31]`처럼 하한=상한이라 매번 같은 위치가 나온다. 시나리오도 `diagonal_right` 하나(확률 1.0)뿐이다.
- 두 큐브 모두 자유 물체라 접촉하면 밀리거나 회전할 수 있다.

### 4.2 Observation — `Box(-inf, inf, (15,), float32)`

| 인덱스 | 내용 | 단위 |
|---|---|---|
| 0–5 | `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` 실측 각도 | rad |
| 6–8 | `moving_fingertip` (손끝) XYZ | m |
| 9–11 | 초록 큐브 중심 XYZ | m |
| 12–14 | 파란 큐브 중심 XYZ | m |

- 정규화하지 않은 **실측값 그대로**다. `observation[9:12]`를 읽으면 초록 큐브 위치를 바로 알 수 있다.
- Space의 `±inf`는 수용 범위 선언일 뿐, 반환값은 유한하다. `check_env`가 무한대 경계 경고를 낼 수 있다(정상).
- EEF 제어점 좌표는 관측에 **없다**. `info['eef_position']`에서 확인한다.

### 4.3 Action — `Box(-1, 1, (3,), float32)`

| 성분 | 한 Step 최대 요청 이동량 | EEF 목표 제한 범위 |
|---|---|---|
| `ax` | X ±0.008 m | `[0.15, 0.40]` |
| `ay` | Y ±0.008 m | `[-0.14, 0.14]` |
| `az` | Z ±0.005 m | `[0.008, 0.035]` |

- 예: `[0.5, -0.25, 1.0]` → `[+0.004, -0.002, +0.005]` m 요청.
- **요청 ≠ 실측**. 실제 EEF는 IK 오차, 관절 변화량 제한, 구동 과도 응답, 접촉의 영향을 받는다.

### 4.4 제어점과 관측점은 다른 점이다

| | 이름 | 쓰임 | 초기 XYZ | 읽는 곳 | Top 화면 표시 |
|---|---|---|---|---|---|
| 제어점 | `gripperframe` | IK가 이동 목표로 쓰는 점 (Action이 움직이는 대상) | `(0.225, 0, 0.015)` | `info['eef_position']` | 노란색 `EEF/IK` |
| 관측점 | `moving_fingertip` | 움직이는 손가락 끝. 보상의 `d_tip` 기준 | `(0.254, -0.0015, 0.0349)` | `observation[6:9]` | 자홍색 `MOVING TIP` |

→ Action은 EEF를 움직이라고 요청하지만, 정책이 보고 보상이 계산되는 점은 손끝이다. 좌표를 비교할 때 어느 점인지 먼저 확인한다.

### 4.5 제어와 시간

- IK는 **3개 관절**(`shoulder_pan`, `shoulder_lift`, `elbow_flex`)만 사용한다. 손목 2개와 그리퍼는 각각 `0`, `0`, `0.20` rad 목표로 고정.
- 물리 계산 2 ms × `frame_skip` 25 = **제어 주기 50 ms**.
- 두 큐브 접촉을 감지하면 그 Action의 남은 물리 계산을 건너뛰고 일찍 끝낸다.

### 4.6 종료 조건

| 구분 | 조건 | 신호 |
|---|---|---|
| 성공 | MuJoCo가 두 큐브 geom 사이의 **물리 접촉**을 감지 | `terminated=True`, `is_success=True` |
| 실패 | 어느 큐브든 XY가 X `[0.14, 0.41]`·Y `[-0.13, 0.13]`을 벗어나거나, 중심 높이가 작업면 기준 `[-0.002, 0.065]`를 벗어남 | `terminated=True`, `is_failure=True` |
| 시간 제한 | 성공·실패 없이 180 Step | `truncated=True` |

- 성공 판정은 **거리가 아니라 접촉**이다. 성공해도 중심 사이 거리는 큐브 크기만큼(약 0.025~0.035 m) 남는다.
- 성공과 실패가 같은 Step에 일어나면 둘 다 참이고 보상도 둘 다 더한다.
- 과제 종료가 난 Step에서는 `truncated`를 거짓으로 돌려준다.
- 로봇이 파란 큐브를 직접 건드리는 것은 실패가 아니다(`info['target_robot_contact']`에 기록만 한다).
- `failure_reason`: `object_outside_workspace`, `target_invalid_height` 등. 시간 제한으로 끝나면 빈 문자열이다.

### 4.7 `info` 주요 항목

| 키 | 내용 |
|---|---|
| `is_success`, `is_failure`, `failure_reason` | 종료 원인 |
| `elapsed_steps` | 현재 Episode의 Step 수 |
| `object_position`, `target_position` | 초록·파란 큐브 XYZ |
| `eef_position`, `moving_tip_position` | 제어점·손끝 XYZ |
| `distance_between_cubes` | 두 큐브 중심 거리 |
| `cubes_touching`, `contact_latched`, `contact_substep` | 접촉 여부와 감지된 물리 Step |
| `target_robot_contact` | 로봇–파란 큐브 접촉 여부 |
| `ik_error`, `ik_rejected` | IK 잔여 오차, 이번 Step의 IK 거부 여부 |
| `reward_terms`, `computed_reward` | 보상 항목별 값과 총합 |
| `environment_version` | `direct_xyz15_distance_v1` |

---

## 5. 보상 설계

### 5.1 보상식

```text
d_tip   = ‖손끝 XYZ − 초록 큐브 XYZ‖
d_cubes = ‖초록 큐브 XYZ − 파란 큐브 XYZ‖

reward = −10 × (d_tip + d_cubes)      ← distance  (매 Step)
         + 100  (성공한 Step)          ← success
         − 100  (실패한 Step)          ← failure
```

| 상황 (`d_tip=0.05`, `d_cubes=0.08`) | distance | success | failure | 합계 |
|---|---|---|---|---|
| 진행 중 | −1.3 | 0 | 0 | **−1.3** |
| 성공 | −1.3 | +100 | 0 | **98.7** |
| 실패 | −1.3 | 0 | −100 | **−101.3** |

- 보상은 항상 음수(비용)에서 시작한다. **0에 가까울수록 좋다.** −1.3보다 −0.9가 더 높은 보상이다.
- 거리는 m 단위로 넣는다. 5 cm는 `0.05`다. `distance_weight=10`이 비용의 크기를 정한다.

### 5.2 Sparse vs Dense

| 종류 | 언제 주나 | 이번 보상에서 |
|---|---|---|
| **Sparse** | 성공·실패 같은 사건이 일어날 때만 | `success` +100, `failure` −100 |
| **Dense** | 매 Step | `distance` 비용 |

Sparse만 쓰면 무작위 탐색으로 우연히 접촉하기 전까지 학습 신호가 없다.
Dense 거리 비용을 더하면 **큐브에 닿기 전에도** "손끝이 초록 큐브에 가까워졌는가"가 피드백이 된다. 이렇게 중간 과정에 보상을 주어 학습 방향을 잡아 주는 것이 **Reward Shaping**이다.

### 5.3 두 거리를 함께 쓰는 이유

| 항 | 유도하는 행동 |
|---|---|
| `d_tip` (손끝–초록) | 초록 큐브에 **접근** |
| `d_cubes` (초록–파랑) | 초록 큐브를 파란 큐브 쪽으로 **밀기** |

- `d_tip`만 있으면 큐브 옆에 붙어 있기만 해도 보상이 좋아진다.
- `d_cubes`만 있으면 큐브를 건드리기 전에는 값이 변하지 않아 접근 신호가 없다.

### 5.4 보상과 실제 성공은 다르다

- 손끝이 초록 큐브 가까이 머물러도 **유리한 방향으로 밀지 않을 수 있다.** 거리 비용만 줄고 접촉은 없는 정책이 나올 수 있다.
- 두 거리만으로는 "뒤로 돌아가서 민다" 같은 방향 정보를 모두 표현하지 못한다.
- 그래서 Return뿐 아니라 **성공률·최종 거리·Viewer 영상**을 함께 본다.
- 보상 스케일(가중치 10, 종료 ±100)은 Critic의 가치 추정과 엔트로피 균형에도 영향을 준다.

---

## 6. 코드 구조와 호출 흐름

### 6.1 파일 구성

```text
~/week11/sac_lab/                  ← python3 -m 실행 위치 (프로젝트 루트)
├── config.yaml                    과제·보상·학습·평가 설정
├── requirements.txt
├── models/
│   ├── push_scene.xml             두 큐브 + 조명 + include
│   ├── physicai_arm.xml, robot.xml
│   └── assets/                    STL·PNG
├── week11_sac/                    Python 8개
│   ├── reward.py                  [TODO 1] 보상
│   ├── train.py                   [TODO 2~4] 생성·학습·저장
│   ├── evaluate.py                [TODO 5~6] 불러오기·예측
│   ├── env.py                     환경 (제공)
│   ├── observation.py             15차원 관측 (제공)
│   ├── inspect_environment.py     API·한 Step 확인 (제공)
│   ├── visualize.py               Viewer 실행 (제공)
│   └── visualization.py           Top/Front 화면 그리기 (제공)
└── outputs/<run-name>/            학습 실행마다 생성
    ├── run_config.yaml            실행에 적용한 설정
    ├── training.monitor.csv       학습 Episode의 r·l·t·성공
    ├── tensorboard/, checkpoints/, evaluations/history.csv
    ├── policy_final.zip           (+ policy_best.zip, policy_converged.zip)
    ├── metrics.json               Step·시간·갱신 수
    └── evaluation_final/          evaluation_summary.json, evaluation_episodes.csv
```

### 6.2 모듈 의존 관계

```text
train.py ──▶ evaluate.py(evaluate_model) ──▶ env.py ──▶ observation.py
    │                                           └─────▶ reward.py
    └──▶ env.py
visualize.py ──▶ env.py, evaluate.py(resolve_policy_path), visualization.py
inspect_environment.py ──▶ env.py
```

`reward.py`가 미완성이면 `env.step()`이 `NotImplementedError`를 내므로 학습·평가·점검이 모두 멈춘다. 그래서 TODO 1이 가장 먼저다.

### 6.3 `env.step(action)` 한 번의 내부 흐름

```text
action (3,)
 │ ① 검사: shape (3,), 유한값 → clip(-1, 1)
 ▼
 ② 목표 = clip(현재 EEF + action × [0.008, 0.008, 0.005], EEF 범위)
 ▼
 ③ IK: 요청 이동을 1.0 → 0.5 → 0.25 → 0.125배로 줄여 가며 시도
        오차 ≤ 0.01 m인 첫 해를 채택, 관절 변화는 ±0.22 rad로 제한
        모두 실패하면 현재 관절 유지 (ik_rejected=True)
 ▼
 ④ ctrl 설정: IK 3관절 + 손목 0, 0 + 그리퍼 0.20
 ▼
 ⑤ mj_step × 최대 25회. 두 큐브 접촉을 감지하면 즉시 중단
 ▼
 ⑥ success = 접촉 감지 / failure = 큐브 XY·높이 이탈
 ▼
 ⑦ transition = {d_tip, d_cubes, success, failure}
    reward, terms = compute_transition_reward(...) → week11_reward(...)   ← TODO 1
 ▼
 ⑧ terminated = success or failure
    truncated  = (Step ≥ 180) and not terminated
 ▼
 반환 (observation(15,), reward, terminated, truncated, info)
```

### 6.4 `model.learn()` 안에서 일어나는 일

```text
reset
반복 (total_timesteps까지):
  ├ Step < 5,000 → 무작위 Action / 이후 → Actor의 확률적 Action
  ├ env.step(action) → Monitor가 Episode 통계 기록
  ├ 경험을 Replay Buffer에 저장
  ├ Step ≥ 5,000이면 256개 추출 → Critic 2개 → Actor → 엔트로피 계수 → 목표 Critic(tau) 갱신
  ├ Callback 호출: 진행 출력 / 검증·조기 종료 / 체크포인트
  └ Episode가 끝나면 SB3가 자동으로 reset
```

---

## 7. 파일별 코드 설명

### 7.1 `reward.py` — 보상 (TODO 1)

```python
def week11_reward(tip_object_distance, object_target_distance,
                  success, failure, common, config): ...

def compute_transition_reward(transition, common, config):
    return week11_reward(transition["tip_object_distance"], ...)
```

- `week11_reward`: 값 6개를 직접 받는 순수 함수. 환경 없이도 호출해 검사할 수 있다.
- `compute_transition_reward`: `env.step()`이 만든 `transition` 사전을 풀어 위 함수에 넘기는 어댑터. `env.py`는 이 함수를 기본 보상 함수로 쓴다.
- `common` = `config.yaml`의 `task.reward_common`(`success_bonus`, `failure_penalty`), `config` = `reward`(`distance_weight`).
- `PushEnv(reward_function=...)`으로 다른 보상 함수를 주입할 수도 있다(12주차 확장 지점).

### 7.2 `observation.py` — 15차원 관측

```python
class ObservationBuilder:
    def __init__(self):
        self.space = spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)

    def build(self, *, joint_qpos, moving_tip_position, object_position, target_position):
        return np.concatenate((joint_qpos, moving_tip_position,
                               object_position, target_position)).astype(np.float32)
```

- Space 선언과 관측 조립을 한 클래스에 모아 **순서·크기가 어긋나지 않게** 한다.
- 6 + 3 + 3 + 3 = 15. 인자는 모두 키워드 전용(`*`)이라 순서를 바꿔 넣는 실수를 막는다.

### 7.3 `env.py` — `MovingJawContactEnv` (= `PushEnv`)

**`__init__`**

1. `config.yaml`을 읽어 `full_config`(전체), `config`(=`task`), `reward_config`(=`reward`)로 나눈다.
2. `push_scene.xml`을 로드하고 `MjData`를 **두 개** 만든다. `data`는 실제 시뮬레이션, `_ik_data`는 IK 계산 전용 복사본이다. IK를 풀 때 실제 상태를 건드리지 않기 위해서다.
3. 이름으로 site·body·geom·joint·actuator의 id를 찾아 둔다(`_object_id`, 없으면 오류).
4. 초기 관절각, 작업 영역, 제어·Episode 설정을 읽는다.
5. `observation_space`(15), `action_space`(3)를 선언하고 `render_fps = 1 / (0.002 × 25) = 20`으로 맞춘다.

**좌표 변환** — `_base_to_world`, `_world_to_base`
MuJoCo 내부는 world 좌표, 과제는 `base_link` 좌표다. `baseframe` site의 위치·회전으로 변환한다. 관측·보상·info에 나가는 위치는 모두 `_world_to_base`를 거친 값이다.

**`reset(*, seed, options)`**

```text
super().reset(seed) → 시나리오 선택 → 최대 20회 시도:
  큐브 XY 추출(고정값) → 최소 중심 거리 0.06 m 검사
  → mj_resetData → 로봇 초기 관절·ctrl 설정 → 큐브 배치(_place_cube) → mj_forward
  → 큐브끼리 이미 닿았거나 바닥 외 물체와 닿으면 재시도
  → 큐브 뒤쪽 지점에 대한 IK 오차가 0.001 m 초과면 재시도
→ 카운터 초기화 → (observation, info)
```

- `_place_cube`는 freejoint의 `qpos` 7개(위치 3 + 쿼터니언 4)를 직접 쓴다. yaw만 회전하므로 쿼터니언은 `(cos(yaw/2), 0, 0, sin(yaw/2))`.
- 이번 주는 범위가 한 점이라 무작위성이 없지만, 코드는 12주차 무작위 배치를 그대로 받을 수 있는 구조다.

**`_solve_ik(target)`** — Damped Least Squares(위치 전용)

```python
error   = target_world - 현재 gripperframe 위치
reduced = jacobian[:, IK 3관절]                     # 3×3
delta   = reduced.T @ solve(reduced @ reduced.T + damping² · I, error)
delta   = clip(delta, -0.12, 0.12)                  # 반복당 변화량 제한
joint_target = clip(joint_target + delta, 관절 범위)
```

- 최대 45회 반복, 오차 < 1e-5 m이면 조기 종료. 감쇠 계수 0.015가 특이점 근처의 발산을 막는다.
- 반환값은 (관절 목표, 남은 오차). 호출한 `step()`이 오차로 채택 여부를 정한다.

**접촉 판정**

| 메서드 | 판정 |
|---|---|
| `_cubes_touching()` | `data.contact`에 (초록 geom, 파랑 geom) 쌍이 있는가 → **성공 조건** |
| `_target_has_robot_contact()` | 파란 큐브가 바닥·초록 큐브 외의 것과 닿았는가 → 기록용 |
| `_cube_has_invalid_reset_contact()` | reset 직후 큐브가 바닥 외 물체와 닿았는가 → 재시도 |

**`_failure_state()`** — 초록(`object`)·파랑(`target`) 순서로 XY 범위와 높이를 검사해 `(True, "object_outside_workspace")` 같은 값을 돌려준다.

**렌더링** — `render_camera(name)`은 `rgb_array` 모드에서만 `mujoco.Renderer`로 640×480 영상을 만든다. `_project_pixel`은 3D 점을 카메라 영상 좌표로 투영해 Viewer 표시에 쓴다(정책 입력과는 무관).

### 7.4 `train.py` — 생성·학습·저장 (TODO 2~4)

```text
① 인자 파싱 (--run-name, --steps, --seed, --device, --threads) → torch 스레드 수 지정
② config.yaml 로드 → total_steps, seed, device, run_name 결정
③ outputs/<run_name>/ 생성. 이미 있으면 FileExistsError ("Run already exists")
④ run_config.yaml 저장 (적용 설정 기록)
⑤ 환경 2개: raw_environment → Monitor로 감싼 environment(학습용) / evaluation_environment(검증용)
⑥ 콜백 정의: ProgressCallback, ValidationCallback
⑦ [TODO 2] model = SAC(...)
⑧ model.environment_version 기록 → CallbackList 구성
⑨ [TODO 3] model.learn(...)
⑩ [TODO 4] model.save(.../policy_final)
⑪ latest_run.txt 갱신 → metrics.json 저장 → "Saved: ..." 출력
⑫ finally: 두 환경 close()
```

**`ValidationCallback`의 조기 종료 논리**

```text
최근 학습 Episode 성공률(ep_success_buffer, 20개 이상 모였을 때)
  < 99%  → 게이트 재장전, 계속 학습
  ≥ 99% 이고 게이트가 장전됨 → 별도 환경에서 결정적 20 Episode 검증 (seed 50000)
        → 최고 기록이면 policy_best 저장
        → 검증도 ≥ 99%면 policy_converged 저장 후 False 반환(조기 종료)
```

- 게이트(`rollout_gate_armed`)는 성공률이 99% 아래로 내려갔다 다시 올라올 때만 재검증하도록 해서 매 Step 검증을 막는다.
- 검증에 쓴 시간은 `excluded_validation_seconds`로 따로 재서 `pure_training_seconds`에서 뺀다.
- `policy_best.zip`은 **검증이 한 번이라도 실행됐을 때만** 생긴다. 없는 것이 정상일 수 있다.

**`metrics.json`**: `steps_completed`, `gradient_updates`(=`model._n_updates`), `training_seconds`, `stopped_early`, `stop_step`, `policy_final` 경로 등.

### 7.5 `evaluate.py` — 불러오기·평가 (TODO 5~6)

| 함수 | 역할 |
|---|---|
| `resolve_run_directory(run)` | `outputs/<run>` 또는 `latest_run.txt`가 가리키는 폴더 |
| `resolve_policy_path(run, policy, model)` | `--policy` 직접 경로 우선, 없으면 `policy_<final\|best>.zip` |
| `evaluate_model(model, environment, episodes_per_scenario, seed)` | 평가 반복문. **[TODO 6]** 포함. `(summary, rows)` 반환 |
| `print_summary`, `write_evaluation` | 요약 출력, JSON·CSV 저장 |
| `main()` | 인자 파싱 → **[TODO 5]** 로드 → 환경 버전 검사 → 평가 → 저장 |

- `evaluate_model`은 `train.py`의 `ValidationCallback`도 호출한다. 그래서 **TODO 6은 학습 중 검증에도 쓰인다.**
- 회차별 Seed는 `seed + 시나리오 번호 × 1000 + 회차`.
- 회차마다 성공, 실패 원인, Step 수, 최종·최소 중심 거리, EEF 이동 경로 길이, IK 거부 횟수, 큐브 이동 거리를 기록한다.
- 로드 직후 `model.environment_version`과 환경의 버전을 비교해, 관측 의미가 달라진 옛 정책 사용을 막는다.

### 7.6 `inspect_environment.py` — 환경 점검

`check_env`(Gymnasium API 검사) → `reset(seed=12)` → Space·초기 위치 출력 → Action `[0.5, -0.25, 1.0]`으로 한 Step → 요청·실측 이동, 보상 항목, 종료 신호 출력 → `Gymnasium API OK`.

### 7.7 `visualize.py` / `visualization.py` — Viewer

- 창 3개: MuJoCo 자유 시점(`launch_passive`) + OpenCV `Top camera keypoints` + `Front camera`.
- `--run` 또는 `--policy`가 있으면 정책을 불러와 결정적 행동으로 움직이고, 없으면 고정 초기 장면만 보여 준다.
- 키: `R` 리셋, `Space` 일시정지(Top/Front 창에서), `Q`/`Esc` 종료. Episode가 끝나면 자동 리셋.
- `top_view_bgr`는 Top 영상 위에 spawn 영역, 큐브 윤곽, `EEF/IK`·`MOVING TIP` 표시와 Action·보상·성공 여부 텍스트를 그린다.

---

## 8. TODO 1~6 정답과 해설

### TODO 1 — 보상 계산 (`reward.py`)

```python
distance_cost = -float(config["distance_weight"]) * (
    float(tip_object_distance) + float(object_target_distance)
)
terms = {
    "distance": distance_cost,
    "success": float(common["success_bonus"]) if success else 0.0,
    "failure": -float(common["failure_penalty"]) if failure else 0.0,
}
return sum(terms.values()), terms
```

- 항목 이름은 `distance`·`success`·`failure` 세 개.
- `failure_penalty`는 설정에 양수 100으로 들어 있으므로 **부호를 직접 붙인다.**
- 성공·실패를 `if/elif`로 묶지 않는다. 같은 Step에 둘 다 참이면 둘 다 더해야 한다.
- 총합은 `sum(terms.values())`로 구해 "총 보상 = 항목 합"이 항상 성립하게 한다(검사 스크립트가 이를 확인).

### TODO 2 — SAC 생성 (`train.py`)

```python
model = SAC(
    "MlpPolicy",
    environment,
    learning_rate=float(training["learning_rate"]),
    buffer_size=int(training["buffer_size"]),
    learning_starts=int(training["learning_starts"]),
    batch_size=int(training["batch_size"]),
    gamma=float(training["gamma"]),
    tau=float(training["tau"]),
    ent_coef=training["ent_coef"],
    train_freq=int(training["train_freq"]),
    gradient_steps=int(training["gradient_steps"]),
    seed=seed,
    device=device,
    policy_kwargs={"net_arch": list(training["network_architecture"])},
    tensorboard_log=str(tensorboard),
    verbose=1,
)
```

- `environment`는 `raw_environment`가 아니라 **`Monitor`로 감싼 것**을 넘긴다. 그래야 `training.monitor.csv`와 성공률 로그가 기록된다.
- `int()`/`float()`는 YAML 값을 API가 요구하는 자료형으로 맞춘다.
- `ent_coef`는 문자열 `"auto_0.1"` 그대로 전달한다(숫자로 변환하지 않는다).
- `seed`·`device`는 `training[...]`이 아니라 CLI 옵션이 반영된 **지역 변수**를 쓴다.
- 반드시 `model` 변수에 받는다. 바로 다음 줄이 `model.environment_version = ...`이다.

### TODO 3 — 학습 실행 (`train.py`)

```python
model.learn(
    total_timesteps=total_steps,
    callback=callbacks,
    log_interval=int(training["log_interval"]),
    tb_log_name="sac_push",
)
```

- `total_steps`는 `--steps`가 있으면 그 값, 없으면 60000.
- 인자 이름은 `callback`(단수)이고, 넘기는 값은 `CallbackList`인 `callbacks`다.
- `log_interval=20`은 **Episode 20개마다** 로그를 낸다는 뜻이다. `Progress 5000/...`은 Step 기준의 별도 콜백 출력이다.

### TODO 4 — 최종 정책 저장 (`train.py`)

```python
model.save(str(run_directory / "policy_final"))
```

- `.zip`은 SB3가 붙인다 → `outputs/<run>/policy_final.zip`.
- 이 줄 뒤의 코드가 `metrics.json`에 같은 경로를 기록하고, 평가는 기본으로 이 파일을 찾는다.

### TODO 5 — 정책 불러오기 (`evaluate.py`)

```python
model = SAC.load(str(policy_path), env=environment, device="cpu")
```

- `SAC`는 `main()` 첫 줄에서 이미 import되어 있다.
- `env`를 연결하면 저장된 모델과 환경의 Space가 맞는지도 검사된다.

### TODO 6 — 평가 행동 예측 (`evaluate.py`)

```python
action, _state = model.predict(observation, deterministic=True)
```

- 반환값은 튜플이다. `action = model.predict(...)`로 받으면 튜플이 통째로 `step()`에 들어가 오류가 난다.
- `deterministic=True`가 없으면 평가할 때마다 결과가 달라진다.
- 바로 다음 줄의 `environment.step(action)`이 `observation`을 갱신하므로 반복문이 이어진다.

---

## 9. 실습 진행 순서

> 모든 명령은 WSL2 Ubuntu 22.04, `~/week11/sac_lab`에서 `python3 -m week11_sac.<모듈>` 형식으로 실행한다.

### 9.1 준비

```bash
cd ~ && rm -rf ~/week11 && mkdir -p ~/week11/sac_lab
cd ~/week11 && python3 -m venv .venv && source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1          # __pycache__ 생성 방지 (새 터미널마다 다시)
# source_code의 내용을 ~/week11/sac_lab/에 배치
cd ~/week11/sac_lab
python3 -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --no-cache-dir -r requirements.txt
python3 -m pip check                      # "No broken requirements found."
```

- `ensurepip is not available` → `sudo apt install python3.10-venv libgl1 libglfw3` 후 venv부터 다시.
- CPU용 PyTorch를 **먼저** 설치한다. 관측 15개·작은 MLP·단일 환경이라 CPU 스레드 1개로 충분하다.

### 9.2 초기 환경 확인 (TODO 전)

```bash
python3 -m week11_sac.visualize           # 자유 시점·Top·Front 세 창, R 리셋, Q 종료
```

### 9.3 TODO 작성 → 검사

```bash
# 1) Source 검사: *.py 8개, 'raise NotImplementedError' 없음, compile 성공 → "Source check OK"
# 2) API·한 Step
python3 -m week11_sac.inspect_environment # "Gymnasium API OK"
# 3) Reward 검사: week11_reward(0.05, 0.08, False, False, ...) → "Reward check OK"
```

### 9.4 확인 학습 (smoke)

```bash
rm -rf outputs/smoke
python3 -m week11_sac.train --run-name smoke --steps 5200 --device cpu --threads 1
python3 -m week11_sac.evaluate --run smoke --episodes-per-scenario 2
```

목적은 성능이 아니라 **학습 → 저장 → 불러오기 → 평가가 끊기지 않고 이어지는지** 확인하는 것이다. 평가 0/2는 정상이다.

### 9.5 본 학습과 평가

```bash
python3 -m week11_sac.train --run-name baseline --steps 60000 --device cpu --threads 1
python3 -m week11_sac.evaluate --run baseline          # 40 Episode
python3 -m week11_sac.visualize --run baseline         # 학습한 정책 움직임
```

학습 중 관찰할 로그: `Progress`, `rollout/ep_rew_mean`(평균 Return), `rollout/success_rate`(최근 학습 성공률), `train/actor_loss`, `train/critic_loss`.

### 9.6 기록과 정리

| 항목 | 기록할 내용 |
|---|---|
| 과제·입출력 | 시작 배치, 성공·실패, 15차원 특징의 예, XYZ 요청 |
| 보상 | 세 항목과 0.05 m·0.08 m 예제의 총합 |
| 학습 | 실제 Step·갱신 수·학습 시간·Return 변화 |
| 평가 | 40회 성공 수·성공률·최종 거리·종료 원인 |
| Viewer | 제어점·관측점, 로봇과 큐브의 움직임 |
| 해석 | 두 거리 비용이 각각 유도하는 행동, 보상 개선 아이디어 |

```bash
deactivate && cd ~ && rm -rf ~/week11     # 소스·기록·정책을 개인 저장소에 백업한 뒤 실행
```

---

## 10. 검증 기준값

### 10.1 초기 환경 (`reset(seed=12)`)

| 항목 | 예상 값 |
|---|---|
| Observation | `Box(-inf, inf, (15,), float32)` |
| Action | `Box(-1.0, 1.0, (3,), float32)` |
| 초록 XYZ | 약 `[0.31, 0.03, 0.0081504]` |
| 파랑 XYZ | 약 `[0.37, 0.0, 0.0081504]` |
| EEF XYZ | 약 `[0.225, 0, 0.015]` |
| 손끝 XYZ | 약 `[0.253904, -0.001521, 0.034919]` |

### 10.2 `inspect_environment` 한 Step

| 항목 | 값 |
|---|---|
| Action XYZ | `[0.5, -0.25, 1.0]` |
| 요청 이동 | `[0.004, -0.002, 0.005]` m |
| 실측 이동 | 약 `[0.003332, -0.001876, 0.004188]` m |
| reward terms | `{'distance': -1.379930, 'success': 0.0, 'failure': 0.0}` |
| 보상 | 약 `-1.379930` |
| 종료 신호 | `False False` |

> 이 표는 완성한 코드로 직접 실행해 안내서의 값과 일치함을 확인했다.
> 역산하면 `d_tip + d_cubes ≈ 0.138 m`다(초기 손끝–초록 약 0.070 m, 초록–파랑 약 0.067 m).

### 10.3 Reward 검사

| 입력 | 결과 |
|---|---|
| `(0.05, 0.08, False, False)` | `-1.3`, `{'distance': -1.3, 'success': 0.0, 'failure': 0.0}` |
| `(0.05, 0.08, True, False)` | `98.7` |
| `(0.05, 0.08, False, True)` | `-101.3` |

### 10.4 학습·평가 (안내서의 검증 PC 측정값)

| 실행 | 결과 |
|---|---|
| smoke 5,200 Step | 갱신 200회, 약 36초, 평가 0/2 |
| baseline 60,000 Step | 갱신 55,000회, 약 586초, 조기 종료 없음, `policy_best` 없음 |
| baseline 40 Episode 평가 | 40/40 성공, 평균 55 Step, 최종 중심 거리 약 0.028619 m |

> 이 절의 값은 `practice.md`와 강의 자료에 적힌 검증 PC 결과다. PC 성능·라이브러리·연산 차이에 따라 시간과 결과는 달라질 수 있으므로 자신의 실행 결과를 따로 기록한다.

---

## 11. 자주 틀리는 부분과 문제 해결

### 11.1 코드에서 자주 틀리는 부분

| 실수 | 결과 | 올바른 방법 |
|---|---|---|
| 거리 비용의 부호를 빼먹음 | 멀어질수록 보상이 커져 큐브에서 달아나는 정책 | `-distance_weight × (d_tip + d_cubes)` |
| `failure_penalty`를 그대로 더함 | 실패하면 +100 | `-float(common["failure_penalty"])` |
| 성공·실패를 `if/elif`로 처리 | 동시 발생 시 한쪽 누락 | 두 항목을 독립적으로 계산 |
| 총합을 따로 계산 | 항목 합과 불일치 | `sum(terms.values())` |
| `SAC(...)` 결과를 변수에 안 받음 | 다음 줄에서 `NameError` | `model = SAC(...)` |
| `raw_environment`를 SAC에 전달 | Monitor CSV·성공률 로그 없음 | `environment`(Monitor) 전달 |
| `learn(callbacks=...)` | `TypeError` | 인자 이름은 `callback` |
| `save(".../policy_final.zip")` 외 다른 이름 | 평가가 정책을 못 찾음 | `policy_final` |
| `model.load(...)`를 반환값 없이 호출 | 가중치가 복원되지 않음 | `model = SAC.load(...)` |
| `action = model.predict(...)` | 튜플이 `step()`에 전달됨 | `action, _ = model.predict(...)` |
| `deterministic=True` 누락 | 평가 결과가 실행마다 달라짐 | 평가에서는 항상 지정 |

### 11.2 실행 중 문제 해결

| 증상 | 확인할 내용 |
|---|---|
| 모듈을 찾지 못함 | `~/week11/sac_lab`에서 `python3 -m week11_sac.<모듈>`로 실행했는지 |
| `NotImplementedError` | 해당 TODO 작성 여부 |
| 라이브러리 import 오류 | venv 활성화, requirements 설치, `pip check` |
| `Run already exists` | 새 `--run-name` 사용(또는 smoke는 `rm -rf outputs/smoke`), 평가도 같은 이름 |
| `policy_best.zip` 없음 | 검증이 실행되지 않은 것. 기본 `policy_final.zip` 사용 |
| 정책 버전 불일치 오류 | 이번 환경(`direct_xyz15_distance_v1`)에서 새로 학습한 정책 사용 |
| Step이 짧게 종료 | `failure_reason`과 큐브 위치·높이 |
| 성공률 0% | `gradient_updates > 0`인지, 학습 로그, 접촉 여부, 보상 항목 부호 |
| Viewer 표시 오류 | WSLg, OpenGL, `libgl1`, `libglfw3`, OpenCV GUI |
| Viewer 키 무반응 | 키를 받을 창 선택, Space는 Top/Front 창에서 |

### 11.3 헷갈리기 쉬운 구분

- `total_timesteps`(예산, Step 수) ≠ Episode 수 ≠ `buffer_size`(용량).
- `log_interval`(Episode 기준) ≠ `progress_interval`(Step 기준).
- 학습 중 `rollout/success_rate`(확률적 행동, 최근 Episode) ≠ 최종 평가 성공률(결정적 행동, 40회).
- 성공 = 물리 **접촉**이지 중심 거리 0이 아니다.
- `gripperframe`(제어점) ≠ `moving_fingertip`(관측점).
- `critic_loss` 감소 ≠ 과제 성공.
- 평가 40/40은 **고정 배치·Seed 12** 조건의 결과다. 다른 배치에서의 성능은 따로 확인해야 한다.

---

## 12. 복습 질문

1. 10주차 Random Agent와 이번 SAC Agent의 가장 큰 차이는? → 경험을 저장하고 그것으로 행동 선택 방법(가중치)을 갱신한다.
2. Actor와 Critic의 입력·출력은? → Actor: 관측 15 → 행동 분포 → Action 3 / Critic: 관측 + 행동 → Q값.
3. SAC가 Critic을 두 개 쓰는 이유는? → 목표값 계산에 작은 쪽을 써서 Q값 과대 추정을 줄이려고.
4. 엔트로피를 보상과 함께 고려하면 무엇이 좋은가? → 행동 다양성이 유지되어 탐색이 일찍 멈추지 않는다.
5. Replay Buffer에 저장하는 것 다섯 가지는? → 현재 관측, 행동, 보상, 다음 관측, 종료 정보.
6. Off-policy란? → 과거 정책이 모은 경험도 학습에 재사용하는 방식.
7. 5,200 Step 학습의 갱신 횟수와 그 이유는? → 200회. 처음 5,000 Step은 수집만 하고 이후 Step당 1회 갱신.
8. `buffer_size=60000`과 `total_timesteps=60000`은 각각 무엇을 정하나? → 경험 저장 용량 / 환경 상호작용 예산.
9. `gamma=0.99`, `tau=0.005`의 의미는? → 미래 보상 할인율 / 목표 Critic을 현재 Critic 쪽으로 0.5%씩 섞는 비율.
10. Observation 15개의 구성과 `observation[9:12]`의 의미는? → 관절각 6 + 손끝 XYZ + 초록 XYZ + 파랑 XYZ / 초록 큐브 중심 XYZ.
11. Action `[0.5, -0.25, 1.0]`의 요청 이동량은? → `[+4, -2, +5]` mm.
12. 두 거리가 0.05 m, 0.08 m일 때 진행 중·성공·실패 보상은? → −1.3 / 98.7 / −101.3.
13. Sparse 보상만 쓰면 왜 학습이 어려운가? → 우연히 접촉하기 전까지 학습 신호가 없다.
14. `d_tip`과 `d_cubes`는 각각 어떤 행동을 유도하나? → 초록 큐브로 접근 / 초록 큐브를 파란 큐브 쪽으로 이동.
15. Return이 높은데 성공률이 0%일 수 있는 이유는? → 큐브 가까이 머물러 거리 비용만 줄이고 접촉은 못 시키는 정책일 수 있다.
16. 성공·실패·시간 제한은 각각 어떤 신호인가? → 성공·실패 `terminated`, 180 Step `truncated`.
17. 성공했는데 중심 거리가 0이 아닌 이유는? → 성공은 표면의 물리 접촉이고 큐브 한 변이 25 mm라 중심은 떨어져 있다.
18. IK는 몇 개 관절로, 어떤 방법으로 푸나? → 3개(pan·lift·elbow), 위치 전용 Damped Least Squares.
19. `Monitor`가 기록하는 것은? → Episode의 `r`·`l`·`t`와 `is_success`·`scenario`·`failure_reason`.
20. 콜백의 `_on_step()`이 `False`를 반환하면? → 학습이 조기 종료된다.
21. 평가에서 `deterministic=True`를 쓰는 이유는? → 행동 선택을 일정하게 해 같은 조건의 결과를 비교하려고.
22. 고정 배치에서 40회 평가 결과가 모두 같을 수 있는 이유는? → 초기 상태도 정책 행동도 결정적이기 때문.
23. `environment_version`을 모델에 저장하는 이유는? → 관측 의미·제어가 달라진 환경에 옛 정책을 쓰는 것을 막으려고.
24. `learn()`과 `predict()` 중 가중치를 바꾸는 것은? → `learn()`만.

---

## 13. 10 → 11 → 12주차 연결

| | 10주차 | 11주차 | 12주차 |
|---|---|---|---|
| Agent | Random (`sample`) | **SAC** 학습 정책 | SAC |
| Observation | XY 6차원 | 관절각 + XYZ **15차원** | 이어서 확장 |
| Action | XYZ 3차원 | 동일 | 동일 |
| Reward | 행동 비용만 | **거리 비용 + 성공·실패** | 보상 확장 |
| 성공 조건 | 없음 | 두 큐브 접촉 | 두 큐브 접촉 |
| 초기 배치 | XML 고정 | 고정 (`diagonal_right`) | **다양화** |
| 학생이 작성 | Space·reset·step | 보상·SB3 호출 5곳 | 보상 설계 |

- 10주차에서 만든 **환경 인터페이스(Space·reset·step·종료 신호·info)** 는 그대로다. 바뀐 것은 Agent와 Observation·Reward다.
- `env.py`에는 이미 12주차용 구조가 들어 있다: 시나리오별 spawn 영역과 확률, 무작위 yaw 옵션, `reward_function` 주입, `straight`/`diagonal_left`/`diagonal_right` 분류.
- 이번 주 40/40은 한 배치에서만 통하는 결과일 수 있다. 12주차는 "배치가 바뀌어도 통하는가"와 "그러려면 보상에 무엇을 더해야 하는가"를 다룬다.

---

### 참고 링크

- [SAC 공식 문서 (SB3)](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
- [SAC 2.9.2 API](https://stable-baselines3.readthedocs.io/en/v2.9.2/modules/sac.html)
- [Monitor](https://stable-baselines3.readthedocs.io/en/v2.9.2/common/monitor.html) · [Callback](https://stable-baselines3.readthedocs.io/en/v2.9.2/guide/callbacks.html)
- [SB3 기본 사용 예제 (학습·저장·불러오기)](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#basic-usage-training-saving-loading)

> 작성: 11주차 제공 자료(`lecture.md`, `practice.md`, `week11_SAC_reward_shaping.pptx`, `source_code/`)를 바탕으로 Claude가 정리했다. TODO 정답은 저장소에 커밋한 코드와 같다. 10.1~10.3절은 실제로 실행해 확인한 값이고, 10.4절(학습·평가)은 직접 실행하지 못해 안내서의 검증 PC 측정값을 옮겼다.
