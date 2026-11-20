# [10주차 총정리] 강화학습 환경과 Gymnasium의 기초

> 범위: `lecture.md`, `practice.md`, `source_code/` 전체 (`gymnasium_lab_env.py`, `mujoco_simulation.py`, `inspect_environment.py`, `random_agent.py`, `contact_demo.py`, `task.yaml`, `models/`)
> 한 줄 요약: **9주차 MuJoCo 장면을 그대로 두고, 그 위에 Gymnasium 규약(`reset`/`step`/Space)을 씌워 "Agent가 쓸 수 있는 환경"을 만든 뒤, 무작위 Agent로 Episode가 제대로 도는지 확인한다.** 정책 학습은 하지 않는다(11주차 SAC).

---

## 목차

1. [전체 흐름 한눈에 보기](#1-전체-흐름-한눈에-보기)
2. [핵심 개념](#2-핵심-개념)
3. [이번 환경의 설계 명세](#3-이번-환경의-설계-명세)
4. [코드 구조와 호출 흐름](#4-코드-구조와-호출-흐름)
5. [파일별 코드 설명](#5-파일별-코드-설명)
6. [TODO 1~6 정답과 해설](#6-todo-16-정답과-해설)
7. [실습 진행 순서](#7-실습-진행-순서)
8. [검증 기준값](#8-검증-기준값)
9. [자주 틀리는 부분과 문제 해결](#9-자주-틀리는-부분과-문제-해결)
10. [복습 질문](#10-복습-질문)
11. [9 → 10 → 11 → 12주차 연결](#11-9--10--11--12주차-연결)

---

## 1. 전체 흐름 한눈에 보기

### 1.1 수업의 위치

```text
9주차 : 사람이 관절 목표를 정해 MuJoCo/로봇에 전달
10주차: MuJoCo 장면을 Gymnasium 환경으로 감싸기 → Random Agent로 실행 확인   ← 지금
11주차: 같은 환경 구조로 SAC 정책 학습 (고정 배치, 두 큐브 접촉이 성공 조건)
12주차: 초기 배치 무작위화 + 과제에 맞는 보상 설계
```

### 1.2 이번 주에 답할 수 있어야 하는 질문

| 질문 | 짧은 답 |
|---|---|
| Agent와 Environment는 무엇을 주고받나? | Agent → Action, Environment → Observation·Reward·종료 신호·info |
| Gymnasium과 MuJoCo는 어떻게 다른가? | MuJoCo = 물리 계산, Gymnasium = 그 계산을 `reset/step` 공통 형식으로 연결 |
| `reset()`과 `step()`은 무엇을 반환하나? | `reset` → `(obs, info)` 2개, `step` → `(obs, reward, terminated, truncated, info)` 5개 |
| Space를 왜 선언하나? | 주고받는 값의 **범위·모양·자료형**을 양쪽이 확인할 수 있게 하려고 |
| Episode 하나를 돌리려면? | `reset` → (`sample` → `step` → Return 누적)을 `terminated or truncated`까지 반복 |

### 1.3 실습 전체 흐름

```text
이전 분반 ~/week10 삭제 → 새 venv → 제공 파일 배치 → 패키지 설치·버전 확인
→ TODO 1~6 작성
→ 빈칸·문법 검사 → inspect_environment(한 Step) → check_env(API 검사)
→ random_agent 1 Episode → 3 Episode·Seed 비교
→ Viewer 확인(--render) → (선택) contact_demo
→ 결과 기록 → deactivate → ~/week10 전체 삭제
```

---

## 2. 핵심 개념

### 2.1 강화학습(Reinforcement Learning)

- 상호작용 경험을 이용해 **앞으로 받을 보상의 누적값이 커지도록 행동 선택 방법(정책)을 개선**하는 학습 방식.
- "지금 당장의 보상"과 "앞으로의 결과까지 고려한 보상"은 다를 수 있다.
  - 예: 물체를 밀려고 먼저 물체 **뒤쪽으로 이동**하는 행동은 당장은 물체를 움직이지 않지만 이후에 도움이 된다.

### 2.2 용어

| 용어 | 정의 | 이번 실습의 예 |
|---|---|---|
| **Agent** | 행동을 선택하는 주체 | `random_agent.py` (무작위 선택) |
| **Environment** | 행동을 받아 결과를 돌려주는 대상. 물리뿐 아니라 "무엇을 알려 줄지·어떤 보상을 줄지"도 포함 | `GymnasiumLabEnv` |
| **Observation** | 환경이 Agent에게 골라서 전달하는 정보 (전체 상태 ≠ 관측) | EEF·초록·파랑의 XY 6개 |
| **Action** | Agent가 환경에 보내는 명령 | EEF XYZ 이동 비율 3개 |
| **Reward** | 행동 이후 환경이 계산한 수치 피드백 (음수 가능) | `-0.01 × ‖action‖²` |
| **Policy** | 상황 → 행동을 정하는 규칙. 신경망·사람 규칙·무작위 모두 가능 | 이번엔 무작위 |
| **Episode** | `reset()`부터 종료까지의 한 회차 | 최대 150 Step |
| **Return** | 한 Episode 동안 Reward의 합 | 약 `-1.45` |

```text
                 ① Observation
          ┌──────────────────────────┐
          ↓                          │
       Agent                    Environment
          │                          ↑
          └──── ② Action ────────────┘
          ↑                          │
          └── ③ 새 Observation·Reward·종료 여부 ──┘
```

> **주의:** 무작위 행동을 반복하기만 해서는 정책이 저절로 학습되지 않는다. Random Agent는 **관측·행동·보상이 올바르게 오가는지 확인하는 용도**다.

### 2.3 Gymnasium

- **강화학습 환경을 만들고 사용하는 공통 인터페이스**를 제공하는 Python 라이브러리.
- 인터페이스 = "어떤 메서드를 어떤 인자로 호출하고, 어떤 값을 돌려받는가"에 대한 약속.
- 로봇·게임·차량 등 내부가 달라도 호출하는 쪽은 항상 같은 형식을 쓴다.

```python
observation, info = environment.reset()
observation, reward, terminated, truncated, info = environment.step(action)
```

### 2.4 역할 분담

| 구성 요소 | 담당 |
|---|---|
| **MuJoCo** | 로봇·큐브의 다음 물리 상태 계산 (관절 운동, 접촉) |
| **Gymnasium 환경** | Action을 받아 물리 계산을 실행하고 Observation·Reward·종료 여부를 반환 |
| **Agent** | 허용된 Action 중 무엇을 실행할지 결정 |
| **학습 알고리즘** | 경험으로 정책을 개선 (**이번엔 사용 안 함**) |

> Gymnasium을 쓴다고 학습이 자동으로 되는 것이 아니다. **환경 실행 코드**와 **정책 학습 코드**는 별개다.

### 2.5 `reset()` / `step()` 반환값

**`reset(seed=...)` → 2개**

| 순서 | 값 | 의미 |
|---|---|---|
| 1 | `observation` | 첫 관측 (첫 행동을 고르기 전에 상황을 알아야 하므로 필요) |
| 2 | `info` | 추가 정보 Dictionary |

- `reset()`에는 Action을 넘기지 않는다.

**`step(action)` → 5개**

| 순서 | 값 | 의미 |
|---|---|---|
| 1 | `observation` | 행동 **이후**의 새 관측 |
| 2 | `reward` | 이번 행동의 보상 |
| 3 | `terminated` | 환경이 정한 **종료 조건** 도달 여부 |
| 4 | `truncated` | **시간 제한** 등으로 중단해야 하는지 |
| 5 | `info` | 점검·분석용 추가 정보 |

- 같은 변수명 `observation`에 다시 받으면 그 변수는 이제 "행동 이후 관측"을 가리킨다.

### 2.6 `terminated`와 `truncated`를 나누는 이유

- "과제 상 끝날 상황이 됐다"와 "정해 둔 시간이 다 됐다"는 의미가 다르다.
- 학습 알고리즘(11주차)은 이 둘을 다르게 처리한다. `truncated`는 "그 뒤로도 계속될 수 있었는데 잘렸다"는 뜻이라서 미래 가치를 0으로 보면 안 된다.

| 상황 | `terminated` | `truncated` |
|---|---|---|
| 계속 진행 가능 | False | False |
| 어느 큐브든 중심 XY가 작업영역 밖 | **True** | False |
| 이탈 없이 150 Step 도달 | False | **True** |
| 150번째 Step에서 동시에 이탈 | **True** | False (**이탈 우선**) |

- 어느 쪽이든 True가 되면 회차를 끝내고 다음 회차는 `reset()`으로 시작한다.

### 2.7 Space

- 값의 **범위·배열 모양·자료형**을 선언한 규격. 실제 값 하나가 아니라 "값이 따라야 할 규칙"이다.
- `observation_space`: 환경이 **반환할** 관측의 규격.
- `action_space`: 환경이 **받을** 행동의 규격.
- 연속값 → `spaces.Box(low, high, shape, dtype)`.

| 인자 | 의미 |
|---|---|
| `low`, `high` | 각 성분의 최소·최대 |
| `shape=(3,)` | 값 3개짜리 **1차원** 배열 (3차원 배열이 아님. `(3,)`는 Tuple) |
| `dtype=np.float32` | 원소 자료형 |

| 메서드 | 기능 |
|---|---|
| `contains(x)` | 범위·모양·자료형이 규격에 맞는지 → `True/False` |
| `sample()` | 규격에 맞는 값을 **무작위로** 하나 추출 (좋은 행동을 찾아 주지는 않음) |
| `seed(n)` | Space 자체의 난수 생성기 초기화 |

**Observation을 `-inf ~ inf`로 선언한 이유**
- 큐브가 작업영역을 벗어나도 실제 위치를 **잘라내지 않고 그대로** 반환하기 위해서.
- "정상 영역이 무한하다"는 뜻은 아니다. 작업영역은 **종료 조건**으로 따로 검사한다.
- `check_env`가 "infinity" 경고를 띄우지만 의도된 선언이므로 실패가 아니다.

### 2.8 Seed와 재현성

```python
observation, info = environment.reset(seed=10)   # 환경의 난수 생성기
environment.action_space.seed(10)                # Action Space의 난수 생성기
```

- 같은 Action 순서를 재현하려면 **두 줄 모두** 필요하다. `reset(seed)`만으로 `sample()` 순서가 고정된다고 가정하지 않는다.
- 이번 초기 배치는 XML로 **고정**이므로 Seed를 바꿔도 큐브 위치는 같고, **Action 순서만** 바뀐다.
- 같은 라이브러리 버전·설정·Seed를 기준으로 결과를 비교한다.

### 2.9 render / close

| `render_mode` | `render()` 동작 |
|---|---|
| `None` | 화면 없음, `None` 반환 (물리 계산은 똑같이 수행) |
| `"human"` | WSLg Viewer에 표시. 제공 환경은 `reset/step` 안에서 자동 호출 |
| `"rgb_array"` | `top_view` 카메라로 480×640×3 RGB 배열 반환 |

- **`render()`만 반복해도 물리 시간은 흐르지 않는다.** 물리 진행은 `step()` 안에서만 일어난다.
- `close()`는 Viewer·렌더러 자원을 정리한다. 오류·중단에도 정리되도록 `try/finally`로 감싼다.

### 2.10 시간 단위 구분

| 단위 | 값 |
|---|---|
| MuJoCo 물리 1회 (`timestep`) | 0.002 s |
| 환경 Step 1회 = `frame_skip` 25회 | 0.05 s (→ `render_fps = 20`) |
| 1 Episode = 150 Step | 물리 시간 7.5 s (실제 실행 시간은 PC·화면 여부에 따라 다름) |

> 환경 Step 1회, MuJoCo 물리 1회, 화면 갱신 1회는 모두 다른 개념이다.

---

## 3. 이번 환경의 설계 명세

### 3.1 장면 (`models/scene.xml`, 9주차 원본 그대로)

- 모델 이름: `Week09 PhysicAI Arm` (Viewer 제목)
- 노란 로봇 팔(SO-101 계열, 관절 6개), 작업대·모니터·기둥, 검정·흰색 바구니
- **초록 큐브** `object_cube`: 25 mm, 질량 11 g, 초기 위치 (0.310, 0.030)
- **파란 큐브** `target_cube`: 25 mm, 질량 11 g, 초기 위치 (0.370, 0.000)
- 각 큐브는 `freejoint` → qpos 7개(위치 3 + 쿼터니언 4), qvel 6개
- `box size="0.0125 ..."`는 **반 길이**(half-size). 한 변 = 25 mm
- 모델 크기: `nq = 6 + 7 + 7 = 20`, `nv = 6 + 6 + 6 = 18`, `nu = 6`, `dt = 0.002 s`
- 목표 표시(주황 원)·성공 판정은 **없다**

### 3.2 Observation — `Box(-inf, inf, (6,), float32)`

```text
[eef_x, eef_y, green_x, green_y, blue_x, blue_y]
초기값 ≈ [0.1999, 0.0000, 0.3100, 0.0300, 0.3700, 0.0000]
```

| 인덱스 | 의미 |
|---|---|
| `[0:2]` | EEF(그리퍼 끝 `gripperframe` site) XY |
| `[2:4]` | 초록 큐브 중심 XY |
| `[4:6]` | 파란 큐브 중심 XY |

- 좌표계: 로봇 **`base_link`**, 단위: **m** (픽셀·rad 아님).
- **순서도 정보다.** 초록·파랑 순서가 바뀌면 의미가 달라진다.
- 속도·회전·높이·접촉은 **빠져 있다**. Action은 XYZ인데 관측에 Z가 없다는 점에 주의. EEF 높이는 `info["eef_position"][2]`로 확인한다.

### 3.3 Action — `Box(-1, 1, (3,), float32)`

```text
action        = [ax, ay, az]  ∈ [-1, 1]³
요청 이동량    = action × [0.008, 0.008, 0.005] m     (X·Y 최대 8 mm, Z 최대 5 mm)
EEF 목표 위치  = clip(현재 EEF XYZ + 요청 이동량, 하한, 상한)
```

| Action | 요청 |
|---|---|
| `[1, 0, 0]` | X +8 mm |
| `[0, -0.5, 0]` | Y -4 mm |
| `[0, 0, 1]` | Z +5 mm |
| `[0.5, -0.25, 1]` | X +4, Y -2, Z +5 mm |
| `[0, 0, 0]` | 이동 요청 0 (물리는 계속 진행되므로 움직이던 물체가 즉시 서지는 않음) |

- EEF **목표** 제한: X `[0.15, 0.40]`, Y `[-0.14, 0.14]`, Z `[0.008, 0.035]` m.
  - 예: 현재 Z = 0.033인데 `az=1` → 목표 Z는 0.038이 아니라 0.035.
  - **목표**를 제한하는 것이지 실제 위치를 덮어쓰는 것이 아니다.
- Action은 관절각도 힘도 아니다. 이 환경에서 **"EEF 이동 요청"으로 해석하기로 정한 값**이다.
- 요청한 만큼 정확히 움직이지 않는다. IK 오차·관절 제한·관절 변화량 제한·물리 응답·접촉 때문에 실측이 요청보다 작거나 다를 수 있다.
  - 예: 요청 `[+4.0, -2.0, +5.0] mm` → 실측 약 `[+3.45, -1.84, +4.26] mm`

### 3.4 Reward

```text
reward = -0.01 × (ax² + ay² + az²)          (action_cost_scale = 0.01)
```

| Action | Reward |
|---|---:|
| `[0, 0, 0]` | 0 |
| `[0, 0, 1]` | -0.01 |
| `[0.5, -0.25, 1]` | -0.013125 |
| `[1, -1, 1]` | -0.03 |

- **행동 크기에 대한 비용**일 뿐이다. 실제 이동량·큐브 이동·목표 거리는 들어가지 않는다.
- 이 보상만 최대화하면 "가만히 있기(`[0,0,0]`)"가 최적이 된다. 과제에 맞는 보상은 11·12주차에서 설계한다.
- 보상 정의가 학습 방향을 결정한다. 보상은 **환경 설계자가 정하는 것**이다.

### 3.5 종료 조건

- **terminated**: 초록 또는 파랑 큐브 중심 XY가 작업영역 X `[0.15, 0.40]`, Y `[-0.14, 0.14]` 밖. 경계값 자체는 **안쪽**으로 취급한다(`<`, `>` 비교).
- **truncated**: `elapsed_steps >= 150`이면서 terminated가 아닐 때.
- 높이·회전·두 큐브 간 접촉은 종료 조건에 포함하지 않는다.
- 성공 조건·성공률 없음. Return이 높아야 완료되는 것도 아니다.

> EEF **목표** XYZ 제한(Action 쪽)과 **큐브** XY 작업영역 검사(종료 쪽)는 다른 것이다.

### 3.6 `info` 항목

| 키 | 내용 |
|---|---|
| `eef_position` | EEF 실제 XYZ (base_link) |
| `cube_position` / `blue_cube_position` | 초록/파랑 큐브 XYZ |
| `elapsed_steps` | 이번 Episode에서 실행한 Step 수 |
| `out_of_bounds` | 영역 밖 큐브가 하나라도 있는지 |
| `outside_cubes` | 영역 밖 큐브 색 이름 목록 (예: `["blue"]`) |
| `reward_terms` | 보상 구성 항 (`{"action_cost": ...}`) |
| `ik_error` | 마지막 IK 해의 위치 오차(m) |
| `robot_cube_contacts` | 이번 Step 동안 로봇과 힘을 주고받은 큐브 목록 |
| `coordinate_frame` | `"base_link"` |

- `info`는 **점검·분석용**이다. 코드에서 따로 꺼내 쓰지 않으면 Agent 입력(Observation)에 들어가지 않는다.

### 3.7 설정 파일 `config/task.yaml`

```yaml
workspace:
  x: [0.15, 0.40]
  y: [-0.14, 0.14]
  eef_z: [0.008, 0.035]        # EEF 목표 Z 제한 (11·12주차와 동일)
initial:
  robot_qpos: [-0.00007, -1.2809, 1.6512, 0.4482, 0.0, 0.20]   # pan, lift, elbow, wrist_flex, wrist_roll, gripper
control:
  action_scale_m: [0.008, 0.008, 0.005]
  frame_skip: 25
  ik_iterations: 45
  ik_damping: 0.015
  max_joint_delta_rad: 0.08    # 한 Step당 IK 관절 변화량 상한
episode:
  max_steps: 150
reward:
  action_cost_scale: 0.01
```

---

## 4. 코드 구조와 호출 흐름

### 4.1 파일 구성

```text
~/week10/gymnasium_lab/            ← 프로젝트 루트 (python -m 실행 위치)
├── requirements.txt               numpy 1.26.4 / gymnasium 1.3.0 / mujoco 3.11.0 / PyYAML 6.0.3
├── config/task.yaml               작업영역·제어·Episode·보상 설정
├── models/                        scene.xml, physicai_arm.xml, robot.xml, assets/(STL·PNG)
└── week10_push_env/
    ├── gymnasium_lab_env.py       [TODO 1~5] Gymnasium 환경 (Space·obs·reset·step·종료)
    ├── mujoco_simulation.py       [제공] 모델 로딩·좌표변환·IK·물리·화면
    ├── inspect_environment.py     [실행] 장면·Space·한 Step 결과 출력
    ├── random_agent.py            [TODO 6] 무작위 Action으로 Episode 반복
    └── contact_demo.py            [선택] 두 큐브 접촉·이동 확인
```

- 층 분리: `gymnasium_lab_env.py`는 **Gymnasium 입출력과 회차 처리**만, `mujoco_simulation.py`는 **물리·좌표·IK·화면**을 맡는다. Agent는 MuJoCo 배열을 직접 건드리지 않는다.

### 4.2 한 Step의 전체 호출 흐름

```text
random_agent.py
  │ action = action_space.sample()
  │ environment.step(action)
  ↓
GymnasiumLabEnv.step()
  │ ① 입력 검사: float64 변환, shape (3,) 확인, NaN/inf 거부, [-1,1] clip
  │ ② simulation.move_eef(action)
  ↓
MujocoArmSimulation._move_eef_state()
  │ ③ target = clip(현재 EEF + action × scale, EEF 작업영역)
  │ ④ _solve_ik(target)  → 관절 4개 목표 (Damped Least Squares)
  │ ⑤ 관절 변화량을 최대 0.08 rad로 균일 축소 (방향 유지)
  │ ⑥ data.ctrl[0:4] = 관절 목표 (position actuator)
  │ ⑦ mj_step × 25 (매 substep마다 로봇-큐브 접촉 기록)
  │ ⑧ mj_forward (site/body 좌표 갱신)
  ↓
GymnasiumLabEnv.step()
  │ ⑨ elapsed_steps += 1
  │ ⑩ reward = -0.01 × ‖action‖²
  │ ⑪ outside = 영역 밖 큐브 목록 → terminated / truncated
  │ ⑫ render_mode == "human"이면 render()
  │ ⑬ return (_observation(), reward, terminated, truncated, _info(...))
  ↓
random_agent.py
     5개 값 저장 → episode_return += reward → while 조건 검사 → 반복 또는 종료
```

### 4.3 Episode 루프 흐름

```text
for episode in range(N):
    obs, info = reset(seed=s+episode)    ← 로봇·큐브 초기화, step 수 0
    action_space.seed(s+episode)         ← Action 순서 재현
    return = 0, terminated = truncated = False
    while not (terminated or truncated):
        action = sample()
        obs, r, terminated, truncated, info = step(action)
        return += r
    출력: steps, return, end(out_of_bounds / time_limit)
close()
```

---

## 5. 파일별 코드 설명

### 5.1 `gymnasium_lab_env.py` — Gymnasium 환경 클래스

```python
class GymnasiumLabEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
```

- `gym.Env`를 **상속**한다. 클래스 = 환경 동작 정의, 객체 = 실행 중인 한 환경의 상태.
- `render_fps = 20`은 Step 간격 0.05 s와 맞춘 값이다.
- 환경 ID 등록(`gym.make`)이나 `__init__.py` 없이 클래스를 직접 import해서 쓴다.

**`__init__(render_mode, model_path, config_path)`**

1. `render_mode`가 `None`/`"human"`/`"rgb_array"`인지 검사한다.
2. 기본 경로: `project_directory()`(= 이 파일의 상위 2단계 = 프로젝트 루트) 기준 `models/scene.xml`, `config/task.yaml`.
3. YAML을 읽어 `MujocoArmSimulation`을 생성하고 `model`, `data`를 노출한다.
4. **[TODO 1]** `observation_space`, `action_space`를 선언한다.
5. `_max_steps`, `_action_cost_scale`, `_elapsed_steps`를 준비한다.

**`reset(*, seed=None, options=None)`**

```python
super().reset(seed=seed)      # Gymnasium 규약: self.np_random 초기화
del options                   # 초기 무작위화는 이번 범위 밖
self.simulation.reset()       # 로봇·큐브를 XML 초기 상태로
self._elapsed_steps = 0
if self.render_mode == "human": self.render()
return (self._observation(), self._info())        # [TODO 3]
```

- `seed`는 **키워드 전용**(`*`) 인자다. Gymnasium 규약을 따른 것이다.

**`step(action)`**

- 입력 방어: `np.asarray(..., float64)` → shape `(3,)` 확인 → `isfinite` 확인 → `clip(-1, 1)`.
- 순서: 이동 → Step 수 증가 → 보상 → 이탈 검사 → **[TODO 5]** 종료 신호 → 렌더 → **[TODO 4]** 5개 반환.
- `info`의 `reward_terms`로 보상 구성을 함께 넘긴다.

**`_observation()`** — **[TODO 2]** EEF·초록·파랑 XYZ에서 `[:2]`로 XY만 골라 `concatenate` 후 `float32`로 변환한다.

**`_cubes_outside_workspace()`**

```python
for color, position in (("green", cube_position()), ("blue", blue_cube_position())):
    if np.any(position[:2] < workspace_low) or np.any(position[:2] > workspace_high):
        outside.append(color)
```

- 엄격한 부등호를 쓰므로 경계값은 안쪽이다.

**`_info(outside, reward_terms)`** — 3.6절의 Dictionary를 만든다. `outside`를 넘기지 않으면(`reset` 시) 새로 계산한다.

**`render()` / `close()`** — 시뮬레이션 객체에 위임한다.

### 5.2 `mujoco_simulation.py` — MuJoCo 담당 (제공 코드)

**상수**
- `ROBOT_JOINT_NAMES`: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
- `IK_JOINT_COUNT = 4`: **앞 4개 관절만 IK로 움직인다**. wrist_roll·gripper는 초기값을 유지한다.

**`_object_id(model, type, name)`** — `mj_name2id`로 이름 → ID를 얻고, 없으면 명확한 오류를 낸다.

**`__init__`**
- `MjModel.from_xml_path` → `MjData`. IK 계산 전용으로 `_ik_data`를 **별도로** 만들어 실제 상태를 오염시키지 않는다.
- site: `baseframe`(로봇 기준 좌표), `gripperframe`(EEF). body: `object_cube`, `target_cube`.
- 큐브 freejoint의 qpos/dof 주소, 큐브 geom ID → 색 이름 매핑.
- **로봇 body 집합**: `base` body부터 시작해 부모가 집합에 속한 body를 차례로 추가한다(트리 하위 전부). 접촉 판정에 쓴다.
- 관절 qpos/dof 주소, IK 관절 범위(`jnt_range`), actuator ID.
- 작업영역 XY 상·하한, EEF 목표 XYZ 상·하한(`eef_z` 추가).
- 제어 파라미터(`action_scale`은 양수 3개인지 검증).

**`reset()` → `_reset_state()`**

```python
mj_resetData(model, data)                  # XML 초기값으로 (큐브 위치 포함)
data.qpos[관절] = robot_qpos               # 로봇 초기 자세
data.ctrl[actuator] = robot_qpos           # 액추에이터 목표도 같게 → 튀지 않음
mj_forward(...)
for _ in range(10): mj_step(...)           # 큐브를 바닥에 안착(settle)
data.time = 0.0
mj_forward(...)                            # mj_step 후 좌표 갱신 → 관측이 최종 qpos와 일치
```

- 큐브 위치를 코드로 덮어쓰지 않는다. **XML 원본 배치를 그대로** 쓴다.
- Viewer가 열려 있으면 `viewer.lock()`으로 상태 변경을 보호한다(`_viewer_lock`).

**`move_eef(action)` → `_move_eef_state()`**

```python
target_xyz = clip(current_xyz + action * action_scale, eef_low, eef_high)
joint_target, last_ik_error = _solve_ik(target_xyz)
joint_delta = joint_target - current_joints
if max|joint_delta| > 0.08:  joint_delta *= 0.08 / max|joint_delta|   # 균일 축소 → 방향 유지
data.ctrl[앞 4개] = current_joints + joint_delta
for _ in range(25): mj_step(); _record_cube_contacts()
mj_forward()
```

- 관절 변화량 제한 때문에 큰 이동 요청은 한 Step에 다 가지 못할 수 있다(요청과 실측이 다른 이유 중 하나).
- `data.ctrl`은 position actuator의 **목표각**이다. 실제 각도는 물리 계산으로 따라간다.

**`_solve_ik(target)` — Damped Least Squares (위치만)**

```text
반복 최대 45회:
  e  = target_world - EEF_world            (‖e‖ < 0.1 mm면 중단)
  J  = mj_jacSite(...)의 위치 야코비안 중 IK 관절 4개 열  (3×4)
  Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e               (λ = 0.015)
  ‖Δq‖ > 0.20 rad 이면 0.20으로 축소
  q  = clip(q + Δq, 관절범위 ± 1e-3)
  mj_forward(_ik_data)
반환: (관절 4개 해, 최종 위치 오차)
```

- 목표는 base_link 좌표이므로 먼저 `base_to_world`로 World 좌표로 바꾼다.
- 감쇠항 λ²I는 특이자세 근처에서 해가 폭주하지 않게 한다.

**`_record_cube_contacts()`**
- 모든 접촉(`data.ncon`)을 돌며 한쪽이 큐브 geom, 다른 쪽이 로봇 body인지 본다.
- `mj_contactForce`로 법선 힘이 `> 1e-6`일 때만 "접촉"으로 기록한다. 스치기만 하는 0힘 접촉은 제외한다.
- 25 substep 전체에서 누적하므로 Step 중간에 잠깐 닿은 것도 잡힌다.

**좌표 변환**

```text
world_to_base(p) = Rᵀ (p − p_base)          R: baseframe site의 회전행렬(World←Base)
base_to_world(p) = p_base + R p
```

- `eef_position()`, `cube_position()`, `blue_cube_position()`은 모두 base_link 좌표로 반환한다.

**`render()`**
- `human`: 처음 호출 시 `viewer.launch_passive`로 띄우고 카메라(lookat (0.28,0,0.22), distance 1.15, azimuth -40, elevation -24)를 설정한다. 이후 `sync()`.
- `rgb_array`: `mujoco.Renderer(480, 640)`으로 `top_view` 카메라 이미지를 반환한다.

**`viewer_is_running()`** — Viewer가 없거나 열려 있으면 True. Agent가 창 닫힘을 감지하는 데 쓴다.

**`close()`** — Viewer를 닫고 WSLg 스레드 종료를 위해 1초 대기, 렌더러도 닫는다.

### 5.3 `inspect_environment.py` — 환경 점검

1. `reset(seed=10)`
2. **장면 출력**: `nq, nv, nu, dt`, 필수 객체(base, 두 큐브, floor, 두 site) ID를 확인한다. 없으면 오류.
3. **Space 출력**과 초기 Observation 6개를 라벨과 함께 출력한다.
4. `action = [0.5, -0.25, 1.0]`으로 한 Step 실행 후 출력:
   - 요청 이동량(`action × action_scale`) vs 실측 이동량(`info` EEF 차이)
   - 이후 EEF XYZ, reward terms, reward, 종료 신호, 새 Observation
5. `--render`면 창을 닫을 때까지 정지 화면을 유지한다(물리 진행 없음).

### 5.4 `random_agent.py` — 무작위 Agent

- 인자: `--episodes`(기본 3), `--seed`(기본 10), `--render`.
- Episode마다 `episode_seed = seed + index`로 `reset(seed=...)`과 `action_space.seed(...)`를 **둘 다** 호출한다.
- **[TODO 6]** 루프 안에서 `sample()` → `step()` → `episode_return += reward`.
- 첫 Episode의 처음 3 Step만 action·reward·종료 신호를 상세 출력한다.
- `--render`: Viewer 닫힘을 감지하면 중단하고, **중간에 닫은 회차는 완료 수에 넣지 않는다**. 실시간처럼 보이도록 `1/20 s`씩 sleep한다.
- 종료 원인: `info["out_of_bounds"]`가 True면 `out_of_bounds`, 아니면 `time_limit`.
- 마지막 Observation에 NaN/inf가 있으면 `RuntimeError`.
- `finally: environment.close()`로 항상 정리한다.

### 5.5 `contact_demo.py` — (선택) 두 큐브 접촉 확인

- **Waypoint 추종 비례 제어**:
  ```python
  action = clip((target - eef_xyz) / action_scale, -speed, speed)
  ```
  목표까지 남은 거리를 "몇 Step 분량의 최대 이동인지"로 나눠 Action으로 만든다. Z 목표는 초기 EEF 높이로 고정한다.
- 초록: (0.275, 0.030) → (0.355, 0.030), speed 0.8 → +X로 밀기.
- 파랑: (0.200,-0.060) → (0.335,-0.060) → (0.335,0.000) → (0.400,0.000), speed 1.0 → 옆으로 돌아 들어가서 밀기.
- Waypoint마다 최대 60 Step, 6 mm 이내 도달 또는 종료 신호 시 다음으로 넘어간다.
- 판정: `contact=True`이고 XY 변위 > 5 mm. 검증값은 초록 약 64.0 mm, 파랑 약 37.2 mm.
- 각 케이스 전에 `reset`만 하고 큐브 좌표를 직접 덮어쓰지 않는다. **로봇 접촉으로만** 움직인다.
- 파랑이 영역을 벗어나 terminated돼도 이 확인에서는 접촉·이동 여부만 본다.

---

## 6. TODO 1~6 정답과 해설

### TODO 1 — 두 Space 선언 (`__init__`)

```python
self.observation_space = spaces.Box(
    low=-np.inf, high=np.inf, shape=(6,),
    dtype=np.float32,
)
self.action_space = spaces.Box(
    low=-1.0, high=1.0, shape=(3,),
    dtype=np.float32,
)
```

- Observation 6개(XY × 3), 범위는 자르지 않도록 무한대로 둔다.
- Action 3개(XYZ), 정규화 비율이므로 [-1, 1].
- `shape`은 반드시 Tuple `(6,)`로 쓴다. `6`이나 `(6)`이 아니다.

### TODO 2 — Observation 구성 (`_observation`)

```python
return np.concatenate((
    self.simulation.eef_position()[:2],
    self.simulation.cube_position()[:2],
    self.simulation.blue_cube_position()[:2],
)).astype(np.float32)
```

- 순서 **EEF → 초록 → 파랑**. `[:2]`로 Z를 뺀다.
- `concatenate`의 인자는 **Tuple 하나**다(괄호 두 겹). 각 줄 끝 쉼표를 빠뜨리지 않는다.
- `float32` 변환은 `observation_space.contains()`를 통과하는 데 필요하다.

### TODO 3 — `reset()` 반환

```python
return (
    self._observation(),
    self._info(),
)
```

- Gymnasium 규약: `(observation, info)` **2개**. 순서를 바꾸거나 하나만 반환하면 언패킹 오류가 난다.

### TODO 4 — `step()` 반환

```python
return (
    self._observation(),
    reward,
    terminated,
    truncated,
    self._info(outside, {"action_cost": reward}),  # info는 제공
)
```

- 순서 `obs, reward, terminated, truncated, info`. `terminated`와 `truncated`를 바꾸면 `check_env`는 통과해도 의미가 틀어진다.
- 반드시 **행동 이후**의 관측을 새로 계산해 반환한다.

### TODO 5 — 종료 조건 (`step`)

```python
terminated = bool(
    len(outside) > 0
)
truncated = bool(
    (self._elapsed_steps >= self._max_steps) and not terminated
)
```

- `outside`는 영역 밖 큐브 색 목록이다. 비어 있지 않으면 종료. `bool(outside)`로 써도 같다.
- `and not terminated`: 마지막 Step에서 동시에 일어나면 **이탈 종료를 우선**한다.
- `bool(...)`로 감싸서 numpy bool이 아닌 Python bool을 반환한다.

### TODO 6 — Random Agent 한 Step (`random_agent.py`)

```python
action = (
    environment.action_space.sample()
)
observation, reward, terminated, truncated, info = (
    environment.step(action)
)
episode_return += (
    float(reward)
)
```

- `sample()`은 Space 규격에 맞는 값(float32, shape (3,), [-1,1])을 보장한다.
- 반환 5개를 순서대로 받고, while 조건이 새 `terminated/truncated`로 다시 검사된다.
- Return = Reward의 **합**이다. 한 Step의 Reward와 구분한다.

---

## 7. 실습 진행 순서

> 실행 환경: Windows PC 안의 **WSL2 Ubuntu 22.04**. Viewer는 WSLg로 표시. ROS 2·Jetson은 사용하지 않는다.

### 7.1 준비

```bash
cd ~
rm -rf ~/week10                 # 이전 분반 자료 삭제 (보관할 것은 먼저 복사)
mkdir -p ~/week10 && cd ~/week10
python3 -m venv .venv           # --system-site-packages 붙이지 않음 (ROS 불필요)
source ~/week10/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
command -v python3              # ~/week10/.venv/ 아래인지 확인
```

- `ensurepip is not available`이면 `sudo apt install python3.10-venv` 후 venv 생성부터 다시 한다.
- 4.1절 구조대로 파일을 배치한다(`task.yaml`은 `config/`에, `models/`는 assets 포함 전체).

```bash
cd ~/week10/gymnasium_lab
python3 -m pip install -r requirements.txt
# 버전 확인 스크립트 → "PC packages OK"
```

- 새 터미널을 열면 루트 이동·venv 활성화·환경 변수를 다시 적용한다.

### 7.2 TODO 작성 → 검사

```bash
# 1) 빈칸·문법 검사 → "Source check OK"
#    (week10_push_env/*.py가 5개, '### TODO ###' 없음, compile 성공)

# 2) 한 Step 실행
python3 -m week10_push_env.inspect_environment

# 3) Gymnasium API·보상 검사 → "Gymnasium API OK, reward = -0.013125"
#    check_env, obs shape/dtype/contains, action shape/dtype, reward 값, Z 증가, 종료 False
```

### 7.3 Episode 반복

```bash
python3 -m week10_push_env.random_agent --episodes 1 --seed 10
python3 -m week10_push_env.random_agent --episodes 3 --seed 10
python3 -m week10_push_env.random_agent --episodes 1 --seed 11     # Action 순서만 바뀜
```

### 7.4 Viewer

```bash
python3 -m week10_push_env.inspect_environment --render    # 한 Step 후 정지 화면, 드래그 회전·휠 확대
python3 -m week10_push_env.random_agent --episodes 1 --seed 10 --render   # 150 Step 후 자동 종료
```

### 7.5 (선택) 접촉 확인

```bash
python3 -m week10_push_env.contact_demo
python3 -m week10_push_env.contact_demo --render
```

### 7.6 정리

```bash
deactivate
cd ~
rm -rf ~/week10        # 코드·모델·venv 모두 삭제 (다음 분반용)
```

- 조교 확인과 제출을 마친 뒤, 보관할 소스·기록은 **개인 저장소에 먼저 복사**한다.

---

## 8. 검증 기준값

### 8.1 `inspect_environment`

| 항목 | 기준값 |
|---|---|
| 모델 | `nq=20`, `nv=18`, `nu=6`, `dt=0.0020 s` |
| Space | Observation `(6,)`, Action `(3,)`, `float32` |
| 첫 Observation | ≈ `[0.1999, 0, 0.3100, 0.0300, 0.3700, 0]` |
| Action | `[0.5, -0.25, 1.0]` |
| 요청 이동 | `[+0.004, -0.002, +0.005] m` |
| 실측 이동 | ≈ `[+0.003447, -0.001836, +0.004262] m` |
| 이후 EEF XYZ | ≈ `[0.203383, -0.001836, 0.014109] m` (Z: 0.009847 → 0.014109) |
| Reward | `{'action_cost': -0.013125}`, 합계 `-0.013125` |
| 종료 신호 | `False/False` |

> 소수점 끝자리보다 **요청과 실측이 다르다는 점**을 확인하는 것이 핵심이다.

### 8.2 `random_agent --episodes 3 --seed 10`

```text
step=  1 action=[ 0.912 -0.585  0.657] reward=-0.016051 terminated/truncated=False/False
step=  2 action=[-0.701  0.026 -0.728] reward=-0.010229 terminated/truncated=False/False
step=  3 action=[ 0.378  0.683 -0.149] reward=-0.006323 terminated/truncated=False/False
episode=1, steps=150, return=-1.446450, end=time_limit
episode=2, steps=150, return=-1.492363, end=time_limit
episode=3, steps=150, return=-1.629437, end=time_limit
completed episodes: 3/3
```

- 무작위 Action은 평균 `E[a²] = 1/3`(성분당) → Step당 기대 보상 ≈ `-0.01`, 150 Step이면 Return ≈ `-1.5`. 실제 값이 이 근처인 이유다.
- Return이 개선되는 것은 합격 조건이 **아니다**(학습하지 않으므로).

### 8.3 `contact_demo`

| 큐브 | contact | XY 변위 (검증값) |
|---|---|---|
| 초록 | True | ≈ 64.0 mm |
| 파랑 | True | ≈ 37.2 mm |

---

## 9. 자주 틀리는 부분과 문제 해결

| 증상 | 원인 / 확인 |
|---|---|
| `TypeError`·`ValueError`·"not enough values to unpack" | TODO 빈칸 남음, 반환값 개수·순서, 쉼표 누락, `shape`를 Tuple로 안 씀 |
| `observation_space.contains(obs)`가 False | `_observation()`에서 `float32` 변환 누락 또는 shape가 (6,)이 아님 |
| Episode가 1 Step 만에 끝남 | TODO 5 조건이 반대(`len(outside) == 0`) 또는 `truncated` 비교 오류 |
| Episode가 끝나지 않음 | `truncated` 조건 누락, `step`의 반환값을 루프 변수로 받지 않음 |
| Seed가 같은데 결과가 다름 | `action_space.seed()` 누락, 라이브러리 버전 불일치 |
| `check_env`가 infinity 경고 | 의도된 선언이라 실패 아님 (`spec` 없음 안내도 무시 가능) |
| venv 생성 실패 | `python3.10-venv` 설치 (권한 없으면 조교) |
| Viewer가 안 뜨거나 프롬프트로 안 돌아옴 | WSLg 확인, 창을 닫거나 Ctrl+C. `finally`의 `close()`가 정리 |
| 모델 로딩 오류 | `models/assets/` STL·PNG 누락, `config/task.yaml` 위치 확인 |

**개념상 혼동 포인트**
- Observation ≠ 시뮬레이터 전체 상태 (속도·회전·높이·접촉 없음).
- Action ≠ 관절각/힘, Action 요청량 ≠ 실제 이동량.
- EEF **목표** 제한(XYZ, Action 쪽) ≠ 큐브 **작업영역** 검사(XY, 종료 쪽).
- Reward(한 Step) ≠ Return(Episode 합).
- `render()` ≠ 물리 진행.
- 환경 Step ≠ MuJoCo `mj_step` ≠ 화면 갱신.
- `info`는 Agent 입력이 아니다.

---

## 10. 복습 질문

1. `reset()`과 `step()`의 반환값 개수와 순서는? → 2개 `(obs, info)`, 5개 `(obs, reward, terminated, truncated, info)`
2. 왜 `reset()`이 첫 Observation을 반환해야 하나? → 첫 행동을 고르기 전에 현재 상황을 알아야 하므로.
3. `terminated`와 `truncated`의 차이와 이번 환경의 조건은? → 과제 종료(큐브 이탈) vs 시간 제한(150 Step). 동시에 일어나면 이탈 우선.
4. Observation Space를 `±inf`로 둔 이유는? → 영역을 벗어난 위치도 잘라내지 않고 반환하려고. 영역은 종료 조건으로 따로 검사.
5. `shape=(3,)`의 의미는? → 원소 3개짜리 1차원 배열.
6. Action `[0.5, -0.25, 1.0]`의 요청 이동량과 Reward는? → `[+4, -2, +5] mm`, `-0.013125`.
7. 요청과 실측 이동이 다른 이유는? → IK 오차, 관절 제한, 관절 변화량 0.08 rad 제한, 물리 응답·접촉.
8. 이 보상으로 학습하면 어떤 정책이 나오나? → `[0,0,0]`(가만히 있기). 과제 보상이 따로 필요하다.
9. 같은 Action 순서를 재현하려면? → `reset(seed=n)`과 `action_space.seed(n)` 둘 다.
10. Seed를 바꾸면 큐브 초기 위치가 바뀌나? → 아니다. 초기 배치는 XML 고정. Action 순서만 바뀐다.
11. 한 Step은 물리 시간으로 몇 초인가? → 0.002 × 25 = 0.05 s. 150 Step = 7.5 s.
12. `gymnasium_lab_env.py`와 `mujoco_simulation.py`를 나눈 이유는? → 학습 인터페이스(입출력·회차)와 물리 세부(IK·좌표·접촉·화면)를 분리해 Agent가 MuJoCo 내부를 몰라도 되게 하려고.
13. IK는 몇 개 관절로 푸나? 방법은? → 4개(pan·lift·elbow·wrist_flex), 위치 전용 Damped Least Squares.
14. `try/finally`로 `close()`를 부르는 이유는? → 오류나 중단이 생겨도 Viewer·렌더러 자원을 정리하려고.

---

## 11. 9 → 10 → 11 → 12주차 연결

| | 10주차 | 11주차 | 12주차 |
|---|---|---|---|
| Agent | Random (`sample`) | **SAC** 학습 정책 | SAC |
| Action | XYZ 3차원, 같은 배율 | 동일 | 동일 |
| Observation | XY **6차원** | 관절·영상 좌표 등 **38차원** | 38차원 |
| 초기 배치 | XML 고정 | 고정 | **무작위화** |
| Reward | 행동 비용만 | 두 큐브 접촉 성공 기반 | 과제 맞춤 보상 설계 |
| 성공 조건 | 없음 | 두 큐브 접촉 | 과제 수행 |

- 이번 주의 **환경 인터페이스(Space·reset·step·종료 신호·info)** 를 그대로 유지한 채 Agent와 Observation·Reward만 바꿔 가는 구조다.
- 그래서 10주차의 완료 기준은 "학습 성과"가 아니라 **"환경을 규약에 맞게 구성하고 올바르게 사용하는 것"** 이다.

---

### 참고 링크

- [Gymnasium 기본 사용 안내](https://gymnasium.farama.org/introduction/basic_usage/)
- [Gymnasium Env API](https://gymnasium.farama.org/api/env/)
- [Gymnasium Box API](https://gymnasium.farama.org/api/spaces/fundamental/)

> 작성: 10주차 제공 자료(`lecture.md`, `practice.md`, `source_code/`)를 바탕으로 Claude가 정리했다. TODO 정답과 8절의 출력값은 클라우드 환경에서 실제로 실행해 확인한 결과다(`--render` Viewer 확인 제외).
