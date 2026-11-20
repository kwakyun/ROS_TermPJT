# [11주차] SAC로 로봇 정책 학습하기 — 수업 내용

## 0. 이번 시간에 알아볼 것

10주차에는 Gymnasium 환경을 만들고 무작위 Action으로 Episode를 실행했다.
환경은 Observation·Reward·종료 여부를 돌려주었고, Agent는 매번 새로운 행동을 뽑았다.
이번에는 행동의 결과를 경험으로 저장하고, 그 경험을 이용하여 행동 선택 방법을 바꾼다.

로봇이 수행할 과제는 초록 큐브와 파란 큐브를 접촉시키는 것이다.
학생은 보상을 계산하고 SAC 모델을 생성하여 학습한다. 학습한 정책을 파일로 저장한 뒤 다시 불러와 평가한다.

- Policy는 Observation으로 어떤 값을 출력하는가?
- Actor·Critic·Replay Buffer는 학습에서 어떤 역할을 하는가?
- 보상식을 바꾸면 Agent가 경험을 평가하는 기준은 어떻게 달라지는가?
- `learn()`, `save()`, `load()`, `predict()`는 언제 사용하는가?
- 학습 중 Return과 평가 성공률은 각각 무엇을 알려 주는가?

## 1. 환경 실행에서 정책 학습으로

### 1.1 Gymnasium의 입출력을 다시 연결한다

환경을 초기화하면 `reset()`이 초기 Observation과 `info`를 반환한다.
Agent가 Action을 선택해 `step()`에 전달하면 새로운 관측, 보상, 두 종료 신호, 추가 정보를 받는다.

```python
observation, info = environment.reset(seed=12)
action = environment.action_space.sample()
observation, reward, terminated, truncated, info = environment.step(action)
```

`terminated`는 과제의 성공·실패, `truncated`는 시간 제한을 나타낸다.
둘 중 하나가 참이면 해당 Episode를 끝내고 다음 Episode를 초기화한다.
`render()`는 화면을 만들고, `close()`는 화면과 관련 자원을 정리한다.

### 1.2 Policy가 행동 선택 방법을 표현한다

정책(Policy)은 현재 Observation을 받아 Action을 선택하는 함수이다.
이번 정책의 입력은 15개의 실수이고 출력은 XYZ 이동을 요청하는 3개의 실수이다.
신경망의 가중치를 바꾸면 같은 관측에서도 행동 선택이 달라진다.

```text
Observation 15개 → 정책 신경망 → Action [ax, ay, az]
       ↑                              ↓
       └──── 새로운 관측·보상 ← 환경 ──┘
```

강화학습은 환경에서 얻은 경험을 이용해 앞으로 받을 할인 누적 보상이 커지도록 정책을 개선한다.
신경망의 각 출력에 대해 사람이 정답 Action을 제공하는 대신, 환경이 행동 결과의 보상을 계산한다.

## 2. SAC는 경험을 어떻게 사용하는가

### 2.1 연속 행동과 확률적 정책

SAC(Soft Actor-Critic)는 연속적인 Action을 다루는 강화학습 알고리즘이다.
로봇의 XYZ 이동 요청처럼 크기와 방향을 실수로 표현하는 문제에 사용할 수 있다.
학습 중에는 확률적인 정책으로 여러 행동을 시도한다.
SAC는 보상과 함께 정책의 엔트로피도 고려하여 행동 탐색을 유지한다.

엔트로피는 정책이 행동을 얼마나 다양하게 선택하는지를 나타내는 양이다.
엔트로피 계수는 보상과 행동 다양성 사이의 비중을 조절한다.
이번 설정의 `ent_coef: auto_0.1`은 초기 계수 0.1에서 자동 조정을 시작한다.

### 2.2 Actor가 행동을 선택한다

Actor는 관측으로 행동 분포를 만들고, 그 분포에서 Action을 선택한다.
학습 초기에는 경험이 적기 때문에 유리한 방향을 알기 어렵다.
여러 행동의 결과를 경험하면서 Actor의 가중치를 갱신한다.

평가에서는 `predict(observation, deterministic=True)`를 사용한다.
이 옵션은 평가 시 행동 선택을 일정하게 만들어 같은 조건의 결과를 비교하기 쉽게 한다.

### 2.3 Critic이 행동의 가치를 추정한다

Critic은 관측과 행동을 받아 해당 행동 이후에 얻을 할인 누적 보상을 추정한다.
이 값을 Q값이라고 한다. SAC는 두 Q함수를 사용하고 목표값 계산에 작은 쪽 추정치를 활용한다.
Actor는 Critic이 추정하는 가치와 엔트로피를 고려하여 갱신된다.

Critic의 손실은 Q값 추정과 학습 목표 사이의 오차를 나타낸다.
손실의 크기와 과제 성공률은 서로 다른 측정값이므로 평가 결과도 함께 확인해야 한다.

### 2.4 Replay Buffer에 경험을 저장한다

한 번의 상호작용에서 다음과 같은 경험을 얻는다.

```text
(현재 Observation, Action, Reward, 다음 Observation, 종료 정보)
```

Replay Buffer는 이 경험들을 저장한다.
학습은 Buffer에서 Mini-batch를 뽑아 Actor·Critic을 갱신하는 방식으로 진행된다.
과거 정책이 모은 경험도 다시 사용할 수 있는 학습 방식을 Off-policy라고 한다.

이번에는 `learning_starts=5000`이다. 처음 5,000 Step 동안 무작위 행동으로 경험을 수집하고,
그 이후부터 경험을 사용한 신경망 갱신을 시작한다.
짧은 확인 학습도 5,200 Step으로 실행하여 가중치 갱신까지 확인한다.

### 2.5 주요 설정을 읽는다

| 설정 | 기본값 | 의미 |
|---|---|---|
| `total_timesteps` | 60000 | 전체 환경 상호작용 예산 |
| `learning_rate` | 0.0003 | 가중치 갱신 크기 |
| `buffer_size` | 60000 | 저장할 경험 수 |
| `learning_starts` | 5000 | 갱신 전 초기 경험 수집 Step |
| `batch_size` | 256 | 한 번의 갱신에 사용할 경험 수 |
| `gamma` | 0.99 | 미래 보상의 할인율 |
| `tau` | 0.005 | 목표 네트워크의 완만한 갱신 비율 |
| `network_architecture` | `[128, 128]` | 은닉층 크기 |
| `train_freq`, `gradient_steps` | 1, 1 | 경험 수집과 갱신 빈도 |

이번 구현은 Stable-Baselines3의 `SAC`를 사용한다.
`MlpPolicy`는 15차원 위치·관절각 벡터를 입력받는 다층 퍼셉트론 정책을 선택한다.
학습 라이브러리가 경험 수집·Buffer·역전파를 담당한다. [SAC 공식 문서](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)

### 2.6 Stable-Baselines3는 무엇을 해 주는가

Stable-Baselines3(SB3)는 SAC·PPO·DQN 등 강화학습 알고리즘을 구현한 Python 라이브러리이다.
이번 수업에서는 `from stable_baselines3 import SAC`로 SAC 구현을 가져온다.
학생이 만든 Gymnasium 환경을 연결하면 SB3가 경험 수집, Replay Buffer 관리, Actor·Critic의 가중치 갱신을 수행한다.

| 구성 요소 | 이번 실습에서 담당하는 일 |
|---|---|
| MuJoCo | 로봇·큐브의 운동과 접촉 계산 |
| Gymnasium 환경 | Observation·Action Space, reset·step, 보상·종료 조건 제공 |
| Stable-Baselines3 | 환경과 상호작용하며 SAC 정책 학습 |
| 학생 실행 코드 | 모델 생성, 학습 요청, 저장, 불러오기, 평가 연결 |

```text
train.py가 SAC 모델을 만든다
    ↓
model.learn() 안에서 환경 reset·행동 선택·step을 반복한다
    ↓
경험을 저장하고 Mini-batch로 가중치를 갱신한다
    ↓
model.save()로 저장한다
    ↓
SAC.load()로 불러와 model.predict()로 평가한다
```

### 2.7 기본 API 다섯 가지

| API | 입력과 역할 | 반환값 |
|---|---|---|
| `SAC("MlpPolicy", env, ...)` | 정책 종류·환경·학습 설정을 받아 모델 생성 | SAC 모델 객체 |
| `model.learn(total_timesteps=N)` | 환경과 최대 N Step 상호작용하며 학습 | 학습된 모델 객체 |
| `model.save("policy")` | 가중치와 모델 설정 저장 | `policy.zip` 파일 생성 |
| `SAC.load("policy.zip", env=env)` | 저장한 모델을 복원하고 환경 연결 | 복원한 SAC 모델 객체 |
| `model.predict(obs, deterministic=True)` | 현재 관측으로 평가 행동 계산 | `(action, state)` |

`MlpPolicy`는 실수 벡터를 입력받는 신경망 정책이다.
생성자의 `env`에는 Gymnasium 환경 객체를 전달한다. SB3는 이 객체의 두 Space에서 입력·출력 크기를 알아낸다.
`learning_rate`·`batch_size` 등의 인자는 앞의 설정 표와 연결된다.

`total_timesteps=60000`은 환경의 Step 호출 횟수이며 Episode 수와 구분한다.
학습 중 Episode가 끝나면 SB3가 환경을 초기화하고 경험 수집을 계속한다.
초기 무작위 수집 구간도 전체 Step 수에 포함된다.
이번 실행의 콜백은 검증 성공률 조건을 충족하면 학습을 일찍 종료할 수 있다.

`save()`에 `.zip`을 생략하면 SB3가 확장자를 붙인다.
`load()`는 SAC 클래스에서 호출하고 반환한 모델을 변수에 받아 사용한다.
`predict()`의 첫 반환값인 `action`을 환경에 전달한다.
두 번째 `state`는 순환 정책의 내부 상태에 사용하는 값이며 이번 MLP 정책에서는 `None`이다.

### 2.8 `SAC()`의 옵션을 실습 코드와 연결한다

`train.py`의 TODO 2에서는 `SAC()`가 반환하는 모델을 `model` 변수에 저장한다.
함수에 전달하는 `이름=값` 형태가 옵션이다.
`training['batch_size']`처럼 설정 사전에서 읽은 값도 옵션으로 전달할 수 있다.
코드의 `int()`·`float()`는 YAML에서 읽은 값을 API가 요구하는 자료형으로 맞춘다.

| 인자 | 이번 코드의 값·변수 | 상세 의미 |
|---|---|---|
| `policy` | `"MlpPolicy"` | 15개 실수 관측을 입력받는 MLP 정책을 선택한다. 모델은 환경의 Space에서 입력 15개·출력 3개의 크기를 읽는다. |
| `env` | `environment` | `Monitor`로 감싼 로봇 환경이다. 학습 중 reset·step을 호출할 대상이다. |
| `seed` | `seed`, 기본 12 | 모델 초기화·무작위 행동 등 난수 생성에 사용할 기준이다. 실행 조건에 따른 수치 차이는 생길 수 있다. |
| `device` | `device`, 실습 명령에서 `"cpu"` | 신경망 계산 장치를 선택한다. `"auto"`는 사용 가능한 장치를 자동으로 선택한다. |
| `learning_rate` | `0.0003` | 최적화기가 가중치를 갱신하는 학습률이다. 한 번의 갱신으로 가중치를 얼마나 바꿀지에 영향을 준다. |
| `buffer_size` | `60000` | Replay Buffer가 보관하는 최대 경험 수이다. 가득 차면 오래된 경험부터 덮어쓴다. |
| `learning_starts` | `5000` | 가중치 갱신 전에 먼저 모을 환경 Step 수이다. 이번 기본 설정에서는 이 구간에 무작위 행동을 수집한다. |
| `batch_size` | `256` | 한 번의 학습 갱신에 Buffer에서 뽑는 경험 수이다. |
| `gamma` | `0.99` | 미래 보상 할인율이다. k Step 뒤 보상에는 `gamma**k`가 곱해지므로 멀리 있는 보상일수록 작게 반영된다. |
| `tau` | `0.005` | 목표 Critic을 천천히 갱신하는 비율이다. 목표 가중치는 `0.995 × 기존 목표 + 0.005 × 현재 Critic`으로 섞는다. |
| `ent_coef` | `"auto_0.1"` | 행동 다양성의 비중을 정하는 엔트로피 계수를 0.1에서 시작해 자동 조정한다. |
| `train_freq` | `1` | 환경 Step 1회마다 학습 갱신 구간으로 넘어간다. 실제 갱신은 초기 경험 수집 후 시작한다. |
| `gradient_steps` | `1` | 학습 갱신 구간마다 Mini-batch 갱신을 1회 수행한다. |
| `policy_kwargs` | `{"net_arch": [128, 128]}` | 정책 네트워크에 전달할 추가 옵션 사전이다. `net_arch`는 Actor와 Critic에 각각 128개 뉴런의 은닉층 두 개를 구성한다. |
| `tensorboard_log` | `str(tensorboard)` | TensorBoard 학습 지표를 저장할 폴더 경로이다. `Path` 객체를 `str()`로 문자열 경로로 바꿔 전달한다. |
| `verbose` | `1` | 장치·학습 진행 등의 정보를 출력한다. `0`은 SB3의 정보 출력을 줄이고 `2`는 디버그 정보까지 표시한다. |

`buffer_size=60000`은 경험 저장 용량이고 `total_timesteps=60000`은 이번 실행의 상호작용 예산이다.
두 값은 서로 다른 역할을 하므로 같은 숫자여도 별도로 설정한다.
`train_freq=1`, `gradient_steps=1`, `learning_starts=5000`인 이번 환경에서
5,200 Step 학습을 마치면 기본적으로 200회의 Mini-batch 갱신이 수행된다.

### 2.9 `learn()`으로 수집·학습을 실행한다

TODO 3은 `model.learn()`을 호출하여 이미 생성한 모델을 학습시키는 부분이다.

| 인자 | 이번 코드에서 전달할 값 | 의미 |
|---|---|---|
| `total_timesteps` | `total_steps` | 초기 무작위 수집까지 포함한 전체 환경 상호작용 수이다. `--steps`가 있으면 그 값을, 생략하면 설정의 60000을 사용한다. |
| `callback` | `callbacks` | 진행 출력·검증·중간 저장을 학습 중 실행하도록 연결한다. 콜백이 종료를 요청하면 예산 전에 학습을 마칠 수 있다. |
| `log_interval` | `int(training['log_interval'])`, 기본 20 | SAC에서 Episode 20개를 마칠 때마다 학습 로그를 출력한다. |
| `tb_log_name` | `"sac_push"` | TensorBoard 실행 폴더 이름의 앞부분이다. 생성자에서 지정한 로그 폴더 아래에 저장된다. |

`learn()`은 학습한 모델 자신을 반환하며 `model`의 가중치도 갱신한다.
이번 코드에서는 이미 `model` 변수에 객체가 있으므로 반환값을 다시 대입하지 않고 호출해도 된다.
`Progress 5000/...` 출력은 제공한 콜백이 출력한다. Episode 기준인 `log_interval`과 주기가 다르다.

### 2.10 `save()`와 `load()`로 정책을 보관하고 복원한다

TODO 4에서는 `model.save()`로 `run_directory` 안에 `policy_final.zip`을 만든다.
저장할 경로를 `str(run_directory / 'policy_final')`로 전달할 수 있다.
`save()`의 반환값은 `None`이다. 저장 후 `model`은 계속 학습 모델 객체를 가리킨다.
ZIP에는 가중치와 모델 설정 등이 저장된다. 이 실습은 환경 XML·설정을 프로젝트에 함께 배치한다.
Replay Buffer를 보관하려면 별도의 `save_replay_buffer()`가 필요하다.

TODO 5에서는 다음 인자를 사용해 `SAC.load()`의 반환값을 `model`에 저장한다.

| 인자 | 이번 코드의 값 | 의미 |
|---|---|---|
| `path` | `str(policy_path)` | 불러올 정책 ZIP 경로이다. 앞의 제공 코드가 CLI 옵션으로 선택한 경로를 계산해 둔다. |
| `env` | `environment` | 복원한 모델에 연결할 평가 환경이다. 모델과 관측·행동 Space가 맞는지도 검사한다. |
| `device` | `"cpu"` | 복원한 신경망을 CPU에서 실행한다. 저장 당시 계산 장치와 달라도 지정할 수 있다. |

`model = SAC.load(...)`로 새 모델 객체를 받는다.
그 다음 제공 코드가 `environment_version`도 검사하여 관측 의미가 달라진 정책 사용을 막는다.

### 2.11 `predict()`와 `step()`으로 평가 한 회차를 실행한다

TODO 6에서는 `model.predict()`에 현재 `observation`과 `deterministic=True`를 전달한다.

| 인자·반환값 | 이번 실습에서의 의미 |
|---|---|
| 입력 `observation` | reset 또는 직전 step에서 받은 `(15,)` 관측 배열 |
| 입력 `deterministic=True` | 정책에서 평가용 결정적 행동을 선택한다. 기본 `False`에서는 확률적으로 행동을 뽑는다. |
| 반환 `action` | 환경에 전달할 `(3,)` XYZ 행동 배열 |
| 반환 `state` | 이번 MLP 정책에서는 `None`. 사용하지 않는 반환값은 `_`에 받을 수 있다. |

예측 다음에는 제공 코드의 `environment.step(action)`이 실행된다.
이때 Observation·Reward·종료 신호·info를 받는다.
두 종료 신호 중 하나가 참이면 회차를 마치고 성공·길이·최종 거리를 기록한다.
평가 코드는 학습된 모델로 행동을 예측하며 가중치 갱신은 `learn()`에서 수행한다.

### 2.12 제공된 `Monitor`와 Callback을 읽는다

`Monitor`는 환경을 감싸 Episode 결과를 기록하는 Wrapper이다.
`Monitor(raw_environment, filename=..., info_keywords=...)`로 생성한 객체를 SAC의 환경으로 전달한다.
`filename`은 CSV 저장 경로이고 `info_keywords`는 Episode 종료 시 `info`에서 추가로 기록할 항목이다.
이번에는 `is_success`, `scenario`, `failure_reason`을 저장하며 기본 열 `r`·`l`·`t`는
각각 Episode Return·길이·실행 경과 시간을 나타낸다. [Monitor 공식 문서](https://stable-baselines3.readthedocs.io/en/v2.9.2/common/monitor.html)

Callback은 학습 중 SB3가 호출하는 코드이다.
제공한 `ProgressCallback`은 진행 상황을 출력하고 `ValidationCallback`은 성공률을 확인한다.
`BaseCallback`을 상속한 클래스의 `_on_step()`이 `True`를 반환하면 계속 학습하고,
`False`를 반환하면 조기 종료한다.
`CallbackList([...])`는 여러 콜백을 하나로 묶어 `learn(callback=...)`에 연결한다.

`CheckpointCallback`의 `save_freq=10000`은 단일 환경에서 10,000 Step마다 저장하라는 뜻이다.
`save_path`는 `checkpoints/` 폴더, `name_prefix="sac_push"`는 중간 정책 파일명의 앞부분이다.
이 수업의 환경은 하나이므로 콜백 호출 횟수와 환경 Step 수가 같다. [Callback 공식 문서](https://stable-baselines3.readthedocs.io/en/v2.9.2/guide/callbacks.html)

### 2.13 실행 명령의 옵션을 API 인자와 연결한다

| CLI 옵션 | 연결되는 곳 | 예 |
|---|---|---|
| train의 `--steps` | `learn(total_timesteps=...)` | `--steps 5200`으로 확인 학습 |
| train의 `--seed` | `SAC(seed=...)` | 생략 시 설정의 12 |
| train의 `--device` | `SAC(device=...)` | 실습에서는 `--device cpu` |
| train의 `--threads` | `torch.set_num_threads()` | `--threads 1`로 PyTorch CPU 연산 스레드 수 지정 |
| train의 `--run-name` | 저장 폴더 `outputs/<이름>/` | `--run-name baseline` |
| evaluate의 `--run` | 정책이 있는 실행 폴더 선택 | `--run baseline` |
| evaluate의 `--model` | 해당 폴더의 정책 종류 선택 | 기본 `final`, 검증 정책이 있으면 `best` |
| evaluate의 `--policy` | 불러올 ZIP의 직접 경로 지정 | 이 값이 있으면 정책 경로 선택에 우선 사용 |
| evaluate의 `--episodes-per-scenario` | 제공 평가 반복문의 회차 수 | 기본 40, 확인 실습은 2 |
| evaluate의 `--seed` | 평가 환경 `reset(seed=...)`의 시작 Seed | 기본 12000, 회차마다 1 증가 |

위 옵션 중 실행 이름·평가 횟수는 우리가 작성한 프로그램에서 처리한다.
SAC 생성·학습·복원의 인자에 연결되는 값은 앞의 표를 따라 확인한다.
생성자와 메서드의 전체 API는 수업에서 사용하는 버전의 문서에서 확인할 수 있다. [SAC 2.9.2 API](https://stable-baselines3.readthedocs.io/en/v2.9.2/modules/sac.html)

### 2.14 작은 환경에서 전체 호출 순서를 읽는다

다음은 Gymnasium의 `Pendulum-v1`을 사용하는 API 예제이다.
로봇 실습에서는 환경 객체와 학습 설정을 이번 제공 코드의 값으로 연결한다.

```python
import gymnasium as gym
from stable_baselines3 import SAC

environment = gym.make("Pendulum-v1")
try:
    model = SAC("MlpPolicy", environment, learning_rate=0.0003,
                learning_starts=100, seed=12, device="cpu")
    model.learn(total_timesteps=200)
    model.save("pendulum_policy")

    loaded_model = SAC.load("pendulum_policy.zip", env=environment, device="cpu")
    observation, info = environment.reset(seed=12)
    terminated = truncated = False
    while not (terminated or truncated):
        action, state = loaded_model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(action)
finally:
    environment.close()
```

학습 구간의 `learn()` 내부에서는 SB3가 반복문을 실행한다.
평가 구간에서는 학생 코드가 `predict()`와 `step()`을 연결한다.
`predict()`를 호출하는 동안에는 정책 가중치를 갱신하지 않는다.
저장한 정책을 복원하는 기본 호출 방식은 공식 예제에서도 확인할 수 있다. [SB3 기본 사용 예제](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#basic-usage-training-saving-loading)

## 3. 로봇이 수행할 과제

### 3.1 시작 조건과 완료 조건

좌표는 로봇 `base_link` 기준이며 길이 단위는 m이다.

| 항목 | 조건 |
|---|---|
| 초록 큐브 시작 XY | `(0.310, 0.030)` |
| 파란 큐브 시작 XY | `(0.370, 0.000)` |
| 큐브 크기·질량 | 한 변 25 mm, 각각 11 g |
| 초기 회전 | 두 큐브 모두 yaw 0 |
| 초기 EEF XYZ | 약 `(0.225, 0, 0.015)` |
| 성공 | 두 큐브 사이의 물리 접촉 발생 |
| 실패 | 어느 큐브든 허용 XY·높이를 벗어남 |
| 시간 제한 | 성공·실패 없이 180 Step 도달 |

두 큐브는 모두 자유 물체이므로 접촉하면 이동·회전할 수 있다.
파란 큐브가 목표 역할을 한다. 로봇과 파란 큐브의 접촉 여부는 `info`에 기록한다.
현재 과제에서는 이 접촉 자체를 실패로 판정하지 않는다.

성공은 MuJoCo가 계산한 두 큐브의 충돌 접촉으로 판정한다.
중심 사이 거리와 접촉 여부는 구분해서 읽는다. 성공했을 때에도 중심 사이 거리는 큐브 크기만큼 남을 수 있다.

실패 XY 범위는 X `[0.14, 0.41]`, Y `[-0.13, 0.13]`이다.
큐브 중심의 작업면 기준 높이는 `[-0.002, 0.065]`를 허용한다.
작업면 Z는 약 `-0.0048496`이다.

성공과 실패가 같은 Step에 발생하면 두 정보가 함께 참이며 성공·실패 보상도 함께 적용한다.
과제 종료가 발생한 Step에서는 시간 제한 신호를 거짓으로 반환한다.

### 3.2 XYZ 이동 요청과 실제 움직임

Action은 `float32`의 `(3,)` 배열이며 각 값은 `[-1, 1]` 범위이다.

| 성분 | 한 Step의 최대 요청 이동량 |
|---|---|
| `ax` | X ±0.008 m |
| `ay` | Y ±0.008 m |
| `az` | Z ±0.005 m |

예를 들어 `[0.5, -0.25, 1.0]`은 `[+0.004, -0.002, +0.005]` m 이동을 요청한다.
EEF 목표 위치는 X `[0.15, 0.40]`, Y `[-0.14, 0.14]`, Z `[0.008, 0.035]`로 제한한다.
이 범위는 요청 목표의 제한이다. 실제 EEF는 관절 구동과 물리 계산의 영향을 받는다.

IK는 `shoulder_pan`, `shoulder_lift`, `elbow_flex`의 세 관절을 사용한다.
손목 두 관절과 그리퍼는 각각 `0`, `0`, `0.20` rad의 구동 목표를 유지한다.
IK의 제어점은 `gripperframe`이다.
물리 계산은 2 ms마다 실행하고, 한 Action에 최대 25회 계산하므로 기본 제어 주기는 50 ms이다.
큐브 접촉을 감지하면 해당 Action의 물리 계산을 일찍 끝낸다.

이동 요청과 실측 이동은 달라질 수 있다.
관절 구동의 과도 응답 때문에 초기 Z 단독 양의 Action에서도 첫 Step의 높이가 잠시 낮아질 수 있다.
학생은 요청·실측 값을 따로 관찰하고 여러 Step의 움직임을 확인한다.

### 3.3 제어점과 손끝 관측점

`gripperframe`은 IK가 이동 목표로 사용하는 점이다.
`moving_fingertip`은 움직이는 그리퍼 손가락에 정의된 관측 기준점이다.
두 점은 로봇 모델에서 위치가 다르며 초기 손끝 관측점은 약 `(0.254, -0.0015, 0.0349)` m이다.

Top 화면에서는 노란색 `EEF/IK`, 자홍색 `MOVING TIP`으로 구분한다.
좌표를 비교할 때는 어떤 점을 나타내는지 먼저 확인한다.

## 4. 15차원 Observation을 읽는다

### 4.1 실제 관절각과 XYZ를 순서대로 담는다

학생이 수치를 직접 읽고 의미를 설명할 수 있도록 관절각과 위치를 그대로 사용한다.
로봇의 관절각 6개, 손끝 XYZ 3개, 초록 큐브 XYZ 3개, 파란 큐브 XYZ 3개를 이어 붙이면 15개가 된다.
위치는 모두 `base_link` 좌표계의 m 단위이고 관절각은 rad 단위이다.

| 인덱스 | 내용 | 단위 |
|---|---|---|
| 0–5 | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper의 실측 각도 | rad |
| 6–8 | moving_fingertip의 XYZ | m |
| 9–11 | 초록 큐브 중심 XYZ | m |
| 12–14 | 파란 큐브 중심 XYZ | m |

예를 들어 `observation[9:12]`를 읽으면 초록 큐브의 현재 X·Y·Z를 알 수 있다.
`observation[8]`은 손끝의 Z이고, `observation[11]`은 초록 큐브 중심의 Z이다.
제어점 EEF의 XYZ는 `info['eef_position']`에서 확인한다.

### 4.2 Observation Space와 측정값

관측은 `(15,)` 모양의 `float32` 배열이다.
각 값을 실제 단위로 반환하며 Observation Space는 `Box(-inf, inf, (15,), float32)`로 선언한다.
공간 선언의 무한대는 관측값을 수용할 범위를 나타낸다. 반환하는 값 자체는 유한한 실측값이다.

```text
관절각 6개 + 손끝 XYZ + 초록 XYZ + 파랑 XYZ
                         ↓
                실측값 15개를 이어 붙임
                         ↓
                 float32 배열 → Actor
```

카메라의 Top/Front 영상과 픽셀 표시는 Viewer에서 위치를 관찰하는 데 사용한다.
정책이 받는 15개 값의 순서는 위 표를 따른다.
Action은 EEF 이동 요청 XYZ 3개이고, Observation은 실제 상황을 알려 주는 15개 값이다.

## 5. 두 거리로 보상을 계산한다

### 5.1 가까울수록 거리 비용이 작아진다

손끝과 초록 큐브 중심의 XYZ 거리를 `d_tip`, 초록과 파랑 큐브 중심의 XYZ 거리를 `d_cubes`라고 하자.
각 거리는 두 XYZ 벡터의 차이에 `np.linalg.norm()`을 적용하여 구한다.

```text
reward = -10 × (d_tip + d_cubes)
         + 성공하면 100
         - 실패하면 100
```

손끝이 초록 큐브에 가까워지고 두 큐브가 가까워질수록 음수 비용의 크기가 작아진다.
예를 들어 두 거리가 각각 0.05 m와 0.08 m이면 거리 비용은 `-1.3`이다.
이때 성공하면 총 보상은 `98.7`, 실패하면 `-101.3`이다.
성공과 실패가 같은 Step에 발생하면 두 종료 항목을 함께 더한다.

`distance_weight=10`은 m 단위 거리 비용의 크기를 조절한다.
`reward.py`는 거리·성공·실패 세 항목과 총합을 반환한다.
`info['reward_terms']`를 읽으면 어떤 항목이 점수에 기여했는지 확인할 수 있다.

### 5.2 보상 함수 호출

보상 함수는 손끝–초록 거리, 초록–파랑 거리, 성공 여부, 실패 여부, 공통 보상 설정, 거리 가중치 설정의 순서로 값을 받는다.

```python
reward, terms = week11_reward(
    0.05, 0.08, False, False,
    environment.full_config['task']['reward_common'],
    environment.full_config['reward'],
)
```

첫 반환값은 총 보상이고 두 번째는 항목별 보상을 담은 사전이다.

### 5.3 왜 두 거리를 함께 사용하는가

손끝–초록 거리는 큐브에 접근하는 행동에 피드백을 준다.
초록–파랑 거리는 두 큐브를 가까이 보내는 행동에 피드백을 준다.
두 거리만으로 바람직한 밀기 방향을 모두 표현하기는 어려우므로 실제 접촉과 성공률도 확인한다.

### 5.4 Reward·Return·성공률

Reward는 한 Step의 값이고 Return은 한 Episode의 보상 합이다.
학습 목표에는 할인율이 사용되며 로그의 Episode Return은 할인하지 않은 합계이다.
평가 성공률은 성공 Episode 수를 전체 평가 Episode 수로 나눈 값이다.
거리 비용이 줄었는지와 실제 두 큐브 접촉 성공 여부를 함께 읽는다.

## 6. 학습·저장·평가의 흐름

### 6.1 각 프로그램의 역할

| 파일 | 역할 |
|---|---|
| `env.py` | 모델 로딩, reset·step, IK, 접촉·종료, 렌더링 |
| `observation.py` | 관절각·XYZ를 15차원 실측 배열로 구성 |
| `reward.py` | 세 항목의 보상 계산, TODO 1 |
| `train.py` | SAC 생성·학습·저장, TODO 2–4 |
| `evaluate.py` | 정책 불러오기·예측·평가 결과 저장, TODO 5–6 |
| `inspect_environment.py` | API·초기 배치·한 Step 확인 |
| `visualize.py`, `visualization.py` | 자유 시점·Top/Front 화면과 좌표 표시 |
| `config.yaml` | 과제·보상·학습·평가 설정 |
| `models/` | 독립 실행에 필요한 XML·STL·PNG |

파일을 프로젝트의 `week11_sac/`에 배치한 뒤 클래스가 정의된 모듈에서 가져온다.

```python
from week11_sac.env import PushEnv
```

### 6.2 학습을 실행한다

`SAC`에 `MlpPolicy`, 환경, 설정값을 전달해 모델을 만든다.
`learn(total_timesteps=...)`을 호출하면 라이브러리가 환경 초기화·행동 선택·Step·경험 수집·갱신을 반복한다.
`Monitor`는 Episode의 Return·길이와 성공 정보를 기록한다.
`save()`는 학습한 모델을 ZIP 파일로 저장한다.

학습마다 별도 실행 폴더를 만든다. 같은 실행 이름을 다시 사용하면 기존 파일 보호를 위해 오류를 출력한다.
`outputs/latest_run.txt`는 최종 정책 저장까지 완료한 최근 실행을 가리킨다.
기본 설정에서는 학습 중 최근 성공률이 99%에 도달하면 별도 20 Episode 검증을 수행한다.
이 검증에서도 99% 이상이면 조기 종료하며, 그때까지 수행한 Step 수를 기록한다.

### 6.3 저장한 정책을 평가한다

`SAC.load()`로 정책을 불러오고 `predict()`로 Action을 계산한다.
평가 루프는 환경에 Action을 전달하고 종료될 때까지 반복한다.
최종 평가는 40 Episode이며 `deterministic=True`를 사용한다.
고정 배치·결정적 정책이라 여러 Episode가 같은 결과를 보일 수 있다.
이 성공률은 해당 고정 과제에서의 결과로 해석한다.

| 결과 파일 | 읽을 내용 |
|---|---|
| `policy_final.zip` | 학습 종료 시 저장한 정책 |
| `policy_best.zip` | 별도 검증이 수행된 경우의 최상위 정책 |
| `training.monitor.csv` | 학습 Episode의 Return·길이·성공 |
| `metrics.json` | 실행 Step·학습 시간·가중치 갱신 수 |
| `run_config.yaml` | 실행에 적용한 설정 |
| `evaluation_final/evaluation_summary.json` | 최종 평가 성공률·평균 거리 |
| `evaluation_final/evaluation_episodes.csv` | 회차별 성공·길이·종료 원인 |

모델에는 환경 버전을 저장하고 평가 시 일치 여부를 검사한다.
관측의 의미나 제어·물리 동작을 바꿨다면 새 정책으로 다시 학습한다.

## 7. 실습을 마치며

학생은 보상 세 항목, SAC 생성·학습·저장, 정책 불러오기·예측의 여섯 곳을 작성한다.
완료 후에는 Actor와 환경 사이의 입출력, Replay Buffer의 역할, Reward와 Return,
그리고 평가 성공률을 자신의 실행 결과에 연결해 설명한다.
12주차에는 같은 접촉 과제의 초기 배치를 다양화하고 보상을 확장한다.

실행 순서는 [실습 안내서](practice.md)를 따른다.
