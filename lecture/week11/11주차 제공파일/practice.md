# [11주차] 고정 배치 두 큐브 접촉 SAC — 실습 안내서

> `lecture.md`를 먼저 읽고 순서대로 수행한다.
> 명령은 Windows PC의 WSL2 Ubuntu 22.04 터미널에서 실행한다.
> 프로젝트는 `~/week11/sac_lab`, 가상환경은 `~/week11/.venv`를 사용한다.
> 실습 시작과 종료 시 공용 PC의 `~/week11`을 정리한다.

## 0. 실습 전

```text
이전 분반 자료 삭제 → 새 venv → 파일 배치·패키지 설치 → 초기 장면 확인
TODO 1~6 → API·한 Step → 5,200 Step 확인 학습 → 60,000 Step 본 학습
정책 저장·불러오기 → 40 Episode 평가 → Viewer → 기록·제출·정리
```

이번 과제는 고정 배치에서 초록 큐브와 파란 큐브가 실제로 접촉하도록 정책을 학습하는 것이다.
두 큐브는 각각 25 mm·11 g 자유 물체이다. Observation은 15차원, Action은 XYZ 3차원이다.

| 파일 | 실습 범위 |
|---|---|
| `reward.py` | TODO 1, 보상 계산 |
| `train.py` | TODO 2~4, SAC 생성·학습·저장 |
| `evaluate.py` | TODO 5~6, 정책 불러오기·행동 예측 |
| `env.py`, `observation.py` | 환경·관측 제공 코드 |
| `inspect_environment.py` | API·한 Step 확인 |
| `visualize.py`, `visualization.py` | Viewer·Top/Front 화면 제공 코드 |
| `config.yaml`, `models/`, `requirements.txt` | 설정·모델·의존성 |

## 1. 파일 배치와 환경 준비

### 1.1 이전 분반 자료 삭제와 새 가상환경

아래 명령은 `~/week11`의 코드·모델·학습 결과·가상환경을 모두 삭제한다.
보관할 자료는 개인 저장소에 복사하고 폴더 밖으로 이동한 뒤 실행한다.

```bash
cd ~
rm -rf ~/week11
mkdir -p ~/week11/sac_lab
cd ~/week11
python3 -m venv .venv
source .venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
```

마지막 줄은 Python의 `__pycache__`·`.pyc` 생성을 막는다. 새 터미널에서도 다시 적용한다.
`ensurepip is not available` 오류가 발생하면 설치

```bash
sudo apt update
sudo apt install python3.10-venv libgl1 libglfw3
```

설치 후 `cd ~/week11`에서 `python3 -m venv .venv`부터 다시 실행한다.
WSLg가 제공하는 창 표시 환경도 필요하다.

### 1.2 제공 파일 배치

제공된 `source_code`의 내용이 아래 구조로 준비되어 있는지 확인한다.
`week11_sac/`에는 Python 파일 8개, 프로젝트 루트에는 설정·의존성·모델이 있어야 한다.

```text
~/week11/
├── .venv/
└── sac_lab/                       python -m 실행 위치
    ├── config.yaml
    ├── requirements.txt
    ├── models/
    │   ├── push_scene.xml
    │   ├── physicai_arm.xml
    │   ├── robot.xml
    │   └── assets/                STL·PNG 전체
    └── week11_sac/          Python 파일 8개
```

작성할 Python 파일은 `~/week11/sac_lab/week11_sac/` 안에 있다.
`from week11_sac.env import PushEnv`는 이 폴더의 `env.py`에서 클래스를 가져온다.
패키지명 `week11_sac`는 코드 폴더 이름이다.

실습 중 새 터미널을 열었다면 다음 명령으로 실행 위치와 가상환경을 복원한다.

```bash
cd ~/week11/sac_lab
source ~/week11/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1
```


### 1.3 패키지 설치

CPU용 PyTorch를 먼저 설치하고 나머지 의존성을 설치한다.
이번 정책은 관측 15개와 작은 MLP를 사용하며 단일 MuJoCo 환경을 반복 실행한다.
CPU·스레드 1개로 약 10분 내 학습을 검증했으므로 수업에서는 이 실행 조건을 사용한다.
CPU용 설치는 CUDA·GPU 드라이버 조합에 따른 설치 차이를 줄인다.
GPU도 사용할 수 있지만 CUDA 지원 PyTorch 설치가 필요하고, 이 규모의 환경에서 더 빠를지는 별도 측정이 필요하다.

```bash
source ~/week11/.venv/bin/activate
cd ~/week11/sac_lab
python3 -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --no-cache-dir -r requirements.txt
python3 -m pip check
python3 - <<'PY'
import sys
from importlib.metadata import version
assert sys.prefix != sys.base_prefix
for name in ('numpy', 'gymnasium', 'mujoco', 'PyYAML', 'stable-baselines3',
             'torch', 'tensorboard', 'opencv-python'):
    print(name, version(name))
print('PC packages OK')
PY
```

주요 검증 버전은 NumPy `1.26.4`, Gymnasium `1.3.0`, MuJoCo `3.11.0`,
Stable-Baselines3 `2.9.2`, PyTorch `2.9.1+cpu`이다.
`No broken requirements found.`와 `PC packages OK`를 확인한다.

## 2. 초기 환경 확인

TODO 작성 전에 초기화와 관측 구성을 확인한다.
관측 순서는 관절각 6개 → 손끝 XYZ → 초록 XYZ → 파랑 XYZ이다.

```bash
python3 - <<'PY'
from week11_sac.env import PushEnv
environment = PushEnv()
try:
    observation, info = environment.reset(seed=12)
    print(environment.observation_space)
    print(environment.action_space)
    print('object XYZ:', info['object_position'])
    print('target XYZ:', info['target_position'])
    print('EEF XYZ:', info['eef_position'])
    print('moving tip XYZ:', info['moving_tip_position'])
finally:
    environment.close()
PY
```

| 항목 | 예상 값 |
|---|---|
| Observation | `Box(-inf, inf, (15,), float32)` |
| Action | `Box(-1.0, 1.0, (3,), float32)` |
| 초록 XYZ | 약 `[0.31, 0.03, 0.0081504]` |
| 파랑 XYZ | 약 `[0.37, 0.0, 0.0081504]` |
| EEF XYZ | 약 `[0.225, 0, 0.015]` |
| 손끝 XYZ | 약 `[0.253904, -0.001521, 0.034919]` |

```bash
python3 -m week11_sac.visualize
```

자유 시점·Top·Front 세 창을 확인한다.
Top의 `EEF/IK`와 `MOVING TIP`, 초록 `OBJECT`와 파랑 `TARGET`을 구분한다.
`R`로 초기화하고 `Q`로 종료한다. 화면이 키 입력을 받도록 해당 창을 선택한다.
Space 일시정지는 Top/Front 창에서 사용한다. MuJoCo 창이나 카메라 창을 닫아도 프로그램이 종료된다.

## 3. 핵심 코드 TODO 1~6

`lecture.md` 2.6~2.14절의 Stable-Baselines3·API 옵션 설명을 읽고 아래 여섯 곳을 작성한다.
제공 코드의 인자·설정을 참고한다.
`raise NotImplementedError("TODO ...")`를 해당 코드로 교체한다.

### 3.1 TODO 1 — 보상 계산

`reward.py`에서 두 거리의 비용·성공·실패의 세 보상 항목과 총합을 반환하도록 완성한다.

### 3.2 TODO 2 — SAC 생성

`train.py`에서 제공된 환경과 `training` 설정을 사용하여 SAC 모델을 생성한다.

### 3.3 TODO 3 — 학습 실행

같은 파일에서 지정된 Step 수·콜백·로그 설정으로 학습을 실행한다.

### 3.4 TODO 4 — 최종 정책 저장

같은 파일에서 실행 폴더의 `policy_final` 이름으로 학습 결과를 저장한다.

### 3.5 TODO 5 — 정책 불러오기

`evaluate.py`에서 선택한 정책 파일을 평가 환경에 연결하여 CPU로 불러온다.

### 3.6 TODO 6 — 평가 행동 예측

같은 파일의 평가 반복문에서 현재 Observation으로 결정적 Action을 예측한다.

## 4. 소스·API·보상 검사

```bash
python3 - <<'PY'
from pathlib import Path
paths = sorted(Path('week11_sac').glob('*.py'))
assert len(paths) == 8
for path in paths:
    code = path.read_text(encoding='utf-8')
    assert 'raise NotImplementedError' not in code, f'남은 TODO: {path}'
    compile(code, str(path), 'exec')
print('Source check OK')
PY
python3 -m week11_sac.inspect_environment
```

`Gymnasium API OK`를 확인한다. 실측값을 그대로 반환하는 Space의 무한대 경계에 관한 경고와 직접 생성한 환경에 `spec`이 없다는 안내가 나올 수 있다.

| 한 Step 항목 | 검증 환경의 출력 |
|---|---|
| Action XYZ | `[0.5, -0.25, 1.0]` |
| 요청 이동 XYZ | `[0.004, -0.002, 0.005]` m |
| 실측 이동 XYZ | 약 `[0.003332, -0.001876, 0.004188]` m |
| 보상 | 약 `-1.379930` |
| 종료 신호 | `False False` |

이번 보상은 손끝–초록·초록–파랑의 XYZ 거리 비용과 성공·실패로 계산한다. 실측과 보상의 끝자리에는 실행 환경에 따른 차이가 있을 수 있다.

```bash
python3 - <<'PY'
from week11_sac.env import PushEnv
from week11_sac.reward import week11_reward
environment = PushEnv()
try:
    reward, terms = week11_reward(0.05, 0.08, False, False,
        environment.full_config['task']['reward_common'],
        environment.full_config['reward'])
    assert abs(sum(terms.values()) - reward) < 1e-12
    print('Reward check OK', reward, terms)
finally:
    environment.close()
PY
```

이 검사는 총 보상과 항목 합계가 일치하는지 확인한다.
각 항목의 부호·거리 비용·종료 보상은 출력한 `terms`와 수업의 식을 비교한다.

## 5. 학습·정책 저장

### 5.1 짧은 확인 학습

재시도할 때 이전 확인 학습의 정책·로그를 지우고 같은 이름으로 다시 실행한다.
`outputs/smoke`만 삭제하므로 다른 실행 결과는 보관할 수 있다.

```bash
cd ~/week11/sac_lab
rm -rf outputs/smoke
python3 -m week11_sac.train --run-name smoke --steps 5200 --device cpu --threads 1
python3 - <<'PY'
import json
from pathlib import Path
directory = Path('outputs/smoke')
metrics = json.loads((directory / 'metrics.json').read_text())
assert metrics['steps_completed'] == 5200
assert metrics['gradient_updates'] > 0
assert (directory / 'policy_final.zip').is_file()
print('Training/save OK', metrics['gradient_updates'])
PY
python3 -m week11_sac.evaluate --run smoke --episodes-per-scenario 2
```

처음 5,000 Step은 무작위 수집 구간이며, 5,200 Step 완료 시 기본 설정에서 갱신 수는 200이다.
실제 학습 시간과 평가 결과를 기록한다. 확인 학습의 목적은 학습·저장·평가 연결을 점검하는 것이다.
최종 재검증에서는 약 36초가 걸렸고 평가 결과는 0/2였다.
정책 ZIP 생성·불러오기·평가 완료와 실제 가중치 갱신을 확인한다.
5.1절을 다시 실행하면 이전 smoke 결과가 새 결과로 교체된다. 보관할 결과는 먼저 다른 위치에 복사한다.

### 5.2 본 학습

```bash
python3 -m week11_sac.train --run-name baseline --steps 60000 --device cpu --threads 1
```

`Progress`, `rollout/ep_rew_mean`, `rollout/success_rate`, `train/actor_loss`, `train/critic_loss`를 관찰한다.
학습 종료 시 `Saved: .../policy_final.zip`을 확인한다.
실행 시간과 실제 수행 Step은 `outputs/baseline/metrics.json`에 기록된다.
검증 조건을 충족하면 60,000 Step 전에 조기 종료될 수 있다.
`policy_best.zip`은 별도 검증이 실행되었을 때 생성된다.

실행 시간과 성공률의 측정 결과는 정답 폴더의 검증 결과에 기록한다. PC 성능·라이브러리·연산 결과에 따라 시간과 학습 결과는 달라질 수 있다.
15차원 버전의 최종 재검증은 60,000 Step 전체를 수행했고 약 586초가 걸렸다.

## 6. 최종 평가와 Viewer

### 6.1 40 Episode 평가

```bash
python3 -m week11_sac.evaluate --run baseline
python3 - <<'PY'
import json
from pathlib import Path
directory = Path('outputs/baseline')
metrics = json.loads((directory / 'metrics.json').read_text())
summary = json.loads((directory / 'evaluation_final/evaluation_summary.json').read_text())
assert metrics['gradient_updates'] > 0
assert summary['overall']['episodes'] == 40
print('steps:', metrics['steps_completed'])
print('training seconds:', metrics['training_seconds'])
print('success:', summary['overall']['successes'], '/ 40')
print('mean final center distance:', summary['overall']['mean_final_distance_m'])
print('Evaluation OK')
PY
```

고정 배치에서 결정적 행동으로 평가하므로 40회 결과가 모두 같을 수 있다.
성공률·평균 종료 Step·최종 중심 거리와 회차별 `failure_reason`을 기록한다.
시간 제한으로 끝난 회차는 성공·실패 정보가 모두 거짓이고 실패 원인은 빈 문자열이다.
평가 결과는 고정 배치 한 조건에서 얻은 측정값으로 해석한다.
검증 PC에서는 40/40 성공, 평균 55 Step, 최종 중심 거리 약 0.028619 m였다.

### 6.2 학습한 정책 움직임

```bash
python3 -m week11_sac.visualize --run baseline
```

정책 Action에 따라 로봇이 움직이고 Episode가 끝나면 초기화되는 것을 확인한다.
`R`로 리셋, Top/Front 창에서 Space로 일시정지·재개, `Q`로 종료한다.
이동·접촉 여부를 직접 관찰하고 수치 결과와 연결해서 설명한다.
프로그램이 끝나면 터미널 프롬프트로 돌아오는지 확인한다.

## 7. 결과 기록과 자리 정리

| 항목 | 기록할 내용 |
|---|---|
| 과제·입출력 | 시작 배치, 성공·실패, 15차원 특징의 예, XYZ 요청 |
| 보상 | 세 항목과 거리 0.05 m·0.08 m 예제의 총합 |
| 학습 | 실제 Step·갱신 수·학습 시간·Return 변화 |
| 평가 | 40회 성공 수·성공률·최종 거리·종료 원인 |
| Viewer | 제어점·관측점, 로봇과 큐브의 움직임 |
| 해석 | 두 거리 비용이 각각 유도하는 행동과 보상 개선 생각 |

조교 확인과 제출을 마친 뒤 작성한 소스·기록·보관할 정책을 개인 저장소에 복사한다.
실습 종료 전 모든 Viewer를 닫고 다음 명령으로 공용 PC의 실습 파일을 정리한다.

```bash
deactivate
cd ~
rm -rf ~/week11
```

자리를 떠나기 전에 코드·모델·학습 결과·가상환경을 포함한 `~/week11`을 모두 삭제한다.
제출물에는 가상환경·Python 캐시를 포함하지 않는다.

## 8. 문제 해결과 완료 체크

| 증상 | 확인할 내용 |
|---|---|
| 모듈을 찾지 못함 | `~/week11/sac_lab`에서 안내된 `python3 -m ...` 명령 사용 |
| `NotImplementedError` | 해당 TODO 작성 완료 여부 |
| 라이브러리 import 오류 | venv 활성화·requirements 설치·`pip check` |
| `Run already exists` | 새 실행 이름 사용, 후속 평가도 같은 이름 지정 |
| `policy_best.zip` 없음 | 기본 정책 파일 `policy_final.zip` 선택 |
| 정책 버전 불일치 | 이번 환경에서 새로 학습한 정책 사용 |
| Step이 짧게 종료 | `failure_reason`과 큐브 위치·높이 확인 |
| 0% 성공률 | 가중치 갱신·학습 로그·접촉 여부·보상 항목을 함께 확인 |
| Viewer 표시 오류 | WSLg·OpenGL·libgl1·libglfw3·OpenCV GUI 설치 확인 |
| Viewer 키가 반응하지 않음 | 키를 받을 창을 선택하고 Top/Front에서 Space 사용 |

- [ ] 새 가상환경과 독립 모델 배치를 완료한다.
- [ ] TODO 여섯 곳과 Source·API·Reward 검사를 통과한다.
- [ ] 짧은 학습에서 정책 저장·불러오기와 실제 갱신을 확인한다.
- [ ] 본 학습과 40회 평가를 실행하고 측정값을 기록한다.
- [ ] Viewer의 모델·두 큐브·제어점·관측점과 정상 종료를 확인한다.
- [ ] Actor·Critic·Replay Buffer·평가 결과를 설명한다.
- [ ] 제출·백업 후 `~/week11` 전체를 삭제하고 자리를 정리한다.
