# [2주차] ROS2 basic 1 — 수업 내용 (이론)

## 0. 수업 개요

### 수업 목표
* Windows PC에 WSL2와 Ubuntu 22.04 환경을 준비하고 각 개발 요소의 역할 설명
* ROS 2 Workspace와 Package의 개념 및 차이점 설명
* Python 기반 ROS 2 Package를 생성하고 `colcon`으로 빌드하는 구조 이해
* Publisher와 Subscriber가 Topic을 통해 통신하는 구조(ROS Graph) 설명
* `ros2 node`, `ros2 topic` 명령으로 실행 중인 ROS graph를 관찰하는 도구 이해

### 수업 내용
1. WSL2, Ubuntu, ROS 2의 역할과 관계
2. ROS 2 Workspace와 Package의 기본 구조 및 생성 시각화
3. Node, Topic, Publisher, Subscriber의 통신 관계 및 코드 구동 원리
4. `ros2 pkg create`를 이용한 Package 생성과 파일 역할
5. `setup.py`의 `console_scripts`와 ROS 2 실행 파일 등록 원리
6. `colcon build`와 `source install/setup.bash` 오버레이 메커니즘
7. ROS 2 CLI를 이용한 노드 및 토픽 관찰

---

## 1. WSL2, Ubuntu, ROS 2의 역할과 관계

Windows PC 환경에서 ROS 2를 구동하기 위한 개발 환경 계층 구조:

```text
+-------------------------------------------------------------------+
|  ROS 2 Humble (로봇 미들웨어: 노드 통신, 빌드 도구 제공)               |
+-------------------------------------------------------------------+
|  Ubuntu 22.04 LTS (Linux OS: ROS 2 실행 표준 기본 OS)               |
+-------------------------------------------------------------------+
|  WSL2 (Windows Subsystem for Linux 2: Hyper-V 기반 가상화 커널)     |
+-------------------------------------------------------------------+
|  Windows 10 / 11 (호스트 PC OS)                                    |
+-------------------------------------------------------------------+
```

* **WSL2**: Windows 내부에서 Linux 커널을 동작시키는 가상화 시스템
* **Ubuntu 22.04**: ROS 2 Humble 버전의 Tier 1 지원 Linux 운영체제
* **ROS 2 Humble**: 로봇의 센서, 제어, 통신을 모듈화하여 개발하는 미들웨어 플랫폼

---

## 2. ROS 2 Workspace와 Package의 구조 및 시각화

### 2.1 Workspace와 Package 개념 차이
* **Workspace (작업 공간)**: 여러 ROS 2 패키지를 모아 소스 코드 작성, 빌드(`colcon`), 환경 설정(`source`)을 수행하는 최상위 프로젝트 디렉터리
* **Package (패키지)**: 독립적인 기능을 수행하는 노드, 설정 파일, 리소스가 포함된 최소 소프트웨어 단위

### 2.2 패키지 생성 명령어와 디렉터리 구조 시각화

이번 실습에서는 주차별 작업 공간인 `week2` 디렉터리를 만들고, 그 안에 ROS 2 패키지를 생성한다. 이후 코드 작성, 빌드, 실행도 모두 이 작업 공간을 기준으로 진행한다.

```bash
# 1. 워크스페이스 디렉터리 생성 및 이동
mkdir -p week2 && cd week2

# 2. Python 패키지 생성 (의존성: rclpy, std_msgs)
ros2 pkg create week2_ros2_pc \
  --build-type ament_python \
  --dependencies rclpy std_msgs
```

명령의 각 부분은 다음 의미를 가짐.

| 명령 요소 | 의미 |
|---|---|
| `ros2 pkg create` | 새 ROS 2 패키지의 기본 파일과 디렉터리를 생성함 |
| `week2_ros2_pc` | 생성할 ROS 2 패키지 이름이며 Python 모듈 이름으로도 사용함 |
| `--build-type ament_python` | CMake가 아닌 Python `setuptools` 기반 패키지로 생성함 |
| `--dependencies rclpy std_msgs` | 이 패키지가 사용하는 ROS 2 의존성을 `package.xml`에 자동 등록함 |

두 의존성의 역할은 다음과 같음.

- **`rclpy`**: Python에서 ROS 2 노드, Publisher, Subscriber, Timer와 실행 루프를 만들기 위한 클라이언트 라이브러리
- **`std_msgs`**: 문자열, 정수, 실수 등 기본 메시지 타입을 제공하는 패키지이며 이번 실습에서는 `std_msgs/msg/String`을 사용함

따라서 위 명령은 “`rclpy`로 노드를 만들고 `std_msgs`의 메시지를 사용하는 Python ROS 2 패키지 `week2_ros2_pc`를 생성한다”는 뜻임.

▼ **`ros2 pkg create` 실행 직후 자동 생성되는 구조 시각화:**

```text
week2/                                 <-- [Workspace] 빌드 및 환경 설정의 최상위 폴더
└── week2_ros2_pc/                     <-- [Package] ros2 pkg create가 생성한 패키지 폴더
    ├── week2_ros2_pc/                 <-- 실제 Python 노드 코드가 들어가는 모듈 디렉터리
    │   └── __init__.py                # Python 패키지 인식 파일
    ├── resource/                      <-- ament 빌드 시스템 등록용 마커
    │   └── week2_ros2_pc
    ├── package.xml                    <-- 패키지 이름, 버전, 의존성(rclpy, std_msgs) 선언
    ├── setup.cfg                      <-- setuptools 기본 설치 경로 설정
    └── setup.py                       <-- 빌드 설정 및 ros2 run 실행 키워드 등록
```

### 2.3 `colcon build` 실행 후 빌드 결과물 구조 시각화

Workspace 최상위(`week2`)에서 `colcon build`를 실행하면 다음 디렉터리가 추가로 생성됨.

```bash
# Workspace 최상위에서 빌드 실행
colcon build --symlink-install --packages-select week2_ros2_pc
```

▼ **`colcon build` 실행 후 최종 Workspace 구조 시각화:**

```text
week2/                                 <-- [Workspace 최상위]
├── week2_ros2_pc/                     <-- (개발자가 수정하는 원본 소스 패키지)
├── build/                             <-- [자동 생성] colcon 빌드 중간 과정 파일 보관
├── install/                           <-- [자동 생성] 빌드 완료 파일 및 setup.bash 오버레이 환경 위치
└── log/                               <-- [자동 생성] 빌드 및 실행 과정의 로그 기록
```

> **참고 사항**: 
> - 소스 코드 작성 및 수정: `week2_ros2_pc/week2_ros2_pc/` 디렉터리 내부
> - 빌드(`colcon build`) 및 환경 적용(`source install/setup.bash`): Workspace 최상위(`week2`) 디렉터리에서 수행함

---

## 3. ROS Graph — Node, Topic, Publisher, Subscriber의 관계 및 코드 구조

ROS 2의 기본 통신 구조는 노드 간 **발행/구독(Publish/Subscribe)** 모델인 **Topic 통신**으로 이루어짐.

```text
 [ hello_talker (Node) ]  ──(Publish)──>  [ /hello_topic (Topic) ]  ──(Subscribe)──>  [ hello_listener (Node) ]
   (1초마다 메시지 송신)                  (타입: std_msgs/msg/String)               (메시지 수신 시 출력)
```

### 3.1 `rclpy`와 Node 클래스

`rclpy`는 Python 프로그램이 ROS 2의 노드, 토픽, 서비스, 타이머 및 실행기를 사용할 수 있게 해주는 ROS 2 Python 클라이언트 라이브러리임.

```python
import rclpy
from rclpy.node import Node
```

Python에서 ROS 2 노드를 작성할 때는 `rclpy.node.Node`를 상속하여 클래스를 정의함. `Node`를 상속하면 다음과 같은 ROS 2 기능을 메서드로 사용할 수 있음.

- `create_publisher()`: Publisher 생성
- `create_subscription()`: Subscriber 생성
- `create_timer()`: 주기적인 Timer callback 등록
- `get_logger()`: ROS 2 Logger 사용
- `destroy_node()`: 노드가 사용하던 ROS 2 자원 해제

`super().__init__(노드이름)`은 상위 `Node` 클래스의 생성자를 호출하고 ROS graph에 표시될 노드 이름을 지정함.

### 3.2 QoS와 `depth`

QoS(Quality of Service)는 ROS 2가 메시지를 보관하고 전달하는 방식을 정하는 통신 정책임. 2주차 코드에서는 Publisher와 Subscriber를 생성할 때 마지막 인자로 `10`을 전달함.

이 숫자는 기본 QoS 설정에서 사용하는 `depth`이며, `KEEP_LAST` 방식으로 최근 메시지를 최대 10개까지 보관할 수 있도록 설정한다는 의미임.

```text
Publisher가 메시지를 발행
          │
          ▼
최근 메시지 대기 영역(depth=10)
          │
          ▼
Subscriber callback에서 처리
```

Publisher가 메시지를 보내는 속도보다 Subscriber가 처리하는 속도가 느리면 메시지가 대기할 수 있음. 대기 영역이 가득 차면 오래된 메시지가 밀려날 수 있음.

| `depth` 예 | 특성 | 사용 예 |
|---:|---|---|
| `1` | 최신 메시지 중심, 오래된 데이터가 빠르게 교체됨 | 카메라나 현재 센서 상태 |
| `10` | 일시적인 처리 지연을 흡수하기에 무난함 | 이번 문자열 송수신 실습 |
| 큰 값 | 더 많은 메시지를 보관하지만 메모리와 지연이 증가할 수 있음 | 처리 순서를 더 오래 유지해야 하는 경우 |

`depth=10`이 메시지의 무조건적인 전달을 보장하는 것은 아님. ROS 2 QoS에는 신뢰성(Reliability), 내구성(Durability), 이력(History) 같은 다른 정책도 있으며 Publisher와 Subscriber의 QoS가 서로 호환되어야 통신할 수 있음. 이번 실습에서는 “최근 대기 메시지를 최대 10개 보관하는 기본 QoS” 정도로 이해함.

명시적인 `QoSProfile` 객체를 사용할 때는 `depth`를 원하는 대기 개수로 설정한 뒤 Publisher 또는 Subscriber 생성 메서드에 전달함. 이번 실습에서는 해당 객체를 직접 작성하지 않고 숫자 형태의 축약 인자를 사용함.

### 3.3 Publisher와 Timer의 구동 원리

* **Publisher 생성**: `self.create_publisher(메시지타입, '토픽이름', QoS큐크기)`
  - 지정한 토픽 이름(예: `'/hello_topic'`)과 메시지 타입(예: `String`)으로 메시지를 내보내는 발행자 객체 생성
* **타이머 및 콜백**: `self.create_timer(주기, 콜백함수)`
  - `Node`가 제공하는 ROS 2 메서드이며 지정한 주기마다 실행할 콜백을 Executor에 등록함
  - 이번 실습에서는 1초 주기와 Talker의 발행 콜백을 연결함
  - 일반 Python의 `while True`와 `time.sleep()`을 사용하는 대신 ROS 2 실행기가 다른 콜백과 함께 관리함
  - 운영체제 부하나 콜백 실행 시간의 영향을 받으므로 정확한 실시간 주기를 보장하는 기능은 아님
* **메시지 발행**: Publisher 객체의 `publish(메시지객체)` 메서드 호출
  - 생성된 메시지 객체를 토픽 통로로 전송함

Timer 객체를 `self._timer`에 저장하면 노드가 실행되는 동안 Timer가 유지되며 필요할 때 `cancel()` 또는 `reset()`할 수 있음.

### 3.4 Subscriber (구독자) 구동 원리
* **Subscriber 생성**: `self.create_subscription(메시지타입, '토픽이름', 콜백함수, QoS큐크기)`
  - 지정한 토픽(예: `'/hello_topic'`)을 구독 등록하고, 메시지 수신 시 실행할 콜백 함수 지정
* **수신 콜백(Callback)**: `def _on_message(self, message: String):`
  - 메시지 도착 시 ROS 2 Executor가 수신한 `message` 객체를 인자로 전달하여 실행함

Subscription 객체도 `self._subscription`에 저장하여 노드가 실행되는 동안 구독 객체가 유지되도록 함.

### 3.5 노드의 전체 실행 주기

```text
프로그램 시작
    │
    ▼
rclpy.init()
ROS 2 통신 Context 초기화
    │
    ▼
HelloTalker() 또는 HelloListener()
Node와 Publisher/Subscriber/Timer 생성
    │
    ▼
rclpy.spin(node)
Timer 및 메시지 이벤트를 기다리며 callback 실행
    │
    ▼ Ctrl+C
KeyboardInterrupt 처리
    │
    ▼
node.destroy_node()
Node가 소유한 Publisher, Subscriber, Timer 등의 자원 해제
    │
    ▼
rclpy.shutdown()
ROS 2 통신 Context 종료
```

각 코드의 의미는 다음과 같음.

1. `rclpy.init(args=args)`: ROS 2 통신을 사용하기 위한 Context를 초기화함. Node를 만들기 전에 한 번 실행해야 함.
2. `node = HelloTalker()`: Node 객체를 생성하면서 Publisher와 Timer 등 생성자에 작성된 ROS 2 자원을 등록함.
3. `rclpy.spin(node)`: 프로그램이 즉시 종료되지 않게 유지하면서 실행 가능한 Timer callback과 Subscriber callback을 기다리고 호출함.
4. `except KeyboardInterrupt`: 사용자가 `Ctrl+C`를 눌렀을 때 발생하는 종료 요청을 처리함.
5. `node.destroy_node()`: 해당 Node가 사용하던 ROS 2 자원을 명시적으로 해제함.
6. `rclpy.ok()`: ROS 2 Context가 아직 동작 중인지 확인함. 이미 종료된 Context에 `shutdown()`을 다시 호출하는 오류를 방지함.
7. `rclpy.shutdown()`: ROS 2 Context를 종료하고 통신 시스템을 정리함.

`spin()`이 실행 중인 동안 Talker는 Timer 주기에 따라 발행 콜백을 실행하고, Listener는 메시지가 도착할 때마다 수신 콜백을 실행함. `destroy_node()`는 반복 주기를 멈추는 명령이라기보다 Node와 그 자원을 제거하는 정리 단계이며, 전체 ROS 2 Context의 종료는 `shutdown()`이 담당함.

---

## 4. ROS 2 실행 파일 등록 및 빌드/환경 적용 원리

### 4.1 `setup.py`의 `console_scripts` 등록 구조

Python 코드 작성 후 CLI에서 `ros2 run <패키지명> <실행파일명>` 형태로 실행하기 위해 `setup.py`에 명령어 등록함.

▼ **문법 구조 해석:**

```text
 " 실행이름  =  패키지 . 모듈 : 함수 "
   ────────     ───────   ────   ────
       ①           ②       ③      ④

 ① ros2 run 명령에서 사용할 실행 이름
 ② Python 패키지 디렉터리 이름
 ③ 노드 코드가 작성된 Python 파일명 (.py 제외)
 ④ 파일 내부에서 실행을 시작할 함수 이름
```

### 4.2 `package.xml` 의존성(Dependencies)
패키지가 사용하는 ROS 2 기본 라이브러리 선언. `ros2 pkg create` 시 `--dependencies rclpy std_msgs` 옵션을 통해 자동 선언됨.

의존성은 다음과 같은 XML 항목으로 `package.xml`에 기록되며, 실제 패키지명은
코드에서 import하거나 사용하는 ROS 2 패키지에 맞춰 작성함.

```text
<depend>사용하는_ROS_패키지명</depend>
```

### 4.3 `colcon build`와 오버레이(Overlay) 환경

`colcon build`는 작성한 패키지를 빌드하고, 실행에 필요한 파일을 Workspace의 `install/` 디렉터리에 준비한다. 다만 빌드 직후에는 현재 터미널이 새로 만든 패키지의 위치를 알지 못할 수 있다.

```bash
source install/setup.bash
```

위 명령을 실행하면 현재 터미널에서 `install/`에 있는 패키지를 찾고 `ros2 run`으로 실행할 수 있다. 이 명령은 파일을 시스템 ROS 2 디렉터리인 `/opt/ros/humble`에 복사하거나 기존 ROS 2 설치를 수정하지 않는다.

기본 ROS 2 환경에 직접 빌드한 패키지 환경을 추가해서 사용하는 방식을 **오버레이(Overlay)**라고 한다.

```text
기본 ROS 2 환경: /opt/ros/humble
                +
직접 빌드한 패키지 환경: week2/install
                =
현재 터미널에서 사용할 ROS 2 환경
```

`source install/setup.bash`의 환경 설정은 명령을 실행한 터미널에만 적용된다. 새 터미널을 열었다면 Workspace의 최상위 디렉터리인 `week2`로 이동한 후 다시 실행해야 한다.

---

## 5. ROS 2 CLI를 이용한 노드 및 토픽 관찰

실행 중인 ROS Graph 상태를 관찰하기 위한 CLI 도구:

| 구분 | 명령어 | 주요 역할 및 설명 |
| :--- | :--- | :--- |
| **노드 목록** | `ros2 node list` | 실행 중인 노드 이름 확인 (`/hello_talker`, `/hello_listener`) |
| **토픽 목록** | `ros2 topic list` | 활성화된 토픽 목록 확인 (`/hello_topic`) |
| **토픽 타입** | `ros2 topic type /hello_topic` | 토픽이 사용하는 메시지 타입 확인 (`std_msgs/msg/String`) |
| **토픽 정보** | `ros2 topic info /hello_topic` | 해당 토픽의 Publisher 및 Subscriber 수 확인 |
| **토픽 직접 모니터링** | `ros2 topic echo /hello_topic` | 토픽 데이터 실시간 수신 및 출력 |
