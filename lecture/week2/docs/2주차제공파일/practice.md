# [2주차] ROS2 basic 1 — 실습 안내서

> 수업 내용(`lecture.md`)을 참고하여 아래 실습 과제를 순서대로 수행함.

---

## PC 실습 환경 설치 가이드 (월요일 분반만, 최초 1회 설정)

아래 순서대로 WSL2와 Ubuntu 22.04를 설치한 뒤, Ubuntu에 ROS 2 Humble 패키지
저장소를 먼저 등록하고 ROS 2와 개발 도구를 설치함.

### 1. WSL2와 Ubuntu 22.04 설치

Windows PowerShell을 관리자 권한으로 열고 다음 명령을 실행함.

```powershell
wsl --install -d Ubuntu-22.04
```

설치가 완료되면 Ubuntu를 실행하고, 최초 실행 시 사용할 Linux 사용자 이름과
암호를 생성함.

> ID: user
> PW: user 로 설정할 것


### 2. ROS 2 패키지 저장소 등록

이 단계부터는 WSL의 Ubuntu 22.04 터미널에서 실행함. ROS 2 패키지를 설치하기
전에 Ubuntu가 공식 ROS 2 패키지를 찾을 수 있도록 저장소를 먼저 등록함.

```bash
sudo apt update
sudo apt install -y software-properties-common curl

sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
```

### 3. ROS 2 Humble과 개발 도구 설치

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-dev-tools
```

설치 후 ROS 2 명령을 사용하려면 현재 터미널에서 반드시 다음 명령으로 ROS 2
환경을 불러와야 함.

```bash
source /opt/ros/humble/setup.bash
```

이 환경 설정은 현재 터미널에만 적용되므로 새 터미널을 열 때마다 위 명령을 다시
실행해야 함.

---

## 실습 과제

- 주차별 Workspace인 `week2` 디렉터리를 만들고 ROS 2 Humble 환경을 소싱함
- `rclpy`와 `std_msgs` 의존성을 포함하는 `week2_ros2_pc` Python 패키지를 생성함 (`ros2 pkg create`)
- 제공 디렉터리(`student/provided_files/week2_ros2_pc/`)의 스켈레톤 파일들을 새로 만든 패키지 디렉터리로 복사함
- 복사한 `week2_ros2_pc/week2_ros2_pc/hello_talker.py`의 `# TODO` 주석을 참고하여 소스 코드를 완성함
  - Node 클래스를 상속받아 노드 이름 `'hello_talker'` 지정
  - `/hello_topic` 토픽으로 `std_msgs.msg.String` 타입 메시지(`"Hello ROS2 #카운트"`)를 1.0초 주기로 발행하도록 구현
- 복사한 `week2_ros2_pc/week2_ros2_pc/hello_listener.py`의 `# TODO` 주석을 참고하여 소스 코드를 완성함
  - Node 클래스를 상속받아 노드 이름 `'hello_listener'` 지정
  - `/hello_topic` 토픽 수신 시 터미널에 메시지를 출력하도록 구현
- `week2_ros2_pc/setup.py`의 `console_scripts`에 `hello_talker`와 `hello_listener` 실행 진입점을 등록함
- Workspace 최상위(`week2`)에서 `colcon build`를 실행하여 패키지를 빌드함
- 빌드 완료 후 `install/setup.bash` 오버레이 환경을 터미널에 적용함
- source /opt/ros/humble/setup.bash 명령어로 ros2 환경 불러옴
- 서로 다른 터미널에서 `ros2 run`으로 `hello_talker`와 `hello_listener` 노드를 각각 실행하여 메시지 송수신을 확인함
- 세 번째 터미널에서 ROS 2 CLI 도구를 사용하여 실행 중인 노드 목록, 토픽 정보, 실시간 메시지 데이터를 관찰함 (`ros2 node list`, `ros2 topic list`, `ros2 topic info`, `ros2 topic echo`)

---

## 조교 검사

다음 내용을 조교에게 검사받는다.

- `hello_talker`와 `hello_listener` 노드가 정상적으로 실행되는지
- `Hello ROS2 #카운트` 메시지가 1초마다 정상적으로 송수신되는지
- `/hello_talker`, `/hello_listener` 노드와 `/hello_topic` 토픽이 조회되는지
- `ros2 topic echo /hello_topic`에서 발행된 메시지를 확인할 수 있는지
