# [Week 03] ROS 2 Basic 2 — Jetson & Manipulator

3주차 ROS 2 실습 워크스페이스입니다.
토픽(Topic) 통신과 서비스(Service) 통신을 활용하여 로봇 암(6축 서보 모터)의 관절 상태(`JointState`)를 발행하고, 서비스 요청에 응답하는 ROS 2 Python 패키지(`week03_ros2_jetson`)를 포함합니다.

---

## 📁 디렉터리 구조 (MECE)

중복 요소를 제거하고 역할별로 명확하게 분리된 구조입니다.

```text
week3/ (week03)
├── README.md                      # [현재 파일] 워크스페이스 개요 및 퀵 가이드
├── docs/                          # 학습 문서 및 강의 자료
│   ├── lecture.md                 # 3주차 이론 강의록 (JointState, Service, Launch)
│   ├── practice.md                # 3주차 실습 안내서
│   ├── 실습_진행_가이드.md          # 단계별 실습 매뉴얼 및 조교 체크리스트
│   ├── 실습_하드웨어_장치_가이드.md # 로봇 암, 서보 모터, Jetson 하드웨어 상세 매뉴얼
│   └── week3_ROS2_basic2.pptx     # 3주차 수업 발표 슬라이드
├── scripts/                       # 환경 설정 및 유틸리티 스크립트
│   └── fix_setup.sh               # Jetson setup.bash 경로 에러 수정 스크립트
└── week03_ros2_jetson/            # ROS 2 소스코드 패키지 (colcon 빌드 대상)
    ├── package.xml                # ROS 2 패키지 매니페스트 (의존성 정의)
    ├── setup.cfg                  # Python 패키지 실행 스크립트 디렉터리 설정
    ├── setup.py                   # 노드 실행 진입점(entry_points) 및 데이터 파일 설치 설정
    ├── config/
    │   └── joints.yaml            # 6축 서보 모터 ID, 관절 이름, 오프셋 설정
    ├── launch/
    │   ├── bringup.launch.py      # Publisher + Service Server 동시 실행 런치
    │   └── joint_monitor.launch.py# Publisher + Listener 동시 실행 런치
    ├── resource/
    │   └── week03_ros2_jetson     # ament 인덱스 리소스 마커
    └── week03_ros2_jetson/        # Python 노드 소스코드
        ├── __init__.py
        ├── hardware_interface.py  # Feetech 서보 SDK 연동 하드웨어 제어 클래스
        ├── joint_state_publisher.py       # 20Hz /joint_states 토픽 발행 노드
        ├── joint_state_listener.py        # /joint_states 토픽 구독 및 출력 노드
        ├── joint_state_service_server.py  # /get_joint_state 서비스 서버 (std_srvs/Trigger)
        └── joint_state_service_client.py  # /get_joint_state 서비스 요청 클라이언트
```

---

## 🚀 빠른 시작 가이드 (Quick Start)

### 1. 환경 준비 및 점검
```bash
# 기본 ROS 2 환경 로드
source ~/ros2_base/install/setup.bash

# (옵션) setup.bash 경로 에러 시 스크립트 1회 실행
bash ~/week3/scripts/fix_setup.sh

# 로봇 암 시리얼 포트 확인
ls -l /dev/ttyACM0
# 권한 에러 시: sudo chmod 666 /dev/ttyACM0
```

### 2. 패키지 빌드
```bash
cd ~/week3
source ~/ros2_base/install/setup.bash

# week03_ros2_jetson 패키지 빌드
colcon build --symlink-install --packages-select week03_ros2_jetson

# 빌드 산출물 환경 로드
source install/setup.bash
```

### 3. 노드 실행

#### [실습 1] 토픽 통신 확인 (모니터링 런치)
```bash
# 터미널 1: Publisher + Listener 실행
ros2 launch week03_ros2_jetson joint_monitor.launch.py

# 터미널 2: 토픽 상태 검증
ros2 topic list
ros2 topic echo /joint_states
ros2 topic hz /joint_states
```

#### [실습 2] 서비스 통신 확인 (통합 브링업 런치)
```bash
# 터미널 1: Publisher + Service Server 실행
ros2 launch week03_ros2_jetson bringup.launch.py

# 터미널 2: 클라이언트 노드로 관절값 요청
ros2 run week03_ros2_jetson joint_state_service_client

# 또는 CLI 명령어로 직접 서비스 호출
ros2 service call /get_joint_state std_srvs/srv/Trigger "{}"
```

> **주의**: 로봇 암 시리얼 포트(`/dev/ttyACM0`)는 단 하나의 `joint_state_publisher`만 접근 가능하므로, 런치 파일은 한 번에 하나씩만 실행해야 합니다.

---

## 📚 상세 문서 바로가기
- **[실습 진행 가이드 (체크리스트 포함)](docs/실습_진행_가이드.md)**
- **[실습 하드웨어 장치 가이드 (원리 및 트러블슈팅)](docs/실습_하드웨어_장치_가이드.md)**
- **[이론 강의록](docs/lecture.md)**
- **[실습 안내서](docs/practice.md)**
