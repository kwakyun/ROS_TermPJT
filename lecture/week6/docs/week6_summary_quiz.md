# 6주차 총정리: 게임패드 EEF 제어와 FK/IK

조교님 질문에 개념과 실제 코드 흐름을 연결해 답할 수 있도록 정리한 요약이다.

## 1. 한 문장으로 설명하기

> 게임패드의 `/joy` 입력을 `joystick_to_pose`가 `base_link` 기준 EEF 목표 자세로 바꾸고, `ik_solver`가 이를 5개 arm joint 목표로 계산한다. `arm_driver`가 6개 관절(그리퍼 포함)의 목표를 검사해 서보를 구동하며, 측정된 관절 상태는 `fk_solver`가 현재 EEF 자세로 바꿔 `/ee_pose`에 발행한다.

## 2. 전체 노드와 Topic 흐름

```text
게임패드
  └─ joy_node ── /joy (sensor_msgs/Joy)
       └─ joystick_to_pose
            ├─ /target_pose (geometry_msgs/PoseStamped)
            └─ /gripper_open (std_msgs/Bool)
                 └─ ik_solver
                      └─ /joint_targets (sensor_msgs/JointState)
                           └─ arm_driver ── Feetech 서보
                                └─ /joint_states (sensor_msgs/JointState)
                                     └─ fk_solver
                                          └─ /ee_pose (geometry_msgs/PoseStamped)

/safety/torque_enable (std_msgs/Bool) ──> arm_driver
```

| Topic | 누가 발행하나 | 무엇을 나타내나 |
|---|---|---|
| `/joy` | `joy_node` | 게임패드 축과 버튼 배열 |
| `/target_pose` | `joystick_to_pose` | 목표 XYZ와 방향. `base_link` 기준, 위치 단위 m |
| `/gripper_open` | `joystick_to_pose` | 그리퍼 열림 여부 (`Bool`) |
| `/joint_targets` | `ik_solver` | arm 5개와 `gripper`의 목표 관절값 (rad) |
| `/joint_states` | `arm_driver` | 서보에서 읽은 실제 6개 관절값 (rad) |
| `/ee_pose` | `fk_solver` | 측정 관절값에서 계산한 현재 EEF 위치와 Quaternion |
| `/safety/torque_enable` | 안전 명령 발행자 | `true`는 토크 활성화 요청, `false`는 해제 요청 |

`/target_pose`는 목표이고 `/ee_pose`는 실제 관절 측정값에 기반한 현재 자세다. 따라서 두 값은 같은 의미가 아니며, 움직이는 동안 서로 차이가 날 수 있다.

## 3. 작업 공간, 관절 공간, FK, IK

- **작업 공간 (Task Space):** EEF 위치와 자세로 명령한다. 이번 실습 목표는 `(X, Y, Z, Pitch)`이다.
- **관절 공간 (Joint Space):** 각 관절의 각도로 로봇 자세를 표현한다.
- **FK (Forward Kinematics, 정운동학):** 관절각 → EEF 위치와 방향. `/joint_states`가 입력이고 `/ee_pose`가 결과다.
- **IK (Inverse Kinematics, 역운동학):** EEF 목표 위치·방향 → 관절각. `/target_pose`와 측정 관절 seed가 입력이고 `/joint_targets`가 결과다.

IK는 목표 하나에 대해 여러 관절 해가 있을 수 있고, 도달 불가능한 목표에는 해가 없을 수 있다. 이번 IK는 위치 오차가 0.020 m보다 크면 목표를 거부한다.

### 관절 구성

IK/FK의 arm 관절 5개는 다음 순서다.

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
```

하드웨어 토픽의 `/joint_targets`, `/joint_states`는 여기에 `gripper`를 더해 총 6개 관절을 사용한다. `wrist_roll`은 EEF 작업 공간 목표로 직접 제어하지 않고 현재 값을 유지한다. 그리퍼는 IK로 풀지 않고 별도의 열림·닫힘 목표로 정한다.

## 4. 조이스틱에서 목표 자세까지

`joystick_to_pose`는 `/joy` 메시지를 받으면 축·버튼을 저장하고, 25 Hz Timer가 목표를 갱신한다.

| 현재 코드의 입력 | 동작 |
|---|---|
| `axes[1]` | EEF X |
| `axes[0]` | EEF Y |
| `axes[7]` | EEF Z |
| `axes[3]` | EEF Pitch |
| `buttons[0]` | 그리퍼 토글 |

게임패드 모델마다 버튼의 실제 표기(A/B 등)와 축 배열이 다를 수 있다. 따라서 버튼 이름을 추측하지 말고 먼저 `/joy`를 출력해 `axes[]`, `buttons[]`의 인덱스를 확인한다. 소스의 그리퍼 입력은 `buttons[0]`이다.

축 입력은 속도와 Timer 간격을 곱해 목표에 누적한다.

```text
다음 목표 = 현재 목표 + 축 속도 × 축 입력 × dt
dt = 1 / 25초
```

작은 축 흔들림은 Deadzone (0.05)으로 0 처리한다. 목표는 YAML에 정의된 X/Y/Z/Pitch 범위와 XY 반경 안으로 제한한다. 범위 안의 목표가 전부 도달 가능한 것은 아니므로, 마지막으로 IK가 도달 가능성을 판정한다.

버튼은 계속 누르고 있을 때 반복 토글되지 않도록 이전 값이 0이고 현재 값이 1이 되는 상승 edge에서 한 번 토글한다.

설정 파일의 작업 공간 제한은 다음과 같다. 초기 목표는 `(X, Y, Z, Pitch) = (0.20 m, 0.00 m, 0.10 m, 1.00 rad)`이다.

| 항목 | 범위 |
|---|---:|
| X | 0.05 ~ 0.35 m |
| Y | -0.20 ~ 0.20 m |
| Z | 0.00 ~ 0.10 m |
| Pitch | 0.30 ~ 1.50 rad |
| XY 반경 | 0.145 ~ 0.42 m |

축 Deadzone은 0.05이며 이동 속도 설정은 X/Y 0.075 m/s, Z 0.060 m/s, Pitch 0.90 rad/s이다.

## 5. IK Solver가 해를 찾는 방법

1. 목표 X-Y 방향으로 `shoulder_pan`을 계산한다.
2. 이전 해가 있으면 그것을 seed로 사용하고, 없으면 `/joint_states`에서 얻은 측정 관절값을 사용한다.
3. `shoulder_lift`와 `elbow_flex`를 조금씩 바꾸며 EEF 위치 변화량을 계산한다.
4. 그 변화량으로 3×2 수치 Jacobian을 만들고 Damped Least Squares로 위치 오차를 줄인다.
5. 관절 변화량과 관절 범위를 제한하면서 최대 8회 반복한다.
6. `wrist_flex = target_pitch - shoulder - elbow` 관계를 사용해 Pitch를 맞춘다.
7. 최종 위치 오차가 0.020 m보다 크면 목표를 거부한다.

성공하면 IK arm joint 5개와 별도의 gripper 목표를 합쳐 완전한 6관절 `JointState`를 발행한다. 관절 목표는 encoder tick 단위로 양자화한 뒤 관절 제한 안쪽에 머물도록 다시 제한한다.

## 6. 각 노드의 역할과 주기

| 노드 | 역할 | 주요 동작 |
|---|---|---|
| `joy_node` | 게임패드 드라이버 | 장치 입력을 `/joy`로 발행 |
| `joystick_to_pose` | 조이스틱→EEF 목표 | `/joy` 구독, `/target_pose`와 `/gripper_open` 발행, 25 Hz Timer |
| `ik_solver` | EEF 목표→관절 목표 | 상태·목표·그리퍼 구독, `/joint_targets` 발행, 60 Hz Timer |
| `arm_driver` | 실제 서보 구동 | 목표 검사, 속도 제한, 측정 상태 발행, 토크 요청 처리 |
| `fk_solver` | 관절→현재 EEF | 측정 관절 상태에서 FK 계산 후 `/ee_pose` 발행 |
| `ee_pose_listener` | 현재 자세 표시 | `/ee_pose`를 사람이 읽을 수 있는 로그로 출력 |

`arm_driver`는 관절 목표를 받아 실제 모터를 제어하는 계층이다. EEF 목표나 IK를 계산하지 않는다. YAML 설정 기준 관절 상태 발행은 50 Hz, 하드웨어 제어는 100 Hz이다.

## 7. Launch와 패키지

Launch는 노드를 시작하는 프로세스 설정이며 Topic을 직접 연결하지 않는다. 각 노드가 같은 Topic 이름으로 Publisher/Subscriber를 만들면 ROS 2가 연결한다.

- Launch에 `DeclareLaunchArgument` 두 개와 Node 여섯 개가 있다.
- `device_id`는 `joy_node`가 사용할 SDL 게임패드 장치 번호다.
- `start_joy:=false`이면 Launch가 `joy_node`를 시작하지 않는다.
- ROS 2 패키지 이름은 `week06_ee_control`이다.
- `setup.py`는 실행 명령 5개를 등록하고 config와 launch 파일을 설치한다.
- `package.xml`은 `ament_python`, ROS 메시지/런타임, launch, joy 등 실행 의존성을 선언한다.

빌드 후 실행 파일은 다음 5개다: `arm_driver`, `ee_pose_listener`, `fk_solver`, `ik_solver`, `joystick_to_pose`.

## 8. 입력 단절·제한·안전

### 서로 다른 세 타임아웃

| 계층 | 제한 시간 | 동작 |
|---|---:|---|
| `joystick_to_pose` | 0.50초 동안 새 `/joy` 없음 | 경고 후 새 `/target_pose` 발행 중지 |
| `ik_solver` | 목표가 0.25초보다 오래됨 | 오래된 목표로 `/joint_targets` 발행 중지 |
| `arm_driver` | 300 ms 동안 새 관절 목표 없음 | 진행 중 이동을 멈추고 마지막 명령 자세 유지 |

중립 스틱은 입력 단절이 아니다. 게임패드가 중립 `/joy`를 계속 발행하면 새 목표도 계속 나온다. 타임아웃을 시험하려면 게임패드/`joy_node` 입력을 실제로 끊어야 한다. 자세 유지는 토크 해제와 다르며, 모터 토크가 계속 켜져 있을 수 있다.

### 관절·속도 제한과 토크

- `joints.yaml`의 관절별 `limit_rad`를 벗어나는 전체 목표는 거부한다.
- 관절별 속도 제한은 한 제어 주기의 최대 이동량으로 적용된다: `max_step = max_vel_rad_s × dt`.
- 토크를 켤 때 현재 측정 자세를 첫 hold 목표로 잡아 갑작스러운 점프를 줄인다.
- hold 상태에는 마지막 명령을 Heartbeat (10 Hz)로 다시 전송한다. 이것은 프로세스·케이블·전원 장애를 감시하는 독립 watchdog은 아니다.
- 노드는 시작 및 정상 종료 시 토크 해제를 시도하지만, 강제 종료나 통신 장애 중 성공을 보장하지 않는다.

### 실습 안전 순서

1. 로봇 반경을 비우고 장치·축·토픽을 확인한다. 처음에는 토크를 끈 상태로 `/joy`를 확인한다.
2. 토크 해제 명령을 보낼 별도 터미널을 준비한다.
3. 로봇 주변을 다시 확인한 후에만 `/safety/torque_enable`에 `true`를 보낸다.
4. 작은 스틱 입력으로 천천히 확인한다. 예상과 다른 동작이면 즉시 `false`를 발행한다.
5. 종료할 때 `false`를 보내 토크 해제를 확인하고, 그 뒤 Launch를 종료한다.

토크가 켜진 상태에서 로봇을 손으로 억지로 밀지 않는다. 토크를 끄면 링크가 내려갈 수 있으므로 필요하면 받친다.

## 9. 조교님 예상 질문과 모범 답변

### Q1. FK와 IK의 입력·출력 차이는 무엇인가요?

**답:** FK는 관절각을 입력받아 현재 EEF 자세를 계산합니다. IK는 목표 EEF 위치와 Pitch를 입력받아 이를 만들 관절각을 계산합니다.

### Q2. `/target_pose`와 `/ee_pose`는 어떻게 다른가요?

**답:** `/target_pose`는 조이스틱에서 만든 목표이고, `/ee_pose`는 서보에서 측정한 `/joint_states`에 FK를 적용해 얻은 현재 자세입니다.

### Q3. 왜 IK 결과에 arm joint 5개뿐 아니라 gripper도 있나요?

**답:** IK 자체는 arm 관절 5개만 풉니다. 하지만 `arm_driver`는 `/joint_targets`에 6개 관절 전체를 요구하므로, 그리퍼의 별도 열림/닫힘 목표를 합쳐 발행합니다.

### Q4. 작업 공간 범위 안이면 항상 IK가 성공하나요?

**답:** 아닙니다. 범위 제한은 지나치게 큰 명령을 줄일 뿐이고, 그 안에도 로봇 기구학상 도달 불가능하거나 현재 반복법이 허용 오차 안으로 수렴하지 않는 점이 있습니다. 위치 오차가 2 cm를 넘으면 거부됩니다.

### Q5. `/joint_states`는 명령값인가요?

**답:** 아닙니다. 하드웨어에서 읽은 측정 관절값입니다. 명령값은 `/joint_targets`입니다.

### Q6. 스틱을 놓았는데도 입력 타임아웃이 왜 발생하지 않나요?

**답:** 스틱을 놓으면 축값이 0인 `/joy` 메시지가 계속 들어올 수 있습니다. 타임아웃은 새 메시지가 일정 시간 오지 않을 때 발생합니다.

### Q7. 게임패드 입력이 끊기면 세 계층은 각각 어떻게 동작하나요?

**답:** 0.50초 뒤 조이스틱 노드가 목표 발행을 중단하고, IK는 0.25초보다 오래된 목표를 발행하지 않으며, arm driver는 300 ms 동안 새 관절 목표가 없으면 이동을 멈추고 마지막 명령 자세를 유지합니다.

### Q8. 마지막 자세 유지는 토크 해제와 같은가요?

**답:** 아닙니다. 마지막 자세 유지는 토크를 켠 상태로 명령 자세를 유지하는 동작입니다. 토크 해제는 `/safety/torque_enable`에 `false`를 보내 모터 토크를 끄는 동작입니다.

### Q9. `arm_driver`의 역할은 무엇인가요?

**답:** IK가 만든 전체 관절 목표를 검사하고 속도 제한에 맞춰 Feetech 서보를 구동하며, 실제 관절 상태를 발행하고 토크 요청을 처리합니다. IK나 FK는 계산하지 않습니다.

### Q10. IK는 왜 측정 관절 상태를 구독하나요?

**답:** 첫 IK 계산의 초기 seed로 현재 관절값을 사용하기 위해서입니다. 이전 해가 생긴 뒤에는 연속 목표 계산에 직전 해를 seed로 사용할 수 있습니다.

### Q11. Damped Least Squares와 수치 Jacobian은 왜 사용하나요?

**답:** 어깨와 팔꿈치를 조금 바꿨을 때 EEF 위치가 얼마나 변하는지 수치적으로 근사해 오차를 줄이는 방향을 구합니다. 감쇠 항은 특이점이나 불안정한 계산에서 관절 변화가 지나치게 커지는 것을 줄입니다.

### Q12. Launch 파일은 Topic을 서로 연결하나요?

**답:** 아닙니다. Launch는 필요한 노드를 같은 설정으로 시작합니다. 실제 통신 연결은 노드들이 같은 Topic 이름과 호환되는 메시지 타입으로 Publisher/Subscriber를 만들 때 ROS 2가 처리합니다.

### Q13. `device_id`와 `start_joy`는 무엇인가요?

**답:** `device_id`는 joy 드라이버가 열 SDL 게임패드 번호이고, `start_joy`가 `true`이면 Launch가 `joy_node`를 실행합니다. `false`이면 외부에서 `/joy` 입력을 제공해야 합니다.

### Q14. 실습 종료 순서를 말해 보세요.

**답:** 별도 터미널에서 `/safety/torque_enable`에 `false`를 한 번 발행하고 토크 해제 로그를 확인한 다음 Launch를 종료합니다. 토크 해제 뒤 링크가 내려갈 수 있어 필요하면 받칩니다.

## 10. 자주 헷갈리는 숫자와 키워드

| 기억할 값 | 의미 |
|---|---|
| 25 Hz | 조이스틱 목표 갱신 Timer |
| 60 Hz | IK 목표 처리 Timer |
| 50 Hz | `arm_driver` 관절 상태 발행 설정 |
| 100 Hz | `arm_driver` 하드웨어 제어 설정 |
| 0.50초 | `/joy` 입력 타임아웃 |
| 0.25초 | IK 목표 최신성 제한 |
| 300 ms | arm driver 관절 목표 타임아웃 |
| 0.020 m | IK 허용 위치 오차 상한 (2 cm) |
| 5 + 1 | arm IK 관절 5개 + 별도 gripper 1개 |

외울 흐름: **게임패드 → EEF 목표 → IK 관절 목표 → 서보 → 측정 관절 상태 → FK 현재 EEF 자세**.

## 11. 빠른 실습 확인 명령

```bash
ros2 topic echo /joy
ros2 topic echo /target_pose
ros2 topic echo /joint_targets
ros2 topic echo /joint_states
ros2 topic echo /ee_pose
ros2 topic pub --once /safety/torque_enable std_msgs/msg/Bool "{data: false}"
```

실행 절차와 장비 확인 방법은 [`practice.md`](practice.md), 개념 설명은 [`lecture.md`](lecture.md), Jetson 홈 디렉터리 구성은 [`../JETSON_SETUP.md`](../JETSON_SETUP.md)를 참고한다.
