# 7주차 총정리: YOLO Pose 지도학습과 ROS 2 검출 Topic

조교님 질문에 개념과 실제 코드 흐름을 연결해 답할 수 있도록 강의·실습 안내서·슬라이드·제공 코드를 바탕으로 정리한 요약이다.

## 1. 한 문장으로 설명하기

> Topcam 영상에서 cube1·cube2·tip의 Class·Box·Keypoint 정답을 직접 라벨링하고, 사전학습 `yolo11n-pose.pt`를 Jetson GPU에서 파인튜닝(지도학습)한다. 학습된 `vision_pose_best.pt`로 추론해 큐브는 Box 중심, tip은 예측 Keypoint를 640×480 픽셀 좌표로 `/vision/<클래스>/*` Topic에 발행한다.

## 2. 전체 흐름

```text
[데이터]  yolo_dataset_tool.py
   카메라 1920×1080 ─ Space로 고정 ─ Box/점 라벨링 ─ 640×480 저장
   └─ datasets/week7/raw/{images,labels} ─ --prepare-only ─ train/val (약 80:20) + dataset.yaml

[학습]    yolo_finetune.py
   yolo11n-pose.pt + dataset.yaml ─ GPU 파인튜닝 ─ runs/<name>/weights/best.pt
   └─ 복사: week07_vision_control/models/vision_pose_best.pt (+ .vision.json)

[확인]    yolo_pose_live.py   (ROS 없이 화면·터미널로 좌표 확인, 조교 검사)

[ROS 2]   object_detection.launch.py
   top_camera_publisher ── /top_cam (Image, 640×480, bgr8, 20 Hz)
        └─ yolo_object_detector
              ├─ /vision/<cls>/pixel       (PointStamped)
              ├─ /vision/<cls>/detected    (Bool)
              ├─ /vision/<cls>/confidence  (Float32)
              ├─ /vision/<cls>/bbox        (Float32MultiArray)
              └─ /vision/debug_image       (Image)
```

`<cls>`는 `vision.yaml`의 클래스 이름(cube1, cube2, tip)이다.

## 3. 핵심 개념

### 3.1 컴퓨터 비전의 출력 종류

| 문제 | 얻는 정보 | 이번 실습 |
|---|---|---|
| Classification | 영상 전체의 종류 | - |
| Detection | 객체별 종류 + Box | cube1·cube2·tip의 Box |
| Segmentation | 픽셀 단위 영역 | - |
| Keypoint | 정의한 점의 위치 | tip 점 |

YOLO Pose = Detection + Keypoint. 이번 Pose는 **영상의 2D 점**이지 로봇의 XYZ·Roll/Pitch/Yaw가 아니다.

### 3.2 규칙 기반 vs 학습 기반

- **규칙 기반:** 사람이 색상 범위·밝기·크기 조건을 작성한다. 고정된 환경에 유용하지만, 조명이나 대상이 바뀌면 다시 조정해야 한다.
- **학습 기반:** 이미지와 정답으로 판단 기준을 학습한다. 다양한 데이터가 중요하고, 학습하지 않은 조건에서는 실패할 수 있다.

### 3.3 세 가지 학습 방식

| 구분 | 학습 신호 | 예시 | 이번 수업 |
|---|---|---|---|
| 지도학습 | 입력 + 정답 | 사진 + 큐브 이름·Box·tip 좌표 | **직접 수행** |
| 비지도학습 | 정답 없이 데이터 구조 | 비슷한 사진끼리 군집화 | 개념만 |
| 강화학습 | 행동 후 받은 보상 | 큐브를 목표로 옮기면 높은 보상 | 이후 주차와 연결 |

군집 번호가 자동으로 cube1·cube2라는 의미를 갖지는 않는다. 군집이 무엇인지는 사람이 해석해야 한다.

### 3.4 입력·정답·예측·오차·가중치

| 요소 | 의미 |
|---|---|
| 입력 | Topcam 이미지 |
| 정답 (Label, Ground Truth) | 사람이 지정한 Class·Box·Keypoint |
| 예측 | 현재 모델이 출력한 Class·Box·Keypoint |
| 오차·손실 (Loss) | 예측과 정답의 차이 |
| 가중치 (Weights) | 입력을 예측으로 바꾸는 계산에 쓰는 학습 가능한 값 |

학습 반복: **예측 → 정답과 비교해 오차 계산 → 오차가 줄도록 가중치 갱신**
예: 정답 tip (335, 247), 예측 tip (320, 220) → 위치 오차를 줄이도록 학습한다.

- **분류:** cube1인가 cube2인가 (종류)
- **회귀:** Box·tip의 연속 좌표 (위치)
- 종류가 맞아도 Box나 점은 틀릴 수 있으므로 각각 확인한다.

### 3.5 학습 vs 추론

| 구분 | 학습 | 추론 |
|---|---|---|
| 입력 | 이미지 + 정답 라벨 | 새 이미지만 |
| 가중치 | 오차로 수정 | 고정된 학습 결과 사용 |
| 결과 | 모델 파일 (.pt) | 현재 영상의 검출 결과 |

추론에 정답을 넣지 않는다고 비지도학습이 되는 것은 아니다. **학습에 사람이 만든 정답을 사용했으므로 지도학습**이다.

### 3.6 사전학습과 파인튜닝

- **사전학습:** 다른 데이터에서 이미 학습한 가중치(`yolo11n-pose.pt`)다. 처음부터 학습하는 부담을 줄여 준다.
- **파인튜닝:** 우리 큐브 색·tip 기준·Topcam 환경의 데이터로 추가 학습한다.
- 시작 파일 `yolo11n-pose.pt`와 결과 파일 `vision_pose_best.pt`를 혼동하지 않는다.

## 4. 클래스·라벨 규칙

### 4.1 클래스 정의 (`vision.yaml`)

| ID | 클래스 | point_mode | 라벨링 | 최종 출력 점 |
|---|---|---|---|---|
| 0 | cube1 | `box_center` | 큐브 전체 Box | 검출 Box 중심 |
| 1 | cube2 | `box_center` | 큐브 전체 Box | 검출 Box 중심 |
| 2 | tip | `manual` | 도구 주변 Box + 기준점 클릭 | 예측 Keypoint |

- 클래스 ID는 **목록 순서**다.
- 한 이미지에서 클래스당 객체 1개, 객체당 Keypoint 1개다(`kpt_shape: [1, 3]`).
- 큐브 색과 tip 기준은 **촬영 전에** description에 기록한다. 현재 설정은 cube1 초록, cube2 파랑, tip은 그리퍼 끝 원형 마커 중심이다.
- 이름 규칙: 소문자로 시작하는 소문자·숫자·`_` 조합이며 중복은 안 된다(`load_config`가 검사한다).

### 4.2 좋은 라벨과 나쁜 라벨

| 상황 | 좋은 기준 | 피할 기준 |
|---|---|---|
| 큐브 Box | 큐브 전체를 일관되게 감싼다 | 일부 면만 감싸 중심이 치우친다 |
| tip 주변 Box | 점을 구별할 주변 도구 영역을 포함한다 | 점 주변의 극히 작은 영역만 매번 다르게 지정한다 |
| tip 클릭 | 모든 이미지에서 같은 물리적 점을 지정한다 | 그리퍼 중앙과 움직이는 날 끝을 섞는다 |
| 가려진 점 | 위치를 알면 클릭하고, 모르면 점 단계에서 **N** | 모르는 점을 추측해서 클릭한다 |

### 4.3 영상 크기와 좌표

- 라벨링은 1920×1080 원본에서 하고, 저장·추론은 **640×480**으로 한다. 4:3으로 **늘리는(stretch) 변환**이며 `sample_frame()`의 `cv2.resize`를 쓴다.
- 미리보기의 검정 여백(letterbox)은 화면 표시용이고 저장하지 않는다.
- 좌표계: 왼쪽 위가 (0, 0)이고 u는 오른쪽, v는 아래로 증가한다. 원본 (960, 540)은 저장 영상에서 (320, 240)이다.
- 출력은 m 단위 로봇 좌표나 모니터 좌표가 아니다. 모델 입력 `imgsz=640`과 출력 좌표계 640×480도 구분한다.

### 4.4 YOLO Pose 라벨 한 줄 (8개 값)

```text
class_id  cx  cy  w  h  kx  ky  visibility
```

- 좌표와 크기는 이미지 가로·세로로 나눈 0~1 값이다.
- visibility: `2` = 위치 지정, `0` = 위치 미상이다. 위치 미상이면 마지막 세 값이 `0 0 0`이다.
- 객체가 없으면 그 줄을 생략한다. 모두 없으면 빈 라벨 파일(배경 이미지)이 된다.
- 큐브의 kx, ky는 Box 중심과 같다. tip은 점이 Box 밖에 있으면 Box를 자동 확장한다(`expand_roi_to_include_point`).

예시 (실제 도구 출력 형식):

```text
0 0.10416667 0.18518519 0.10416667 0.18518519 0.10416667 0.18518519 2
1 0.45572917 0.43981481 0.07812500 0.13888889 0.45572917 0.43981481 2
2 0.83958333 0.76574074 0.11666667 0.23518519 0.89583333 0.87962963 2
```

## 5. 제공 코드 구조와 역할

```text
~/week7/
├── yolo_dataset_tool.py      # 촬영·라벨링·train/val 분할
├── yolo_finetune.py          # GPU 파인튜닝 + best.pt 설치
├── yolo_pose_live.py         # ROS 없이 실시간 확인
├── yolo11n-pose.pt           # 사전학습 시작 가중치
└── week07_vision_control/    # ROS 2 ament_python 패키지
    ├── config/vision.yaml
    ├── launch/object_detection.launch.py
    ├── models/               # vision_pose_best.pt + .vision.json
    └── week07_vision_control/
        ├── vision_common.py          # 공통 모듈 (ROS import 없음)
        ├── top_camera_publisher.py   # /top_cam 발행
        └── yolo_object_detector.py   # 구독 → 추론 → /vision/* 발행
```

### 5.1 `vision_common.py` 핵심 함수

| 함수 | 역할 |
|---|---|
| `load_config()` | YAML 로드와 검증(이름 규칙, point_mode, 양수, 임계값 범위). 상대 경로 기준은 `week7/` |
| `contract()` / `write_contract()` / `check_contract()` | 클래스·`kpt_shape`·영상 크기를 `.vision.json`으로 기록하고 비교한다 |
| `load_model()` | 모델 파일과 `.vision.json`을 확인한 뒤 `YOLO(task="pose")`로 로드하고 클래스와 Pose 형식을 검증한다 |
| `sample_frame()` | 라벨·추론에 공통으로 쓰는 640×480 리사이즈 |
| `open_source()` | `csi`(GStreamer `nvarguscamerasrc`), 숫자(USB), 파일 경로를 연다 |
| `extract_detections()` | 클래스별로 신뢰도가 가장 높은 **유효 후보 1개**를 선택한다 |
| `predict()` | `model.predict(conf, iou, imgsz, max_det)` 후 `extract_detections` |
| `draw_detections()` | Box와 십자점을 그린다(표시 해상도로 스케일) |

`extract_detections()` 선택 규칙:
1. 클래스별 후보를 confidence 내림차순으로 검사한다.
2. Box confidence가 `inference.confidence`(0.35) 미만이거나 Box가 비정상이면 건너뛴다.
3. `box_center`이면 점 = Box 중심이다.
4. `manual`이면 점 = Keypoint이고, **Keypoint confidence도 `keypoint_confidence`(0.35) 이상**이어야 한다.
5. 점이 640×480 밖이거나 NaN이면 건너뛴다. 처음 통과한 후보를 채택한다.

### 5.2 설정 기록 (`.vision.json`) 의 의미

```json
{ "classes": [...], "kpt_shape": [1, 3], "image_size": [640, 480] }
```

데이터셋(`datasets/week7/vision.json`)과 모델(`vision_pose_best.vision.json`)에 같은 기준을 기록한다. 수집·학습·추론에서 설정이 다르면 오류로 막는다. 클래스 이름·순서·description·점 기준·영상 크기를 바꾸면 **새 데이터셋과 모델**을 써야 한다.

### 5.3 학습 설정 (`yolo_finetune.py`)

| 항목 | 값 |
|---|---|
| device | GPU `0` (CUDA가 없으면 CPU로 자동 전환하지 않고 오류) |
| epochs / patience | 최대 100 / 20 (개선이 없으면 조기 종료) |
| batch / workers | 4 / 2 (메모리가 부족하면 `--batch 2 --workers 0`) |
| imgsz / AMP | 640 / OFF |
| seed / val 비율 | 7 / 0.20 |
| 증강 | `fliplr=0`(좌우 반전 금지), degrees 3, translate 0.05, scale 0.15, mosaic 0.25 |

- 좌우 반전을 끄는 이유: tip 같은 기준점의 좌우 의미가 바뀌면 안 되기 때문이다.
- `model.trainer.best`(실제 best.pt 경로)를 `models/vision_pose_best.pt`로 복사하고 `.vision.json`을 함께 쓴다.

### 5.4 ROS 노드

| 노드 | 동작 |
|---|---|
| `top_camera_publisher` | `publish_hz`(20 Hz) Timer로 프레임을 읽어 640×480 `bgr8`로 `/top_cam`에 발행한다. frame_id `top_camera`, QoS `qos_profile_sensor_data` |
| `yolo_object_detector` | `/top_cam` 콜백에서 추론하고 클래스별 4개 Topic과 debug_image를 발행한다. 2초마다 `Detected: ...` 로그를 남긴다 |

Launch는 Camera를 `Node`로, Detector를 `ExecuteProcess`(`python -m week07_vision_control.yolo_object_detector`)로 실행한다. 지정한 `python_executable`(venv의 python3)로 실행해야 Ultralytics와 torch를 쓸 수 있기 때문이다. 인자는 `config_path`, `model_path`, `python_executable`이다.

### 5.5 발행 Topic 규칙

| Topic | 타입 | 검출 시 | 미검출 시 |
|---|---|---|---|
| `/vision/<cls>/detected` | Bool | `true` | `false` |
| `/vision/<cls>/confidence` | Float32 | Box 신뢰도 | `0.0` |
| `/vision/<cls>/pixel` | PointStamped | (u, v, 0), stamp = 입력 영상 시각 | **발행 안 함** |
| `/vision/<cls>/bbox` | Float32MultiArray | [x1, y1, x2, y2] | **발행 안 함** |
| `/vision/debug_image` | Image | Box·점 표시 영상 | 그대로 발행 |

- z=0은 깊이를 측정했다는 뜻이 아니다.
- 추론 중 예외가 나도 `detected=false`를 발행한다.
- 영상 입력 자체가 끊기면 false 메시지도 오지 않는다. 그래서 제어에서는 **detected와 메시지 시각을 함께** 검사해야 한다.

## 6. 데이터 다양성·검증·결과 판단

- 위치(중앙·가장자리), 방향, 그리퍼 상태, 조명, 배경, 부분 가림, 일부 객체 없음, 빈 작업판을 포함한다.
- 같은 장면의 반복 촬영 수량으로 데이터가 충분하다고 판단하지 않는다.
- **Train**은 가중치 학습, **Validation**은 학습 중 성능 확인과 best 모델 선택에 쓴다. 같은 원본과 seed면 같은 분할이 나온다.
- 비슷한 연속 사진이 train과 val에 섞이면 검증 결과가 실제보다 좋아 보인다. 그래서 **학습에 쓰지 않은 새 장면**으로 따로 검사한다.
- **과적합:** 학습 장면에만 잘 맞는 현상이다. 다만 라벨 오류·가림·기준 불일치일 수도 있으니 단정하지 않는다.
- **Confidence:** 모델의 예측 점수일 뿐 정확도 %가 아니다. 임계값을 낮추면 틀린 결과도 통과한다.
- **NMS:** 같은 객체 주변의 중복 Box를 정리한다(`iou=0.45`).

| 검사 조건 | 확인 내용 |
|---|---|
| 중앙 | 세 클래스와 점이 실제 대상과 일치하는가 |
| 가장자리 | 옮겨도 같은 대상과 점을 찾는가 |
| 부분 가림 | 엉뚱하게 찾거나 점이 튀는가 |
| 조명 변화 | 색을 혼동하거나 검출이 끊기는가 |
| 빈 작업판 | 없는 객체를 검출하는가 |

Box가 맞아도 tip이 다른 부위를 가리키면 **실패**로 기록한다.

## 7. 조교님 예상 질문과 모범 답변

### Q1. 이번 실습이 지도학습인 이유는?
이미지마다 사람이 Class·Box·tip 좌표라는 정답을 붙였고, 모델이 예측과 정답의 차이를 줄이도록 가중치를 학습하기 때문입니다.

### Q2. 추론 때 정답을 안 넣는데 비지도학습 아닌가요?
아닙니다. 학습 방식은 학습 단계에서 쓴 신호로 구분합니다. 추론은 학습된 가중치로 새 이미지의 결과만 예측하는 단계입니다.

### Q3. 학습과 추론의 차이는?
학습은 이미지와 정답으로 가중치를 수정해 모델 파일을 만들고, 추론은 고정된 가중치로 새 이미지의 Box와 점을 출력합니다.

### Q4. 사전학습 모델을 왜 쓰나요?
다른 데이터에서 배운 일반적인 영상 특징을 시작점으로 써서 적은 데이터로 빠르게 학습하기 위해서입니다. 다만 우리 큐브 색과 tip 기준은 우리 데이터로 추가 학습해야 합니다.

### Q5. 큐브와 tip의 점은 어떻게 다르게 나오나요?
큐브는 검출 Box의 중심을 쓰고(`box_center`), tip은 모델이 예측한 Keypoint를 씁니다(`manual`). tip은 Box 신뢰도와 Keypoint 신뢰도를 모두 넘어야 출력됩니다.

### Q6. tip Box 중심을 tip으로 쓰면 안 되나요?
Box 중심과 팀이 정한 도구 끝점은 일치하지 않습니다. 그래서 같은 물리적 점을 직접 클릭해 Keypoint로 학습시킵니다.

### Q7. 라벨 한 줄의 8개 값은?
class_id, Box 중심 x·y, 너비·높이, keypoint x·y, visibility입니다. 좌표는 0~1로 정규화하고, visibility는 2=지정, 0=위치 미상입니다.

### Q8. 출력 좌표 (320, 240)은 무엇을 의미하나요?
640×480 저장 영상 기준 픽셀 좌표로 화면 중앙입니다. 원본 1920×1080의 (960, 540)에 대응하며 로봇의 m 단위 좌표가 아닙니다.

### Q9. Confidence 0.9면 90% 정확한가요?
아닙니다. 모델의 예측 점수일 뿐 정확도 백분율이나 성공 보장이 아닙니다.

### Q10. 검출이 잘 안 되면 임계값을 낮추면 되지 않나요?
후보는 늘지만 틀린 결과까지 통과할 수 있습니다. 라벨 기준과 데이터 다양성을 먼저 확인하고, 실패한 조건의 이미지를 보완해야 합니다.

### Q11. Train과 Validation은 왜 나누나요?
Train으로 가중치를 학습하고, 학습에 쓰지 않은 Validation으로 성능을 확인해 best 모델을 고르기 위해서입니다. 연속 사진이 섞이면 과대평가될 수 있어 새 장면 검사를 따로 합니다.

### Q12. `.vision.json`은 왜 필요한가요?
클래스 이름·순서·점 기준·영상 크기를 기록해 수집·학습·추론이 같은 기준인지 검사합니다. 기준이 다른 데이터나 모델을 섞으면 오류로 막습니다.

### Q13. 미검출이면 Topic은 어떻게 나오나요?
detected=false, confidence=0을 발행하고 pixel과 bbox는 새로 발행하지 않습니다. 마지막 pixel을 현재 위치로 쓰면 안 됩니다.

### Q14. 카메라가 끊기면 detected=false가 오나요?
오지 않을 수 있습니다. 콜백이 영상을 받아야 발행되기 때문입니다. 그래서 제어에서는 detected와 메시지 시각을 함께 검사해야 합니다.

### Q15. Launch에서 Detector를 왜 `ExecuteProcess`로 실행하나요?
Ultralytics와 torch가 설치된 venv의 python3로 실행하기 위해서입니다. `python_executable` 인자로 그 경로를 지정합니다.

### Q16. 학습할 때 좌우 반전 증강을 끈 이유는?
좌우 반전을 하면 도구 끝점의 좌우 의미가 바뀌어 tip 기준이 흔들릴 수 있기 때문입니다(`fliplr=0.0`).

### Q17. 비지도학습으로 큐브를 묶으면 cube1·cube2가 되나요?
군집 번호는 의미를 자동으로 갖지 않습니다. 각 군집이 무엇인지는 사람이 해석해야 합니다.

## 8. 자주 헷갈리는 숫자와 키워드

| 항목 | 값 |
|---|---|
| 원본 / 저장·추론 영상 | 1920×1080 / 640×480 (stretch) |
| 모델 입력 imgsz | 640 |
| 클래스 ID | cube1=0, cube2=1, tip=2 |
| kpt_shape | [1, 3] (점 1개 × (x, y, visibility)) |
| 라벨 값 개수 | 8 |
| 학습 | 100 epoch, patience 20, batch 4, workers 2, AMP OFF, GPU 0 |
| train : val | 약 80 : 20 (seed 7) |
| 추론 임계값 | confidence 0.35, keypoint 0.35, iou 0.45, max_det 20 |
| 카메라 발행 | 20 Hz, `/top_cam`, `bgr8`, Sensor Data QoS |
| 로그 주기 | 약 2초마다 `Detected: ...` |
| 시작 가중치 / 결과 모델 | `yolo11n-pose.pt` / `vision_pose_best.pt` + `.vision.json` |
| Ultralytics | 8.3.253 (torch 2.4, torchvision 0.19.0 CUDA 빌드) |

### 라벨링 키

| 키 | 실시간 화면 | 라벨링 화면 |
|---|---|---|
| Space | 영상 고정 | 단계 확정, 마지막 단계에서 저장 |
| N | - | Box: 객체 제외 / 점: 위치 미상 |
| Backspace | - | 이전 단계 |
| R | - | 현재 단계 초기화 |
| U | 최근 저장 삭제(복구 불가) | - |
| Q / Esc | 종료 + 자동 분할 | 현재 촬영 취소 |

## 9. 빠른 실습 명령

```bash
# 환경 (매 터미널)
cd ~/week7 && export WEEK7_DIR="$PWD"
source ~/ros2_base/install/setup.bash
source ~/week7/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1

# 수집·라벨링 → 분할
python3 yolo_dataset_tool.py              # CSI 번호가 다르면 --sensor-id 1, USB는 --source 0
python3 yolo_dataset_tool.py --prepare-only

# 학습
python3 yolo_finetune.py --epochs 1 --name environment_check
python3 yolo_finetune.py --name week7_pose
ls -l week07_vision_control/models/vision_pose_best.pt
cat week07_vision_control/models/vision_pose_best.vision.json

# 실시간 확인 (조교 검사)
python3 yolo_pose_live.py

# ROS 빌드·실행
colcon build --base-paths week07_vision_control --symlink-install --packages-select week07_vision_control
source install/setup.bash
ros2 launch week07_vision_control object_detection.launch.py python_executable:="$(command -v python3)"

# 다른 터미널
ros2 node list
ros2 topic info /top_cam --verbose
ros2 topic echo /vision/tip/detected
ros2 topic echo /vision/cube1/pixel
```

## 10. 문제 해결 요약

| 현상 | 확인 |
|---|---|
| CUDA False / torchvision 오류 | Jetson용 GPU 환경을 조교와 확인 |
| 카메라 열기 실패 | GStreamer YES, CSI 연결, sensor-id, 다른 프로세스 점유 |
| GUI 안 열림 | DISPLAY, OpenCV GUI 백엔드 |
| 설정 기록 불일치 | 촬영 당시의 클래스·description·크기. 새 기준이면 새 경로 사용 |
| Dataset YAML 경로 불일치 | `--prepare-only` 재실행 |
| GPU 메모리 부족 | 다른 작업 종료, `--batch 2 --workers 0` |
| 실행 파일 / Launch 없음 | 폴더 구조, 재빌드, `source install/setup.bash` |
| 학습 모델 없음 | `.pt`와 `.vision.json`을 models에 두고 재빌드 |
| 엉뚱한 점·종류 | 라벨 기준과 데이터 다양성 확인. 임계값만 낮추지 않기 |

## 11. 완료 체크리스트

- [ ] 지도학습인 이유와 학습·추론의 차이를 설명한다.
- [ ] 큐브 색과 tip 기준을 촬영 전에 기록하고 일관되게 라벨링한다.
- [ ] 이미지·라벨 쌍과 Train/Validation 구성을 확인한다.
- [ ] 시험 학습(1 epoch)과 본 학습을 완료한다.
- [ ] `vision_pose_best.pt`와 `.vision.json`을 확인한다.
- [ ] 중앙·가장자리·가림·조명·빈 작업판을 검사한다.
- [ ] 실시간 화면의 Box와 기준점을 조교에게 검사받는다.
- [ ] 실패 사례 2개(기대 결과, 실제 결과, 원인 추정, 보완할 촬영 조건)를 기록한다.
- [ ] Launch 후 pixel·detected 연속 출력과 미검출 시 false를 확인한다.
- [ ] 카메라·추론을 종료하고 데이터와 모델을 보관한다.
