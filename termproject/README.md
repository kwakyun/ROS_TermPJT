# 7조 임베디드 시스템 텀프로젝트 — Advanced 단일 큐브 Pick-and-place

RL(PPO)로 무작위 시작 배치(X 0.270~0.330, Y ±0.060, yaw ±π)의 25 mm 큐브를
집기 → 들어 올리기 → 운반 → 내려놓기 → 완전 해제 → (0.390, 0.000) 50×50 mm 영역 안 2초 유지까지 수행한다.

자세한 계획·개념·실행 방법은 `../docs/` 의 세 문서를 본다.

| 문서 | 내용 |
|---|---|
| `docs/01_프로젝트_진행계획서.md` | 요구사항 대응, 시스템 설계, 실험 계획, 일정, 역할 |
| `docs/02_역할별_개념정리.md` | 역할별로 숙지할 개념, 코드 위치, 예상 질의응답 |
| `docs/03_실습_진행가이드.md` | 설치 → 검증 → 학습 → 평가 → 영상 → 실물 적용 명령 순서 |

## 폴더 구조

```text
termproject/
├── config/
│   ├── task.yaml          공통 설정 (시작·성공 조건, 제어, 보상, DR, 학습)
│   ├── exp_A_base.yaml    비교 A 기본: (sin θ, cos θ)
│   ├── exp_A_sym.yaml     비교 A 개선: (sin 4θ, cos 4θ) + 정렬 오차 + 정렬 보상
│   ├── exp_B_dr.yaml      비교 B 개선: A 개선 + Domain Randomization
│   └── eval_noise.yaml    비교 B 평가 전용 노이즈 시험 환경
├── models/                11주차 제공 로봇·작업판 모델 + pickplace_scene.xml(큐브 1개·목표 영역)
├── pickplace/             시뮬레이션·RL (ROS 불필요)
│   ├── env.py             Gymnasium 환경 (시작 조건, 성공 판정, DR)
│   ├── geometry.py        yaw 대칭 인코딩, 밑면 꼭짓점·영역 판정, SuccessTracker
│   ├── kinematics.py      FK/IK (sim·real 공용)
│   ├── controller.py      Action→관절 목표, PolicyCore (sim·real 공용)
│   ├── observation.py     관측 벡터·큐브 추정기 (sim·real 공용)
│   ├── reward.py          단계형 보상
│   ├── train.py / evaluate.py / record_video.py / compare.py
├── real/                  실물 (ROS 2 Humble)
│   ├── vision.py          색 검출·호모그래피·시차 보정
│   ├── calibrate_homography.py, tune_hsv.py
│   ├── cube_detector.py   /top_cam → /cube_obs
│   ├── policy_runner.py   /real/joint_states + /cube_obs → 정책 → /joint_targets
│   └── config/real.yaml
├── tools/                 검증 도구 (scripted_check, check_runner_parity, check_vision_sim)
├── tests/                 성공 판정 규칙 단위 시험
└── requirements.txt
```

## 빠른 실행

```bash
cd ~/termproject && source .venv/bin/activate && export PYTHONPATH=$PWD
python -m tests.test_success_rules                 # 판정 규칙
python -m tools.scripted_check                     # 환경이 과제를 수행 가능한지 (10/10 기대)
python -m pickplace.train --exp config/exp_A_sym.yaml --run A_sym_v8
python -m pickplace.evaluate --run A_sym_v8
MUJOCO_GL=egl python -m pickplace.record_video --run A_sym_v8
```

## AI 활용 표기
환경·IK·판정·학습·실물 노드 코드의 초안은 Claude(Anthropic)와 함께 작성했고,
팀이 실행·검증·수정한 내역은 보고서 '문제 해결' 절과 커밋 기록에 남긴다.
