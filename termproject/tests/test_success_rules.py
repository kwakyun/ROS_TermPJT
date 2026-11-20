"""성공 판정 규칙 단위 시험 — 밀기·잡은 채 맞추기·낙하가 성공으로 잡히지 않는지 확인.

실행: python -m tests.test_success_rules
"""
from pickplace.geometry import StepFacts, SuccessTracker, align_error, fold_quarter

DT = 0.05


def facts(grasp=False, floor=True, clear=0.0, robot=False, open_=False, margin=-0.01):
    return StepFacts(grasp, floor, clear, robot, open_, margin)


def run(seq):
    tr = SuccessTracker(dt=DT, hold_time=2.0, lift_clearance=0.002)
    t = 0.0
    for f, n in seq:
        for _ in range(n):
            t += DT
            tr.update(f, t)
    return tr


def test_full_success():
    tr = run([(facts(grasp=True, robot=True), 5),
              (facts(grasp=True, floor=False, clear=0.03, robot=True), 20),
              (facts(grasp=True, floor=True, robot=True, margin=0.008), 3),
              (facts(open_=True, margin=0.008), 41)])
    assert tr.success, tr.events


def test_push_only_fails():
    # 로봇이 큐브를 밀어서 목표 안에 넣고 떨어짐 (한 번도 들지 않음)
    tr = run([(facts(robot=True, margin=0.008), 30), (facts(open_=True, margin=0.008), 60)])
    assert not tr.success and tr.max_stage == 0


def test_hold_without_release_fails():
    tr = run([(facts(grasp=True, robot=True), 5),
              (facts(grasp=True, floor=False, clear=0.03, robot=True), 20),
              (facts(grasp=True, floor=True, robot=True, margin=0.008), 100)])
    assert not tr.success and tr.stage == 3


def test_hold_resets_when_outside():
    tr = run([(facts(grasp=True, robot=True), 5),
              (facts(grasp=True, floor=False, clear=0.03, robot=True), 20),
              (facts(grasp=True, floor=True, robot=True, margin=0.008), 3),
              (facts(open_=True, margin=0.008), 30),
              (facts(open_=True, margin=-0.001), 1),
              (facts(open_=True, margin=0.008), 30)])
    assert not tr.success and tr.stage == 4


def test_drop_during_carry():
    tr = run([(facts(grasp=True, robot=True), 5),
              (facts(grasp=True, floor=False, clear=0.03, robot=True), 10),
              (facts(floor=True, margin=0.008, open_=True), 60)])
    assert not tr.success and tr.drops == 1 and tr.end_reason(False, True) == "dropped"


def test_yaw_folding():
    import math
    assert abs(fold_quarter(math.radians(100)) - math.radians(10)) < 1e-9
    assert abs(align_error(math.radians(90), 0.0)) < 1e-9          # 90° 돌아간 큐브는 정렬 오차 0
    assert abs(align_error(math.radians(44), 0.0) - math.radians(44)) < 1e-9
    assert abs(align_error(math.radians(46), 0.0) + math.radians(44)) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
