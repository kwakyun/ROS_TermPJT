"""Distance costs plus contact success and failure rewards."""

def week11_reward(
    tip_object_distance, object_target_distance, success, failure, common, config
):
    # TODO 1 — 두 XYZ 거리의 합에 config['distance_weight']를 곱해 음수 거리 비용을 계산한다.
    # 성공·실패에 common의 보상을 적용하여 terms 사전에 담는다.
    # sum()으로 총 보상을 구하고 (총 보상, terms)를 반환한다.
    distance_cost = -float(config["distance_weight"]) * (
        float(tip_object_distance) + float(object_target_distance)
    )
    terms = {
        "distance": distance_cost,
        "success": float(common["success_bonus"]) if success else 0.0,
        "failure": -float(common["failure_penalty"]) if failure else 0.0,
    }
    return sum(terms.values()), terms


def compute_transition_reward(transition, common, config):
    return week11_reward(
        transition["tip_object_distance"],
        transition["object_target_distance"],
        transition["success"],
        transition["failure"],
        common,
        config,
    )
