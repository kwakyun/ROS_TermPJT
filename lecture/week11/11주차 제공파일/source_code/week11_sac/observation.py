"""Direct joint angles and XYZ positions for the Week 11 policy."""
import numpy as np
from gymnasium import spaces

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
OBSERVATION_SIZE = 15


class ObservationBuilder:
    """[six joint angles, moving fingertip XYZ, green XYZ, blue XYZ]."""
    def __init__(self):
        self.space = spaces.Box(
            -np.inf, np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

    def build(self, *, joint_qpos, moving_tip_position, object_position, target_position):
        return np.concatenate((
            joint_qpos,
            moving_tip_position,
            object_position,
            target_position,
        )).astype(np.float32)
