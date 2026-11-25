"""세 프레임으로 YOLO26n-pose 학습·검증 한 바퀴를 실행한다."""

from wlcv.config import DEVICE
from wlcv.training.pose import train_pose


def train_pose_smoke(data_yaml, device=DEVICE):
    return train_pose(data_yaml, epochs=1, name="pose_smoke", device=device)
