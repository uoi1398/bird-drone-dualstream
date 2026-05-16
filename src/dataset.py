%%writefile src/dataset.py
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from degradation import degrade_frame
from optical_flow import frames_to_flow_rgb


class BirdDroneVideoDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split="train",
        mode="rgb",
        num_frames=16,
        image_size=224,
        blur=False,
        degrade_level=None
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.mode = mode
        self.num_frames = num_frames
        self.image_size = image_size

        # 兼容旧写法：blur=True 等价于 strong degradation
        if degrade_level is None:
            self.degrade_level = "strong" if blur else "none"
        else:
            self.degrade_level = degrade_level

        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.flow_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.df)

    def _read_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            raise RuntimeError(f"Cannot read video: {video_path}")

        indices = np.linspace(0, total - 1, self.num_frames).astype(int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()

            if not ok:
                continue

            # OpenCV 读进来是 BGR，这里转成 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 如果设置了降质，就调用 degradation.py 里的 degrade_frame
            if self.degrade_level != "none":
                frame = degrade_frame(frame, level=self.degrade_level)

            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames extracted: {video_path}")

        # 如果视频太短，不够 num_frames，就重复最后一帧补齐
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames[:self.num_frames]

    def _make_rgb_tensor(self, frames):
        tensors = [self.rgb_transform(f) for f in frames]
        return torch.stack(tensors, dim=0)

    def _make_flow_tensor(self, frames):
        # frames_to_flow_rgb 会把 RGB frames 转成 optical flow RGB maps
        flow_frames = frames_to_flow_rgb(frames)

        # 光流图数量通常是 num_frames - 1
        tensors = [self.flow_transform(f) for f in flow_frames]

        return torch.stack(tensors, dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = self._read_frames(row["path"])

        if self.mode == "rgb":
            x = self._make_rgb_tensor(frames)
        elif self.mode == "flow":
            x = self._make_flow_tensor(frames)
        else:
            raise ValueError("mode must be 'rgb' or 'flow'")

        y = torch.tensor(row["label"], dtype=torch.long)
        video_id = row["video_id"]

        return x, y, video_id