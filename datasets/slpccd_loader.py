from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import json


class SLPCCDDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir)
        self.split = split

        self.list_path = self.root_dir / "data" / f"{split}.txt"

        if split in ["train", "val"]:
            self.pc_dir = self.root_dir / "train_seg"
        elif split == "test":
            self.pc_dir = self.root_dir / "test_seg"
        else:
            raise ValueError(f"Unknown split: {split}")

        with open(self.list_path, "r") as f:
            self.pairs = [line.strip().split() for line in f if line.strip()]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ref_rel, query_rel = self.pairs[idx]

        ref_rel = ref_rel.replace("\\", "/")
        query_rel = query_rel.replace("\\", "/")

        ref_path = self.pc_dir / ref_rel
        query_path = self.pc_dir / query_rel

        ref = self._load_txt_pointcloud(ref_path)
        query = self._load_txt_pointcloud(query_path)

        return {
            "ref_xyz": ref[:, :3].astype(np.float32),
            "ref_rgb": ref[:, 3:6].astype(np.float32),
            "ref_label": ref[:, 6].astype(np.int64),

            "query_xyz": query[:, :3].astype(np.float32),
            "query_rgb": query[:, 3:6].astype(np.float32),
            "query_label": query[:, 6].astype(np.int64),

            "ref_path": str(ref_path),
            "query_path": str(query_path),
        }

    @staticmethod
    def _load_txt_pointcloud(path):
        with open(path, "r") as f:
            header = f.readline().strip()
            n_points = int(f.readline().strip())
            data = np.loadtxt(f, dtype=np.float32)

        if data.ndim == 1:
            data = data[None, :]

        if data.shape[0] != n_points:
            print(f"[Warning] {path}: expected {n_points}, loaded {data.shape[0]}")

        return data