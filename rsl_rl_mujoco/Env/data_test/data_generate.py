import numpy as np
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset, DataLoader
def generate_npz_files(
    output_dir,
    num_files: int = 1000,
    shape: tuple = (500, 276),
    key: str = "joint_position",
    mode: str = "mixed",  # 或 "randint", "mixed", "noisy"
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_files):
        if mode == "uniform":
            data = np.random.uniform(-10, 10, size=shape)
        elif mode == "randint":
            data = np.random.randint(-10, 10, size=shape)
        elif mode == "mixed":
            p = np.random.rand()
            if p < 0.33:
                data = np.random.uniform(-10, 10, size=shape)
            elif p < 0.66:
                data = np.random.randn(*shape) * 10
            else:
                data = np.random.randint(-50, 50, size=shape)
        else: 
            data = np.random.randn(*shape)
        
        data = data.astype(np.float32)
        file_path = output_dir / f"data_{i:02d}.npz"
        np.savez_compressed(file_path, **{key: data})
    print(f"已在 {output_dir.resolve()} 下生成 {num_files} 个 .npz 文件，模式：{mode}")

output_dir="Env/data_test"
generate_npz_files(output_dir=output_dir)