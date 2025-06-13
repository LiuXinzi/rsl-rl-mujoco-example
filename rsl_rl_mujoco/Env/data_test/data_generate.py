import numpy as np
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset, DataLoader
def generate_npz_files(
    output_dir,
    num_files: int = 5,
    shape: tuple = (500, 276),
    key: str = "joint_position",
) -> None:
    """
    在 output_dir 下生成 num_files 个 .npz 文件，
    每个文件内包含一个键为 `key`、形状为 `shape` 的随机数据。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_files):
        data = np.random.randn(*shape).astype(np.float32)
        file_path = output_dir / f"data_{i:02d}.npz"
        # 保存时使用 savez_compressed 可以减小文件体积
        np.savez_compressed(file_path, **{key: data})
    print(f"已在 {output_dir.resolve()} 下生成 {num_files} 个 .npz 文件。")

output_dir="Env/data_test"
generate_npz_files(output_dir=output_dir)