# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import List, Union, Tuple, Generator
from dataclasses import dataclass

import torch
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


@dataclass
class MotionData:

    joint_positions: Union[torch.Tensor, np.ndarray]
    joint_velocities: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_local: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_local: Union[torch.Tensor, np.ndarray]
    base_quat: Union[Rotation, torch.Tensor]
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        # Convert numpy arrays (or SciPy Rotations) to torch tensors
        def to_tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.float32)

        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = to_tensor(self.joint_positions)
        if isinstance(self.joint_velocities, np.ndarray):
            self.joint_velocities = to_tensor(self.joint_velocities)
        if isinstance(self.base_lin_velocities_mixed, np.ndarray):
            self.base_lin_velocities_mixed = to_tensor(self.base_lin_velocities_mixed)
        if isinstance(self.base_ang_velocities_mixed, np.ndarray):
            self.base_ang_velocities_mixed = to_tensor(self.base_ang_velocities_mixed)
        if isinstance(self.base_lin_velocities_local, np.ndarray):
            self.base_lin_velocities_local = to_tensor(self.base_lin_velocities_local)
        if isinstance(self.base_ang_velocities_local, np.ndarray):
            self.base_ang_velocities_local = to_tensor(self.base_ang_velocities_local)
        if isinstance(self.base_quat, Rotation):
            quat_xyzw = self.base_quat.as_quat()  # (T,4) xyzw
            # convert to wxyz
            self.base_quat = torch.tensor(
                quat_xyzw[:, [3, 0, 1, 2]],
                device=self.device,
                dtype=torch.float32,
            )

    def __len__(self) -> int:
        return self.joint_positions.shape[0]

    def get_amp_dataset_obs(self, indices: torch.Tensor) -> torch.Tensor:

        return torch.cat(
            (
                self.joint_positions[indices],
                self.joint_velocities[indices],
                self.base_lin_velocities_local[indices],
                self.base_ang_velocities_local[indices],
            ),
            dim=1,
        )

    def get_state_for_reset(self, indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:

        return (
            self.base_quat[indices],
            self.joint_positions[indices],
            self.joint_velocities[indices],
            self.base_lin_velocities_local[indices],
            self.base_ang_velocities_local[indices],
        )

    def get_random_sample_for_reset(self, items: int = 1) -> Tuple[torch.Tensor, ...]:
        indices = torch.randint(0, len(self), (items,), device=self.device)
        return self.get_state_for_reset(indices)


class AMPLoader:
   
    def __init__(
        self,
        device: str,
        dataset_path_root: Path,
        dataset_names: List[str],
        dataset_weights: List[float],
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: Union[List[str], None] = None,
    ) -> None:
        self.device = device
        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)

        # ─── Build union of all joint names if not provided ───
        if expected_joint_names is None:
            joint_union: List[str] = []
            seen = set()
            for name in dataset_names:
                p = dataset_path_root / f"{name}.npy"
                info = np.load(str(p), allow_pickle=True).item()
                for j in info["joints_list"]:
                    if j not in seen:
                        seen.add(j)
                        joint_union.append(j)
            expected_joint_names = joint_union
        # ─────────────────────────────────────────────────────────

        # Load and process each dataset into MotionData
        self.motion_data: List[MotionData] = []
        for dataset_name in dataset_names:
            dataset_path = dataset_path_root / f"{dataset_name}.npy"
            md = self.load_data(
                dataset_path,
                simulation_dt,
                slow_down_factor,
                expected_joint_names,
            )
            self.motion_data.append(md)

        # Normalize dataset-level sampling weights
        weights = torch.tensor(dataset_weights, dtype=torch.float32, device=self.device)
        self.dataset_weights = weights / weights.sum()

        # Precompute flat buffers for fast sampling
        obs_list, next_obs_list, reset_states = [], [], []
        for data, w in zip(self.motion_data, self.dataset_weights):
            T = len(data)
            idx = torch.arange(T, device=self.device)
            obs = data.get_amp_dataset_obs(idx)
            next_idx = torch.clamp(idx + 1, max=T - 1)
            next_obs = data.get_amp_dataset_obs(next_idx)

            obs_list.append(obs)
            next_obs_list.append(next_obs)

            quat, jp, jv, blv, bav = data.get_state_for_reset(idx)
            reset_states.append(torch.cat([quat, jp, jv, blv, bav], dim=1))

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)
        self.all_states = torch.cat(reset_states, dim=0)

        # Build per-frame sampling weights: weight_i / length_i
        lengths = [len(d) for d in self.motion_data]
        per_frame = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.dataset_weights, lengths)
            ]
        )
        self.per_frame_weights = per_frame / per_frame.sum()

    def _resample_data_Rn(
        self,
        data: List[np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> np.ndarray:
        f = interp1d(original_keyframes, data, axis=0)
        return f(target_keyframes)

    def _resample_data_SO3(
        self,
        raw_quaternions: List[np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> Rotation:

        # the quaternion is expected in the dataset as `xyzw` format (SciPy default)
        tmp = Rotation.from_quat(raw_quaternions)
        slerp = Slerp(original_keyframes, tmp)
        return slerp(target_keyframes)

    def _compute_raw_derivative(self, data: np.ndarray, dt: float) -> np.ndarray:
        d = (data[1:] - data[:-1]) / dt
        return np.vstack([d, d[-1:]])

    def load_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
    ) -> MotionData:

        data = np.load(str(dataset_path), allow_pickle=True).item()
        dataset_joint_names = data["joints_list"]

        # build index map for expected_joint_names
        idx_map: List[Union[int, None]] = []
        for j in expected_joint_names:
            if j in dataset_joint_names:
                idx_map.append(dataset_joint_names.index(j))
            else:
                idx_map.append(None)

        # reorder & fill joint positions
        jp_list: List[np.ndarray] = []
        for frame in data["joint_positions"]:
            arr = np.zeros((len(idx_map),), dtype=frame.dtype)
            for i, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    arr[i] = frame[src_idx]
            jp_list.append(arr)

        dt = 1.0 / data["fps"] / float(slow_down_factor)
        T = len(jp_list)
        t_orig = np.linspace(0, T * dt, T)
        T_new = int(T * dt / simulation_dt)
        t_new = np.linspace(0, T * dt, T_new)

        resampled_joint_positions = self._resample_data_Rn(jp_list, t_orig, t_new)
        resampled_joint_velocities = self._compute_raw_derivative(
            resampled_joint_positions, simulation_dt
        )

        resampled_base_positions = self._resample_data_Rn(
            data["root_position"], t_orig, t_new
        )
        resampled_base_orientations = self._resample_data_SO3(
            data["root_quaternion"], t_orig, t_new
        )

        resampled_base_lin_vel_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )
        resampled_base_lin_vel_local = np.stack(
            [
                R.as_matrix().T @ v
                for R, v in zip(
                    resampled_base_orientations, resampled_base_lin_vel_mixed
                )
            ]
        )
        zeros = np.zeros_like(resampled_base_lin_vel_mixed)

        return MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
            base_ang_velocities_mixed=zeros,
            base_lin_velocities_local=resampled_base_lin_vel_local,
            base_ang_velocities_local=zeros,
            base_quat=resampled_base_orientations,
            device=self.device,
        )

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:

        for _ in range(num_mini_batch):
            idx = torch.multinomial(
                self.per_frame_weights, mini_batch_size, replacement=True
            )
            yield self.all_obs[idx], self.all_next_obs[idx]

    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
 
        idx = torch.multinomial(
            self.per_frame_weights, number_of_samples, replacement=True
        )
        full = self.all_states[idx]
        joint_dim = self.motion_data[0].joint_positions.shape[1]

        # The dimensions of the full state are:
        #   - 4 (quat) + joint_dim (joint_positions) + joint_dim (joint_velocities)
        #   + 3 (base_lin_velocities) + 3 (base_ang_velocities)
        #   = 4 + joint_dim + joint_dim + 3 + 3
        dims = [4, joint_dim, joint_dim, 3, 3]
        return torch.split(full, dims, dim=1)
