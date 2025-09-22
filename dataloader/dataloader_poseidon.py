import random
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import netCDF4 as nc
from abc import ABC
from typing import Optional
import time as time_time

import random 
import shutil
import subprocess
from itertools import product

#---------------------------------------------------
# All the datasets (21 of them) are available at: _
#---------------------------------------------------

class BaseDataset(Dataset, ABC):
    """A base class for all datasets. Can be directly derived from if you have a steady/non-time dependent problem."""

    def __init__(
        self,
        which: Optional[str] = None,
        resolution: Optional[int] = None,
        in_dist: Optional[bool] = True,
        num_trajectories: Optional[int] = None,
        in_dim: Optional[int] = 2,
        out_dim: Optional[int] = 2,
        augment: Optional[bool] = False,
        data_path: Optional[str] = None,
        time_input: Optional[bool] = True,
        masked_input: Optional[list] = None,
    ) -> None:
        """
        Args:
            which: Which dataset to use, i.e. train, val, or test.
            resolution: The resolution of the dataset.
            in_dist: Whether to use in distribution or out of distribution data.
            num_trajectories: The number of trajectories to use for training.
            data_path: The path to the data files.
            time_input: Time in the input channels?
        """

        assert which in ["train", "val", "test"]
        assert resolution is not None and resolution > 0
        assert num_trajectories is not None and num_trajectories > 0
        
        #xprint(resolution, "RES")
        self.resolution = resolution
        self.in_dist = in_dist
        self.num_trajectories = num_trajectories
        self.data_path = data_path
        self.which = which
        self.time_input = time_input
        
        self.file_path = None 

        self.masked_input = masked_input
        #if self.masked_input is not None:
        #    self.mask = torch.tensor(self.masked_input, dtype=torch.float32)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.augment = augment

    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        if self.which == "train":
            self.length = self.num_trajectories
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test
            self.start = self.N_max - self.N_test
        
        return None
        
    def _rotate(self, X, Y, angle = 0):
        assert angle in [0, 90, 180, 270]
        if angle == 0:
            return X, Y
        else: 
            k = angle//90
            return torch.rot90(X, k=k, dims=[-2, -1]), torch.rot90(Y, k=k, dims=[-2, -1])
    
    def _transpose(self, X, Y, flip = 0):
        assert flip in [0, 1, 2]
        if flip == 0:
            return X, Y
        elif flip == 1:
            return torch.flip(X, dims=[-1]), torch.flip(Y, dims=[-1])
        else:
            return torch.flip(X, dims=[-2]), torch.flip(Y, dims=[-2])

    def _transform_boundary(self, X, which = 0):
        if which == 0:
            X[1:-1, 1:-1] = 0.0
        return X

    def _transform_data_policy(self, augmentations, policy, X, Y):
        assert len(augmentations) == len(policy)

        d = dict(zip(augmentations, policy))
        #print(augmentations, d)
        if "rotation" in augmentations:
            X, Y = self._rotate(X, Y, angle = d["rotation"])
        if "transpose" in augmentations:
            X, Y = self._transpose(X, Y, flip = d["transpose"])

        return X, Y
    
    def _transform_data_random(self, augmentations, X, Y):
        policy = []
        for aug in augmentations:
            if aug == "rotation":
               i = random.randint(0, 3)
               policy.append(i*90)
            if aug == "transpose":
                i = random.randint(0, 2)
                policy.append(i)
                
        return self._transform_data_policy(augmentations, policy, X, Y)


    def __len__(self) -> int:
        """
        Returns: overall length of dataset.
        """
        return self.length

    def __getitem__(self, idx) -> tuple:
        """
        Get an item. OVERWRITE!

        Args:
            idx: The index of the sample to get.

        Returns:
            A tuple of data.
        """
        pass

#--------------------------------------------------------

class BaseTimeDataset(BaseDataset, ABC):
    """A base class for time dependent problems. Inherit time-dependent problems from here."""

    def __init__(
        self,
        *args,
        max_num_time_steps: Optional[int] = None,
        time_step_size: Optional[int] = None,
        fix_input_to_time_step: Optional[int] = None,
        allowed_transitions: Optional[list] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            max_num_time_steps: The maximum number of time steps to use.
            time_step_size: The size of the time step.
            fix_input_to_time_step: If not None, fix the input to this time step.
        """
        assert max_num_time_steps is not None and max_num_time_steps > 0
        assert time_step_size is not None and time_step_size > 0
        assert fix_input_to_time_step is None or fix_input_to_time_step >= 0

        super().__init__(*args, **kwargs)
        self.max_num_time_steps = max_num_time_steps
        self.time_step_size = time_step_size
        self.fix_input_to_time_step = fix_input_to_time_step
        self.allowed_transitions = allowed_transitions
        
        
    def post_init(self) -> None:
        """
        Call after self.N_max, self.N_val, self.N_test, as well as the file_paths and normalization constants are set.
        self.max_time_step must have already been set.
        """
        assert (
            self.N_max is not None
            and self.N_max > 0
            and self.N_max >= self.N_val + self.N_test
        )
        #assert self.num_trajectories + self.N_val + self.N_test <= self.N_max
        assert self.N_val is not None and self.N_val > 0
        assert self.N_test is not None and self.N_test > 0
        assert self.max_num_time_steps is not None and self.max_num_time_steps > 0

        if self.fix_input_to_time_step is not None:
            assert (
                self.fix_input_to_time_step + self.max_num_time_steps
                <= self.max_num_time_steps
            )

            self.multiplier = self.max_num_time_steps
        else:
            if self.allowed_transitions is None:
                self.time_indices = []
                i = 0
                for j in range(i, self.max_num_time_steps + 1):
                    self.time_indices.append((self.time_step_size * i, self.time_step_size * j))
            else:
                self.time_indices = []
                for i in range(self.max_num_time_steps+1):
                    for j in range(i, self.max_num_time_steps + 1):
                        if (j-i) in self.allowed_transitions:
                            self.time_indices.append((self.time_step_size * i, self.time_step_size * j))
            
            self.multiplier = len(self.time_indices)
            print("time_indices", self.time_indices)
        
        if self.which == "train":
            self.length = self.num_trajectories * self.multiplier
            self.start = 0
        elif self.which == "val":
            self.length = self.N_val * self.multiplier
            self.start = self.N_max - self.N_val - self.N_test
        else:
            self.length = self.N_test * self.multiplier
            self.start = self.N_max - self.N_test

#--------------------------------------------------------
# Navier-Stokes Datasets:
#--------------------------------------------------------

class NavierStokes2dTimeDataset(BaseTimeDataset):
    def __init__(self, 
                *args,
                rel_time: bool = True,
                is_time: bool = True, 
                **kwargs):
        super().__init__(*args, **kwargs)
        assert self.max_num_time_steps * self.time_step_size <= 20

        self.rel_time = rel_time
        self.is_time = is_time

        self.N_max = 20000
        self.N_val = 25
        self.N_test = 240

        if self.masked_input is None:
            self.mean = torch.tensor([0.0, 0.0], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor([0.391, 0.356], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
        else:
            self.mask = torch.tensor([0.,1.,1.,0.] + (self.in_dim - 4)*[0.], dtype=torch.float32)
            self.mean = torch.tensor([0.80, 0.0,   0.0,   0.0] + (self.in_dim - 4)*[0.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)
            self.std = torch.tensor( [0.31, 0.391, 0.356, 0.46] + (self.in_dim - 4)*[1.], dtype=torch.float32).unsqueeze(1).unsqueeze(1)

        if self.augment:
            self.augmentations = ["rotation", "transpose"]

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
        
        if self.rel_time:
            t = t2 - t1
        else:
            t = t1
        
        if self.is_time:
            time = float(t / 20.0)
            if time <= 0.:
                time = 1e-6
        else:
            time = None

        inputs = torch.tensor(self.reader['u'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        inputs = torch.cat((inputs, torch.tensor(self.reader['v'][i + self.start, t1]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        label = torch.tensor(self.reader['u'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)
        label = torch.cat((label, torch.tensor(self.reader['v'][i + self.start, t2]).type(torch.float32).reshape(1, self.resolution, self.resolution)))

        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)

            for i in range(4, self.in_dim):
                inputs_zeros = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
                inputs = torch.cat((inputs, inputs_zeros), 0)
                if i < self.out_dim:
                    label = torch.cat((label, inputs_zeros), 0)

        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std

        if self.augment:
            inputs, label = self._transform_data_random(self.augmentations, inputs, label)

        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label


class BrownianBridgeTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_path = '/path_to_IEU_2D_BB/'

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

class SinesTestDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_path = '/path_to_ns_sin_test_dataset.np.npy/'
        self.data = np.load(self.file_path)
        self.length = self.N_test
        self.start = self.N_test

        self.N_max = 240
        self.length = self.N_test
        self.start = 0

    def __getitem__(self, idx):
        time = 14./20
        
        inputs = (
            torch.from_numpy(self.data[0,idx+self.start])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.data[1,idx+self.start])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        
        return time, inputs, label

class SinesTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_path = '/path_to_velocity_16.nc/'
        self.N_max = 1168
        self.N_val = 25
        self.N_test = 240

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
        
        if self.rel_time:
            t = t2 - t1
        else:
            t = t1
        
        if self.is_time:
            time = float(t / 20.0)
            if time <= 0.:
                time = 1e-6
        else:
            time = None
        
        inputs = (
            torch.from_numpy(self.reader["velocity"][i + self.start, t1][:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["velocity"][i + self.start, t2][:2])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        
        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)
        
        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std

        if time is None:
            return inputs, label

        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label

class SinesEasyTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 5000
        self.N_val = 25
        self.N_test = 240

        self.file_path = "/path_to_sin_easy.nc"

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

    def __getitem__(self, idx):
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier

        if self.fix_input_to_time_step is None:
            t1, t2 = self.time_indices[_idx]
            assert t2 >= t1
        else:
            t1 = self.fix_input_to_time_step
            t2 = self.time_step_size * (_idx + 1)
        
        if self.rel_time:
            t = t2 - t1
        else:
            t = t1
        
        if self.is_time:
            time = float(t / 20.0)
            if time <= 0.:
                time = 1e-6
        else:
            time = None
        
        inputs = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][t1, :])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        label = (
            torch.from_numpy(self.reader["sample_" + str(i + self.start)][t2, :])
            .type(torch.float32)
            .reshape(2, self.resolution, self.resolution)
        )
        
        if self.masked_input is not None:
            inputs_rho = torch.ones((1, self.resolution, self.resolution)).type(torch.float32)
            inputs_p   = torch.zeros((1, self.resolution, self.resolution)).type(torch.float32)
            inputs = torch.cat((inputs_rho, inputs), 0)
            inputs = torch.cat((inputs, inputs_p), 0)
            
            label = torch.cat((inputs_rho, label), 0)
            label = torch.cat((label, inputs_p), 0)
        
        inputs = (inputs - self.mean) / self.std
        label = (label - self.mean) / self.std

        if time is None:
            return inputs, label

        if self.masked_input is not None:
            return time, inputs, label, self.mask
        else:
            return time, inputs, label


class PiecewiseConstantsTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.file_path = '/path_to_IEU_2D_PWC.nc'

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

class GaussiansTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.file_path = '/path_to_IEU_2D_Gauss.nc'

        
        self.reader = h5py.File(self.file_path, "r")
        self.post_init()

class ComplicatedShearLayerTimeDataset(NavierStokes2dTimeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N_max = 20000
        self.file_path = '/path_to_IEU_2D_DDSLTracer.nc'

        self.reader = h5py.File(self.file_path, "r")
        self.post_init()
    