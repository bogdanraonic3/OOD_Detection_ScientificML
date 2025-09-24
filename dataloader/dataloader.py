from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import os
import netCDF4 as nc
import scipy.ndimage
import time

class CIFAR10_Dataset(Dataset):
    def __init__(self,  
                which: str,
                N_samples: int,
                ood_share: float,
                is_diffusion: bool = False) -> None:
        
        assert which in ['train', 'val', 'test']

        self.is_diffusion = is_diffusion
        self.add_noise = False

        if which == "train":
            assert N_samples<=45500
            self.N_id = N_samples
            self.N_per_class = N_samples//9
            self.N_ood = int(self.N_per_class * ood_share)
            self.start_index_id = 0
            self.start_index_ood = - self.N_id

            if self.is_diffusion:
                self.add_noise = True
                print(" ")
                print("ADDING NOISE TO SAMPLES")
                print(" ")

        elif which == "val":
            self.N_id = 9*500
            self.N_per_class = 500
            self.N_ood = int(self.N_per_class * ood_share)
            self.start_index_id = 45000 - self.N_id
            self.start_index_ood = 5000 - self.N_id - self.N_per_class

        else:
            self.N_id = 900
            self.N_ood = 100
            self.start_index_id = 0
            self.start_index_ood = -self.N_id

        self.N_total = self.N_id + self.N_ood

        if which in ["train", "val"]:
            self.data_id = np.load("/cpath_to_ID_classes_train").astype("float32")
            self.data_ood = np.load("/path_to_OOD_class_train/").astype("float32")
        else:
            self.data_id = np.load("/path_to_ID_classes_test/").astype("float32")
            self.data_ood = np.load("/path_to_OOD_class_test/").astype("float32")

    def __len__(self):
        return self.N_total
    
    def __getitem__(self, index):
        if index < self.N_id:
            data = (self.data_id[index + self.start_index_id, :, :, :3]/255-0.5)/0.5
            label = self.data_id[index + self.start_index_id, :, :, -1]

            data = torch.from_numpy(data).type(torch.float32)
            data = torch.permute(data, (2, 0, 1))
            label = torch.from_numpy(label).type(torch.int64)[0,0]

        else:
            data = (self.data_ood[index + self.start_index_ood, :, :, :3]/255-0.5)/0.5
            label = self.data_ood[index + self.start_index_ood, :, :, -1]

            data = torch.from_numpy(data).type(torch.float32)
            data = torch.permute(data, (2, 0, 1))
            label = torch.from_numpy(label).type(torch.int64)[0,0]

        if self.is_diffusion:

            shape = (1,) + data.shape[1:]
            label = (label/9.0) * torch.ones(shape, device = data.device).type(torch.float32)

            if self.add_noise:
                
                num_pixels = data.numel()
                num_replace = int(0.2 * num_pixels)
                flat_data = data.reshape(-1)
                indices = torch.randperm(num_pixels)[:num_replace]
                noise = 0.2*torch.randn(num_replace, dtype=data.dtype, device=data.device)
                flat_data[indices] = noise
                data = flat_data.view_as(data)

                num_pixels = label.numel()
                num_replace = int(0.2 * num_pixels)
                flat_label = label.reshape(-1)
                indices = torch.randperm(num_pixels)[:num_replace]
                noise = 0.2*torch.randn(num_replace, dtype=label.dtype, device=label.device)
                flat_label[indices] = noise
                label = flat_label.view_as(label)

                image_noise = 0.05*torch.randn(data.shape, device = data.device).type(torch.float32)
                label_noise = 0.05*torch.randn(label.shape, device = label.device).type(torch.float32)
                data = data + image_noise
                label = label + label_noise


        return data, label

class MNSIT_Dataset(Dataset):
    def __init__(self,  
                which: str,
                N_samples: int,
                ood_share: float,
                is_diffusion: bool = False) -> None:
        
        assert which in ['train', 'val', 'test']

        self.is_diffusion = is_diffusion

        if which == "train":
            assert N_samples<=7641
            self.N_id = N_samples
            self.N_per_class = N_samples//9
            self.N_ood = int(self.N_per_class * ood_share)
            self.start_index_id = 0
            self.start_index_ood = - self.N_id

        elif which == "val":
            self.N_id = 9*50
            self.N_per_class = 50
            self.N_ood = int(self.N_per_class * ood_share)
            self.start_index_id = 8991- self.N_id
            self.start_index_ood = 1000 - self.N_id - 50 - 100

        else:
            self.N_id = 9 * 100
            self.N_ood = 100
            self.start_index_id = 8991 - 9*50 - 450
            self.start_index_ood = 1000 - 100 - 900

        self.N_total = self.N_id + self.N_ood

        self.data_id = np.load("/path_to_ID_classes_train/").astype("float32")
        self.label_id = np.load("/path_to_OOD_classes_train/").astype("float32")
        self.data_ood = np.load("/path_to_ID_classes_test/").astype("float32")
        self.label_ood = np.load("/path_to_OOD_classes_test/").astype("float32")
        
        self.mean = 0.13
        self.std  = 0.31

    def __len__(self):
        return self.N_total
    
    def __getitem__(self, index):
        if index < self.N_id:
            data = (self.data_id[index + self.start_index_id] - self.mean)/self.std
            label = np.array([self.label_id[index + self.start_index_id]])
        else:
            data = (self.data_ood[index + self.start_index_ood] - self.mean)/self.std
            label = np.array([self.label_ood[index + self.start_index_ood]])

        data = torch.from_numpy(data).type(torch.float32)
        label = torch.from_numpy(label).type(torch.int64)
        
        if self.is_diffusion:
            shape = (1,) + data.shape[1:]
            label = (label/9.0) * torch.ones(shape, device = data.device).type(torch.float32)
        
        return data, label

class BinaryClassificationBaseline(Dataset):
    def __init__(self,
                folder: str,
                which: str,
                N_samples: int,
                N_max: int = 1024,
                mean: float = 0.0,
                std: float  = 1.0,
                only_x: bool = False,
                start_idx:int = -1) -> None:
        
        assert which in ['train', 'val', "test"]

        self.N_max = N_max

        self.num_datasets = len(folder)
        self.labels = []
        self.folder = folder

        self.is_file_load = True
        for f in folder:
            self.labels.append(np.load(f"{f}/classification_labels.npy"))
            if not os.path.exists(f"{f}/inputs.npy"):
                self.is_file_load = False
        

        if self.is_file_load:
            self.data_inp = []
            self.data_out = []
            for f in folder:
                self.data_inp.append(np.load(f"{f}/inputs.npy"))
                self.data_out.append(np.load(f"{f}/preds.npy"))

        self.N_val_per_dataset = N_samples//4

        if which == "train":
            self.start = 0
            self.N_total = N_samples * self.num_datasets
        elif which == "val":
            self.start = self.N_max - self.N_val_per_dataset
            self.N_total = self.N_val_per_dataset * self.num_datasets #self.N_val  * self.num_datasets
        elif which == "test":
            self.start = start_idx
            self.N_total = (self.N_max - self.start) * self.num_datasets

        self.which = which
        self.mean = mean
        self.std  = std
        self.only_x = only_x
        
    def __len__(self):
        return self.N_total
    
    def __getitem__(self, index):
        
        idx_dataset = index % self.num_datasets
        idx = index//self.num_datasets + self.start
        folder = self.folder[idx_dataset]

        if self.is_file_load:
            data  = torch.from_numpy(self.data_inp[idx_dataset][idx]).type(torch.float32)
            label = torch.from_numpy(self.data_out[idx_dataset][idx]).type(torch.float32)
        else:
            data  = torch.from_numpy(np.load(f"{folder}/sample_{idx}_inp.npy")).type(torch.float32)
            label = torch.from_numpy(np.load(f"{folder}/sample_{idx}_pred.npy")).type(torch.float32)
        data = (data - self.mean)/self.std
        label = (label - self.mean)/self.std

        if not self.only_x:
            data = torch.cat((data, label), 0)

        return data, torch.tensor(self.labels[idx_dataset][idx]).type(torch.int64)

class Wave2d_Dataset(Dataset):
    def __init__(self,  
                which: str,
                N_samples: int,
                is_ood: int = None,
                is_wide: bool = True,
                is_time: bool = False,
                use_generated: str = None) -> None:
        
        assert which in ['train', 'val', 'test']

        if which == "train":
            assert N_samples<=10000
            self.N = N_samples
            self.start_index = 0

        elif which == "val":
            self.N = 500
            self.start_index = 9500

        else:
            self.start_index = 0
            self.N = N_samples
        
        if is_ood is None:
            self.file = "/path_to_train_data/"
            if which == "test":
                self.file = "/path_to_standing_wave_test_ID_data/"
        else:
            
            tag = f"_{is_ood}"
            self.file = f"/path_to_standing_test_OOD_data/"
        self.data = np.load(self.file).astype("float32")

        if which in ["train", "test"]:
            print("Data file is: " + self.file)
            print(" ")

        self.std_data = 0.821028
        self.std_label = 0.57195

        self.is_time = is_time
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        
        data = self.data[0, index+self.start_index]/self.std_data
        label = self.data[1, index+self.start_index]/self.std_label
        
        data = torch.from_numpy(data).type(torch.float32).reshape(1,128,128)
        label = torch.from_numpy(label).type(torch.float32).reshape(1,128,128)

        time = 1.0
        if not self.is_time:
            time = None
        
        return time, data, label

    def load_details(self):
        self.Ks = np.load("/path_to_standing_wave_OOD_3_WIDE_2d_detailed/Ks.npy").astype("float32")
        self.coeffs = np.load("/path_to_standing_wave_OOD_3_WIDE_2d_detailed/coeffs.npy").astype("float32")
        self.decays = np.load("/path_to_standing_wave_OOD_3_WIDE_2d_detailed/decays.npy").astype("float32")

        return self.Ks, self.coeffs, self.decays

class Merra2Dataset(Dataset):
    def __init__(self,  
                which: str,
                N_samples: int,
                years: list = [2016,2017,2018,2019,2020,2021],
                max_allowed_transition: int = 15,
                is_time: bool = True,
                is_ood: bool = False,
                is_ood_spatial: int = None,
                in_dim: int = 1,
                out_dim: int = 1,
                is_dense: bool = True) -> None:

        self.max_allowed_transition = max_allowed_transition
        self.which = which
        self.years = years

        if self.which == "train":

            if is_dense:
                postfix = "_alltime"
                self.dt = 1/24.
            else:
                postfix = ""
                self.dt = 1/6.

            self.N = N_samples
            data_path = [f"/path_to_humidity_train_from01to04_{y}{postfix}.nc" for y in years]
            
        elif self.which == "val":
            self.N = 700
            self.max_allowed_transition = 6
            self.dt = 1/6.
            data_path = "/path_to_humidity_val_from01to04_2022.nc"
            self.years = [2022]
        else:
            
            data_path = "/path_to_humidity_test_spatial_south_america_from01to04_2023_alltime.nc"
            if is_ood_spatial == 2:
                #australia and oceania
                data_path = "/path_to_humidity_test_spatial_australia_from01to04_2023_alltime.nc"
            elif is_ood_spatial == 3:
                #africa
                data_path = "/path_to_humidity_test_spatial_africa_from01to04_2023_alltime.nc"
            elif is_ood_spatial == 4:
                #asia
                data_path = "/path_to_humidity_test_spatial_asia_from01to04_2023_alltime.nc"

            self.dt = self.max_allowed_transition / 24.
            self.N = min(int(118/self.dt), N_samples)
            self.years = [2023]
            print("is_ood", is_ood, "is_ood_spatial", is_ood_spatial, "N", self.N)

        self.allowed_transitions = np.arange(1, self.max_allowed_transition + 1)
        if self.which == "test":
            self.allowed_transitions = [1]

        self.multiplier = len(self.allowed_transitions)
        self.N_switch = self.multiplier * self.N #After how many IO pairs should we switch to the next dataset?
        self.N_total  = self.multiplier * self.N * len(self.years) #Total number of samples

        self.in_dim = 1
        self.out_dim = 1
        self.is_time = is_time
        
        self.data = []
        self.mean = 0.012170596
        self.std = 0.0043237656

        if self.which == "train":
            for file in data_path:
                reader = h5py.File(file, "r")
                self.data.append((reader["data"][..., 0, 0]- self.mean)/self.std)
                reader.close()
        else:
            reader = h5py.File(data_path, "r")
            if self.which == "val":
                self.data = [(reader["data"][..., 0, 0]- self.mean)/self.std]
            else:
                self.data = [(reader["data"][:,::self.max_allowed_transition,:, :, 0, 0]- self.mean)/self.std]
            
    def __len__(self):
        return self.N_total
    
    def __getitem__(self, index):

        data_idx = index//self.N_switch
        idx = index%self.N_switch
        i = idx // self.multiplier
        _idx = idx - i * self.multiplier
        time = self.dt * self.allowed_transitions[_idx]
        
        data = torch.from_numpy(self.data[data_idx][:,i]).type(torch.float32).reshape(self.in_dim,128,128)
        label = torch.from_numpy(self.data[data_idx][:,i + self.allowed_transitions[_idx]]).type(torch.float32).reshape(self.out_dim,128,128)

        if not self.is_time:
            time = None

        return time, data, label


'''
    Segmentation Loaders:
'''

class BrainDataset(Dataset):
    def __init__(self, nc_path, which, transform=None, is_ood = False):
        """
        Args:
            nc_path (str): Path to the .nc file
            indices (array-like): Sample indices for subset (train/val/test)
            transform (Albumentations transform): Optional transforms
        """

        self.is_ood = is_ood
        self.slices = np.arange(30,130,1)
        self.multiplier = len(self.slices)

        self.N_max = 210
        self.N_val = 10
        self.N_test = 10
        
        if which == "train":
            self.start = 0
            self.N = (self.N_max - self.N_val - self.N_test) * len(self.slices)
            self.n_traj = self.N_max - self.N_test - self.N_val
        elif which == "val":
            self.start = self.N_max - self.N_val - self.N_test
            self.N = self.N_val * len(self.slices)
            self.n_traj = self.N_val
        else:
            self.start = self.N_max - self.N_val
            self.N = self.N_test * len(self.slices)
            self.n_traj = self.N_test

            if is_ood:
                self.start = 0
                self.N = 10 * len(self.slices)
                self.n_traj = 10

        self.reader = nc.Dataset(nc_path, "r")
        self.transform = transform

        self.images = self.reader.variables['images'][self.start:self.start+self.n_traj,:,:, 30:130:1, 0]
        self.masks = self.reader.variables['masks'][self.start:self.start+self.n_traj,:,:, 30:130:1, 0]
        self.reader.close()

        print("-----------")
        print("FILES READ")
        print("-----------")
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):

        sample_idx = idx // self.multiplier
        sample_slice = idx % self.multiplier
        
        image = self.images[sample_idx,:,:,sample_slice]
        mask = self.masks[sample_idx,:,:,sample_slice]

        image = np.rot90(image)
        mask = np.rot90(mask)
        mask[mask>0.5] = 1.0
        mask[mask<=0.5] = 0.0

        # Apply Albumentations transform if provided
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        image = np.expand_dims(image, axis=0)  # (1, H, W)
        image = torch.from_numpy(image).type(torch.float32)

        image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
        mask = np.expand_dims(mask, axis=0)  # (1, H, W)
        mask = torch.from_numpy(mask).type(torch.float32)

        image_noise = 0.025*torch.randn(image.shape, device = image.device).type(torch.float32)
        image[image<0.01] = image_noise[image<0.01]

        mask_noise = 0.025*torch.randn(mask.shape, device = mask.device).type(torch.float32)
        mask[mask<0.01] = mask_noise[mask<0.01]

        return image, mask