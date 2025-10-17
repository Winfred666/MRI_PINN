import numpy as np
import torch
import lightning as L


class CharacteristicDomain():
    # domain_shape is (x,y,z,t) shape of data
    def __init__(self, domain_shape, mask, t, pixdim, presample_epoch=100, device="cpu"):
        self.pixdim = pixdim
        self._pixdim_tensor = torch.tensor(self.pixdim, dtype=torch.float32).to(device)
        self.domain_shape = domain_shape # (nx,ny,nz,nt)
        self.mask = mask # A global mask for control volume, shape (nx,ny,nz) bool.
        self.device = device

        # Spatial normalization to [-1, 1]
        # L_star is the half-size (radius) of the domain
        self.L_star = pixdim * (np.array(domain_shape[:3]) - 1) / 2.0
        self._L_star_tensor = torch.tensor(self.L_star, dtype=torch.float32).to(device)
        # L_offset is the center of the domain in physical units
        self.L_offset = self.L_star 
        self._L_offset_tensor = torch.tensor(self.L_offset, dtype=torch.float32).to(device)

        # Temporal normalization to [-1, 1]
        self.T_star = (t[-1] - t[0]) / 2.0 if (t[-1] - t[0]) != 0 else 1.0
        self.T_offset = (t[-1] + t[0]) / 2.0
        
        # This is now a utility for the data module, not for get_characteristic_... methods
        self.t_normalized = (t - self.T_offset) / self.T_star
        self.t = t # keep original time array for reference
        self.V_star = self.L_star / self.T_star  # characteristic velocity in grid/unit time
        
        self.presample_epoch = presample_epoch

    def set_DTI_or_coef(self, DTI_or_coef):
        # if we already using DTI, then we should set Pe = 3.0 (water/tracer's diffusivity)
        self.Pe_g = self.V_star.mean() * self.L_star.mean() / (DTI_or_coef if isinstance(DTI_or_coef, float) else 3.0)  # global Peclet number
        # for better leverage 
        self.DTI_or_coef = torch.tensor(DTI_or_coef, dtype=torch.float32).to(self.device) # either a float (isotropic coeff) tensor or a (nx,ny,nz,3,3) tensor
        if not isinstance(DTI_or_coef, float):
            self.DTI_or_coef = self.DTI_or_coef.permute(3, 4, 0, 1, 2).reshape(1, 9, *self.DTI_or_coef.shape[0:3]) # shape: (1, 9, Nx, Ny, Nz)

    # X that feed in PINN is already from [-1, 1], so 
    # also provide method to recover characteristic length and time scale to real world units
    def recover_length_tensor(self, L_normalized):
        L_real = L_normalized * self._L_star_tensor + self._L_offset_tensor
        return L_real
    
    def recover_length_numpy(self, L_normalized):
        L_real = L_normalized * self.L_star + self.L_offset
        return L_real

    def recover_time_numpy(self, T_normalized):
        T_real = T_normalized * self.T_star + self.T_offset
        return T_real
    
    def recover_time_index(self, T_normalized):
        T_real = self.recover_time_numpy(T_normalized)
        # find the corresponding index in self.t
        return np.searchsorted(self.t, T_real)
    
    def characterise_length_time(self, L_real, T_real):
        L_normalized = (L_real - self.L_offset) / self.L_star
        T_normalized = (T_real - self.T_offset) / self.T_star
        return L_normalized, T_normalized

    def get_characteristic_geodomain(self, slice_zindex=None, return_mesh=False):
        """
        Build normalized spatial grid (x,y,z) in [-1,1].
        """
        nx, ny, nz = self.domain_shape[:3]
        x = np.linspace(-1.0, 1.0, nx)
        y = np.linspace(-1.0, 1.0, ny)
        z = np.linspace(-1.0, 1.0, nz)
        if slice_zindex is not None:
            z = z[slice_zindex]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        if return_mesh:
            return X, Y, Z
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        return torch.as_tensor(grid_points, dtype=torch.float32, device=self.device)

    def get_characteristic_geotimedomain(self, slice_zindex=None, slice_tindex=None, return_mesh=False):
        """
        Build normalized spatio–temporal grid (x,y,z,t) all in [-1,1].
        """
        nx, ny, nz = self.domain_shape[:3]
        x = np.linspace(-1.0, 1.0, nx)
        y = np.linspace(-1.0, 1.0, ny)
        z = np.linspace(-1.0, 1.0, nz)
        if slice_zindex is not None:
            z = z[slice_zindex]
        t = self.t_normalized
        if slice_tindex is not None:
            t = t[slice_tindex]
        X, Y, Z, T = np.meshgrid(x, y, z, t, indexing="ij")
        if return_mesh:
            return X, Y, Z, T
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), T.ravel()])
        return torch.as_tensor(grid_points, dtype=torch.float32, device=self.device)

    def points_to_geovolume(self, points, vals):
        """
        Map scattered points (either voxel indices or normalized coords) to a (nx,ny,nz) volume.
        points: (N,3) array
        vals  : (N,) or (N,1)
        If points look normalized (max <= 1.001) they are scaled by (size-1) before indexing.
        """
        nx, ny, nz = self.domain_shape[:3]
        vol = np.zeros((nx, ny, nz), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).copy()
        vals = np.asarray(vals).reshape(-1)
        if pts.max() <= 1.001: # in case represent in percentage
            pts[:, 0] *= (nx - 1)
            pts[:, 1] *= (ny - 1)
            pts[:, 2] *= (nz - 1)
        idx = np.rint(pts).astype(int)
        idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
        for (ix, iy, iz), v in zip(idx, vals):
            vol[ix, iy, iz] = v
        return vol

    def points_to_geotimevolume(self, points, vals):
        """
        Map scattered (x,y,z,t) points to a (nx,ny,nz,nt) volume.
        Accepts normalized or index coordinates (see above).
        """
        nx, ny, nz, nt = self.domain_shape
        vol = np.zeros((nx, ny, nz, nt), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).copy()
        vals = np.asarray(vals).reshape(-1)
        if pts.max() <= 1.001:
            pts[:, 0] *= (nx - 1)
            pts[:, 1] *= (ny - 1)
            pts[:, 2] *= (nz - 1)
            pts[:, 3] *= (nt - 1)
        idx = np.rint(pts).astype(int)
        idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
        idx[:, 3] = np.clip(idx[:, 3], 0, nt - 1)
        for (ix, iy, iz, it), v in zip(idx, vals):
            vol[ix, iy, iz, it] = v
        return vol

class MultiInputDataset(torch.utils.data.Dataset):
    def __init__(self, X_list, train_indices, C):
        self.X_list = X_list
        self.train_indices = train_indices
        self.C = C

    def __len__(self):
        return self.C.shape[0]

    def __getitem__(self, idx):
        # implicitly fall back to tensor if not a list of tensor
        if isinstance(self.X_list, torch.Tensor):
            return self.X_list[idx], self.train_indices[idx], self.C[idx]
        return [X[idx] for X in self.X_list], self.train_indices[idx], self.C[idx]

class MultiEpochWeightedRandomSampler(torch.utils.data.Sampler):
    """
    A WeightedRandomSampler that pre-generates indices for multiple epochs.
    This avoids the costly torch.multinomial call at the start of every epoch.
    """
    def __init__(self, weights, num_samples, replacement=True, num_epochs=100):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.num_epochs = num_epochs
        self.epoch = 0
        self.indices = []
        self._generate_indices()

    # Generate a new set of indices for all epochs
    def _generate_indices(self):
        self.indices = torch.multinomial(self.weights, self.num_samples * self.num_epochs, self.replacement).view(self.num_epochs, self.num_samples)

    def __iter__(self):
        if self.epoch >= self.num_epochs:
            # Regenerate indices if we have exhausted the pre-generated ones
            self._generate_indices()
            self.epoch = 0
        
        epoch_indices = self.indices[self.epoch].tolist()
        self.epoch += 1
        return iter(epoch_indices)

    def __len__(self):
        return self.num_samples


class RBAResampleDataModule(L.LightningDataModule):
    def __init__(self, char_domain:CharacteristicDomain,  batch_size, num_workers=0, device="cpu"):
        super().__init__()
        self.rba_model = None
        self.num_train_points = 0
        self.num_workers = num_workers
        self.device = device
        self.batch_size = batch_size

        self.char_domain = char_domain
        # also make spatial points list here:
        # Spatial points where mask == 1, for indexing
        self.points_indices = np.array(np.where(self.char_domain.mask == 1)).T  # (num_points,3) integer indices
        self.num_points = self.points_indices.shape[0]
        # Normalize spatial coordinates to [-1, 1], only for dataset X input, not for indexing.
        self.points = (self.points_indices * self.char_domain.pixdim - self.char_domain.L_offset) / self.char_domain.L_star


    # the model must be instance of Net_RBAResample
    def set_RBA_resample_model(self, rba_model):
        self.rba_model = rba_model

    def train_dataloader_with_weights(self, X_train, X_train_indice, C_train):
        # throw error if num_train_points not set
        if self.num_train_points == 0:
            raise ValueError("num_train_points must be set before calling train_dataloader_with_weights.")
    
        ds = MultiInputDataset(X_train, X_train_indice, C_train)
        
        if self.rba_model is None:
            sampler = None
            shuffle = True
        else:
            # Use the new, efficient sampler
            sampler = MultiEpochWeightedRandomSampler(
                weights=self.rba_model.get_sample_prob_weight(),
                num_samples=self.num_train_points,
                replacement=True,
                num_epochs=self.char_domain.presample_epoch # Match this to your reload_dataloaders_every_n_epochs
            )
            shuffle = False # Sampler handles shuffling

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
            sampler=sampler
        )
    def val_dataloader_simple(self, X_val, X_val_indice, C_val):
        ds = MultiInputDataset(X_val, X_val_indice, C_val)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0, # use less resources for validation
            persistent_workers=False
        )

class VelocityDataModule(RBAResampleDataModule):
    # velocity is (x,y,z,3) numpy array, unit in cell/min
    def __init__(self, velocity, char_domain, batch_size=1024, num_workers=8, device="cpu"):
        # setting spatial self.points list.
        super().__init__(char_domain, batch_size, num_workers, device)
        self.input_dim = 3  # (x,y,z)
        # convert cell/min to mm/min, then to normalized unit(so velocity should be in grid/mm)
        self.velocity = velocity * (char_domain.pixdim / char_domain.V_star)
        self.num_train_points = self.num_points

        
    def setup(self, stage=None):
        # no validation data, all for training.
        V_train = self.velocity[self.points_indices[:, 0],
                                self.points_indices[:, 1],
                                self.points_indices[:, 2]]
        self.V_train = torch.tensor(V_train, dtype=torch.float32)
        self.X_train = torch.tensor(self.points, dtype=torch.float32)
        self.X_train_indice = torch.arange(self.num_points, dtype=torch.long)
    
    def train_dataloader(self):
        return self.train_dataloader_with_weights(self.X_train, self.X_train_indice, self.V_train)


class PermeabilityDataModule(RBAResampleDataModule):
    # permeability is (x,y,z) numpy array, unit in grid^2
    def __init__(self, permeability, char_domain, batch_size=1024, num_workers=8, device="cpu"):
        # already setting spatial self.points list in super.
        super().__init__(char_domain, batch_size, num_workers, device)
        self.input_dim = 3  # (x,y,z)
        self.K_star = 1e-8  # k_0 as paper set.
        self.permeability = permeability / self.K_star
        self.num_train_points = self.num_points

    def setup(self, stage=None):
        # no validation data, all for training.
        k_train = self.permeability[self.points_indices[:, 0],
                                self.points_indices[:, 1],
                                self.points_indices[:, 2]]
        self.k_train = torch.tensor(k_train, dtype=torch.float32).reshape(-1, 1)
        self.X_train = torch.tensor(self.points, dtype=torch.float32)
        self.X_train_indice = torch.arange(self.num_points, dtype=torch.long)
    
    def train_dataloader(self):
        return self.train_dataloader_with_weights(self.X_train, self.X_train_indice, self.k_train)

# receive lightning module and sample with weight instead of uniform
class DCEMRIDataModule(RBAResampleDataModule):
    def __init__(self, data, char_domain, batch_size=1024, num_workers=8, device="cpu"):
        super().__init__(char_domain, batch_size, num_workers, device)
        # Normalize output C to [0, 100.0] as characteristic scaling. No offset needed.
        self.C_star = (data.max() if data.max() > 0 else 1.0) / 100.0

        self.input_dim = 4  # (x,y,z,t)
        self.data = data / self.C_star


    # for testing instead of infer, we use half of data for training
    def setup(self, stage=None):
        """
        Optimized setup method using vectorized operations.
        """
        print("Running optimized DCEMRIDataModule setup...")
        
        # 1. Get all time steps and create train/val splits
        all_times = self.char_domain.t_normalized
        train_indices_t = np.arange(0, len(all_times), 2)
        val_indices_t = np.arange(1, len(all_times), 2)
        
        train_times = all_times[train_indices_t]
        val_times = all_times[val_indices_t]

        # 2. Get all spatial points within the mask
        # self.points are the normalized spatial coordinates from the parent class
        num_spatial_points = self.points.shape[0]

        # 3. Create the training set
        # Tile spatial coordinates for each training time step
        X_train_spatial = np.tile(self.points, (len(train_times), 1))
        # Repeat each time value for all spatial points
        t_train_temporal = np.repeat(train_times, num_spatial_points).reshape(-1, 1)
        # Combine into a single list of tensors
        self.X_train = [
            torch.tensor(X_train_spatial, dtype=torch.float32),
            torch.tensor(t_train_temporal, dtype=torch.float32)
        ]

        # Get corresponding concentration values
        self.C_train = torch.tensor(
            self.data[:, :, :, train_indices_t][self.char_domain.mask == 1].T.flatten(),
            dtype=torch.float32
        ).reshape(-1, 1)
        
        # RBA indices are just for spatial points, so we tile them
        self.X_train_indice = torch.tile(torch.arange(num_spatial_points, dtype=torch.long), (len(train_times),))
        self.num_train_points = self.X_train_indice.shape[0]

        # 4. Create the validation set (similar to training set but with different indices)
        X_val_spatial = np.tile(self.points, (len(val_times), 1))
        t_val_temporal = np.repeat(val_times, num_spatial_points).reshape(-1, 1)
        self.X_val = [
            torch.tensor(X_val_spatial, dtype=torch.float32),
            torch.tensor(t_val_temporal, dtype=torch.float32)
        ]
        self.C_val = torch.tensor(
            self.data[:, :, :, val_indices_t][self.char_domain.mask == 1].T.flatten(),
            dtype=torch.float32
        ).reshape(-1, 1)
        
        self.X_val_indice = torch.tile(torch.arange(num_spatial_points, dtype=torch.long), (len(val_times),))

    # WARNING: compared to validation dataloader, training provide point_indice, 
    # to allow RBA weight calculation for each point
    def train_dataloader(self):
        return self.train_dataloader_with_weights(self.X_train, self.X_train_indice, self.C_train)


    def val_dataloader(self):
        return self.val_dataloader_simple(self.X_val, self.X_val_indice, self.C_val)
