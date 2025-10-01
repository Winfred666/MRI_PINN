import numpy as np
import torch
import lightning as L


class CharacteristicDomain():
    # domain_shape is (x,y,z,t) shape of data
    def __init__(self, domain_shape, t, pixdim, device="cpu",D_mean = 1e-4):
        self.pixdim = pixdim
        self._pixdim_tensor = torch.tensor(self.pixdim, dtype=torch.float32).to(device)
        self.domain_shape = domain_shape # (nx,ny,nz,nt)
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

        self.V_star = self.L_star / self.T_star  # characteristic velocity in grid/unit time

        self.Pe_g = self.V_star.mean() * self.L_star.mean() / D_mean  # global Peclet number
    
    # also provide method to recover characteristic length and time scale to real world units
    def recover_length_tensor(self, L_normalized):
        L_real = L_normalized * self._L_star_tensor + self._L_offset_tensor
        return L_real

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


class VelocityDataModule(L.LightningDataModule):
    # velocity is (x,y,z,3) numpy array, unit in grid/min
    def __init__(self, velocity, char_domain, mask, batch_size=1024, device="cpu"):
        super().__init__()
        self.input_dim = 3  # (x,y,z)
        # convert to normalized unit
        self.velocity = velocity * (char_domain.pixdim / char_domain.V_star)
        self.batch_size = batch_size
        self.device = device

        self.points = np.array(np.where(mask == 1)).T  # (num_points,3) integer indices
        self.num_points = self.points.shape[0]
        self.points = self.points * char_domain.pixdim / char_domain.L_star  # normalize to [0,1]

    def setup(self, stage=None):
        # no validation data, all for training.
        V_train = self.velocity[self.points[:, 0].astype(int),
                                self.points[:, 1].astype(int),
                                self.points[:, 2].astype(int)]
        self.V_train = torch.tensor(V_train, dtype=torch.float32)
        self.X_train = torch.tensor(self.points, dtype=torch.float32)
    
    def train_dataloader(self):
        ds = torch.utils.data.TensorDataset(self.X_train, self.V_train)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,          # for safety of notebook and widget
            persistent_workers=True
        )
        

class DCEMRIDataModule(L.LightningDataModule):
    def __init__(self, data, char_domain, mask, batch_size=1024, device="cpu"):
        super().__init__()
        # Normalize output C to [0, 100.0] as characteristic scaling. No offset needed.
        self.C_star = (data.max() if data.max() > 0 else 1.0) / 100.0
        self.input_dim = 4  # (x,y,z,t)
        self.char_domain = char_domain
        self.data = data / self.C_star

        self.mask = mask
        self.device = device
        self.batch_size = batch_size

        # Spatial points where mask == 1
        self.points = np.array(np.where(mask == 1)).T  # (num_points,3) integer indices
        self.num_points = self.points.shape[0]
        # Normalize spatial coordinates to [-1, 1]
        self.points = (self.points * self.char_domain.pixdim - self.char_domain.L_offset) / self.char_domain.L_star

    # for testing instead of infer, we use half of data for training
    def setup(self, stage=None):
        X_train_list, C_train_list = [], []
        X_val_list, C_val_list = [], []
        # Deterministic even/odd split (consider random for production)
        for i, ti in enumerate(self.char_domain.t_normalized):
            Ci = self.data[:, :, :, i][self.mask == 1]
            ti_array = np.full((self.num_points, 1), ti, dtype=np.float32)
            Xi = np.hstack((self.points, ti_array))
            if i % 2 == 0:
                X_train_list.append(Xi)
                C_train_list.append(Ci.reshape(-1, 1))
            else:
                X_val_list.append(Xi)
                C_val_list.append(Ci.reshape(-1, 1))
        self.X_train = torch.tensor(np.vstack(X_train_list), dtype=torch.float32)
        self.C_train = torch.tensor(np.vstack(C_train_list), dtype=torch.float32)
        self.X_val = torch.tensor(np.vstack(X_val_list), dtype=torch.float32)
        self.C_val = torch.tensor(np.vstack(C_val_list), dtype=torch.float32)

    def train_dataloader(self):
        ds = torch.utils.data.TensorDataset(self.X_train, self.C_train)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,          # for safety of notebook and widget
            persistent_workers=True
        )

    def val_dataloader(self):
        ds = torch.utils.data.TensorDataset(self.X_val, self.C_val)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            persistent_workers=True
        )