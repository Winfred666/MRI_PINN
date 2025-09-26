import numpy as np
import matplotlib.pyplot as plt
import torch
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger


def load_dcemri_data(path):
    # 1. load training data (DCE-MRI)
    downsample_data = np.load(path)
    data = downsample_data["data"]
    mask = downsample_data["mask"]
    pixdim = downsample_data["pixdim"]
    x, y, z, t = (
        downsample_data["x"],
        downsample_data["y"],
        downsample_data["z"],
        downsample_data["t"],
    )
    print(
        data.shape, mask.shape, pixdim
    )  # (nx, ny, nz, nt), (nx, ny, nz), (dx, dy, dz)
    print(x.shape, y.shape, z.shape, t.shape)  # (nx,), (ny,), (nz,), (nt,)
    return data, mask, pixdim, x, y, z, t


class DCEMRIDataModule(L.LightningDataModule):
    def __init__(self, data, mask, t, pixdim, batch_size=1024, device="cpu"):
        super().__init__()
        self.C_star = max(data.max() / 100.0, 1e-8)  # avoid divide-by-zero
        self.data = data / self.C_star
        self.mask = mask
        self.t = t
        self.L_star = pixdim * (np.array(data.shape[:3]) - 1)
        self.T_offset = t[0]
        self.T_star = (t[-1] - t[0]) if (t[-1] - t[0]) != 0 else 1.0  # use range not absolute last
        self.pixdim = pixdim
        self.device = device
        self.batch_size = batch_size

        # Spatial points where mask == 1
        self.points = np.array(np.where(mask == 1)).T  # (num_points,3) integer indices
        self.num_points = self.points.shape[0]
        # Normalize spatial coordinates axis-wise to [0,1]
        self.points = self.points * pixdim / self.L_star

        # Normalize time to [0,1]
        self.t_normalized = (t - self.T_offset) / self.T_star

    # for testing instead of infer, we use half of data for training
    def setup(self, stage=None):
        X_train_list, C_train_list = [], []
        X_val_list, C_val_list = [], []
        # Deterministic even/odd split (consider random for production)
        for i, ti in enumerate(self.t_normalized):
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
            num_workers=0,            # safer on Windows / notebooks
            persistent_workers=False
        )

    def val_dataloader(self):
        ds = torch.utils.data.TensorDataset(self.X_val, self.C_val)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False
        )

    # also provide method to recover characteristic length and time scale to real world units
    def recover_length_time(self, L_normalized, T_normalized):
        L_real = L_normalized * self.L_star
        T_real = T_normalized * self.T_star + self.T_offset
        return L_real, T_real

    def characterise_length_time(self, L_real, T_real):
        L_normalized = L_real / self.L_star
        T_normalized = (T_real - self.T_offset) / self.T_star
        return L_normalized, T_normalized

    def get_characteristic_geodomain(self, slice_zindex=None, return_mesh=False):
        """
        Build normalized spatial grid (x,y,z) in [0,1].
        Returns:
          (N,3) torch.FloatTensor on device (default) where N = nx*ny*nz (or sliced)
        Optional:
          if return_mesh=True returns (X,Y,Z) numpy arrays (each shape (nx,ny,nz_slice))
        """
        nx, ny, nz = self.data.shape[:3]
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(0.0, 1.0, ny)
        z = np.linspace(0.0, 1.0, nz)
        if slice_zindex is not None:
            z = z[slice_zindex]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        if return_mesh:
            return X, Y, Z
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        return torch.as_tensor(grid_points, dtype=torch.float32, device=self.device)

    def get_characteristic_geotimedomain(self, slice_zindex=None, slice_tindex=None, return_mesh=False):
        """
        Build normalized spatio–temporal grid (x,y,z,t) all in [0,1].
        Returns (N,4) torch tensor unless return_mesh=True.
        """
        nx, ny, nz = self.data.shape[:3]
        x = np.linspace(0.0, 1.0, nx)
        y = np.linspace(0.0, 1.0, ny)
        z = np.linspace(0.0, 1.0, nz)
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

    def points_to_geodomain(self, points, vals):
        """
        Map scattered points (either voxel indices or normalized coords) to a (nx,ny,nz) volume.
        points: (N,3) array
        vals  : (N,) or (N,1)
        If points look normalized (max <= 1.001) they are scaled by (size-1) before indexing.
        """
        nx, ny, nz = self.data.shape[:3]
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

    def points_to_geotimedomain(self, points, vals):
        """
        Map scattered (x,y,z,t) points to a (nx,ny,nz,nt) volume.
        Accepts normalized or index coordinates (see above).
        """
        nx, ny, nz, nt = self.data.shape
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
