import argparse
import os
import numpy as np
import torch

# --- Common utility imports ---
from utils.config_loader import Train_Config
from utils.train_wrapper import train_all_phases
from utils.io import load_dcemri_data, save_velocity_mat, load_DTI
from utils.velocity_guess import front_tracking_velocity
from modules.data_module import CharacteristicDomain, DCEMRIDataModule, VelocityDataModule
from modules.filtered_modules import create_outlier_filter_mask, FilteredDCEMRIDataModule

# --- Model-specific imports will be handled dynamically ---

def main(config_path):
    """
    Main driver script to run either Advection-Diffusion (AD) or Darcy-Coupled (DC) PINN models.
    """
    # --- 1. Setup and Configuration ---
    torch.set_float32_matmul_precision('medium')
    cfg = Train_Config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running with config: {config_path}")

    # Determine model type from config file name
    model_type = 'ad' if 'adpinn' in os.path.basename(config_path).lower() else 'dc'
    print(f"Detected Model Type: {model_type.upper()}")

    # --- 2. Data Preparation (Common) ---
    data, mask, pixdim, x, y, z, t = load_dcemri_data(cfg.dcemrinp_data_path)
    char_domain = CharacteristicDomain(data.shape, mask, t, pixdim, cfg.reload_dataloaders_every_n_epochs, device)
    print(f"L_star: {char_domain.L_star}, T_star: {char_domain.T_star}")

    c_dataset = DCEMRIDataModule(data, char_domain,
                               batch_size=int(mask.sum() * len(t)),
                               num_workers=cfg.dataset_num_workers, device=device)
    c_dataset.setup()
    print(f"num_train_points: {c_dataset.num_train_points}, batch_size: {c_dataset.batch_size}")

    if cfg.use_DTI:
        DTI_tensor, _ , _ = load_DTI(char_domain, cfg.dti_data_path, data.shape[:3])
    else:
        DTI_tensor = None
    char_domain.set_DTI_or_coef(DTI_tensor if cfg.use_DTI else 2.4e-4)
    print(f"Pe_g: {char_domain.Pe_g}")

    # --- 3. Initial Guesses (Common & Specific) ---
    initial_velocity_field = front_tracking_velocity(data[:, :, :, ::1], dt=t[1] - t[0])
    vel_mag = np.linalg.norm(initial_velocity_field, axis=-1)
    print(f"Initial velocity max and min: {vel_mag.max()}, {vel_mag.min()}")

    v_dataset = VelocityDataModule(initial_velocity_field, char_domain,
                                   batch_size=int(mask.sum()),
                                   num_workers=cfg.dataset_num_workers, device=device)
    v_dataset.setup()

    # --- 4. Model-Specific Preparations ---
    if model_type == 'dc':
        from modules.dc_net import AD_DC_Net
        from modules.dc_trainer import DCPINN_InitK, DCPINN_InitP, DCPINN_ADPDE_P, DCPINN_ADPDE_P_K, DCPINN_Joint
        from modules.c_trainer import CNet_Init, CNet_DenoiseInit
        from utils.permeability_guess import estimate_initial_permeability
        from modules.data_module import PermeabilityDataModule

        _, smooth_permeability = estimate_initial_permeability(data, t, ser_threshold=12.0, time_threshold_min=t[4])
        print(f"Initial permeability max and min: {smooth_permeability.max()}, {smooth_permeability.min()}")
        k_dataset = PermeabilityDataModule(smooth_permeability, char_domain,
                                           batch_size=int(mask.sum()),
                                           num_workers=cfg.dataset_num_workers, device=device)
        
        P_star = np.mean(char_domain.V_star * cfg.viscosity * char_domain.L_star / k_dataset.K_star)
        print(f"P_star: {P_star}")

        def trainer_getter(train_phase, ad_dc_net, phase_cfg={}):
            nonlocal c_dataset # Allow modification of the outer scope c_dataset
            incompressible = phase_cfg.get("incompressible", False)
            if train_phase == CNet_Init.train_phase: return CNet_Init(ad_dc_net.c_net, c_dataset.num_train_points), c_dataset
            if train_phase == CNet_DenoiseInit.train_phase: return CNet_DenoiseInit(ad_dc_net.c_net, c_dataset.num_train_points), c_dataset
            if train_phase == DCPINN_InitK.train_phase: return DCPINN_InitK(ad_dc_net, k_dataset.num_train_points), k_dataset
            if train_phase == DCPINN_InitP.train_phase: return DCPINN_InitP(ad_dc_net, v_dataset.num_train_points), v_dataset
            if train_phase == "filter":
                ad_dc_net.c_net.to(device)
                valid_mask = create_outlier_filter_mask(ad_dc_net.c_net, c_dataset.X_train, cfg.result_folder, phase_cfg.get("batch_size", 200_000))
                c_dataset = FilteredDCEMRIDataModule(c_dataset, valid_mask)
                c_dataset.setup()
                return None, c_dataset
            if train_phase == DCPINN_ADPDE_P.train_phase: return DCPINN_ADPDE_P(ad_dc_net, c_dataset.num_train_points, incompressible=incompressible), c_dataset
            if train_phase == DCPINN_ADPDE_P_K.train_phase: return DCPINN_ADPDE_P_K(ad_dc_net, c_dataset.num_train_points, incompressible=incompressible), c_dataset
            if train_phase == DCPINN_Joint.train_phase: return DCPINN_Joint(ad_dc_net, c_dataset.num_train_points, incompressible=incompressible), c_dataset
            raise ValueError(f"Unknown train_phase {train_phase}")

        net = AD_DC_Net(c_layers=[4] + [cfg.c_neuron_num] * cfg.hid_layer_num + [1],
                        k_layers=[3] + [cfg.neuron_num] * cfg.hid_layer_num + [1],
                        p_layers=[3] + [cfg.neuron_num] * cfg.hid_layer_num + [1],
                        data=data, char_domain=char_domain, C_star=c_dataset.C_star,
                        K_star=k_dataset.K_star, P_star=P_star,
                        positional_encoding=cfg.use_positional_encoding,
                        freq_nums=cfg.positional_encode_nums,
                        gamma_space=cfg.position_encode_freq_scale,
                        use_learnable_D=cfg.use_learnable_D)

    elif model_type == 'ad':
        from modules.ad_net import AD_Net
        from modules.ad_trainer import ADPINN_InitV, ADPINN_PDE_V, ADPINN_Joint
        from modules.c_trainer import CNet_Init, CNet_DenoiseInit

        def trainer_getter(train_phase, ad_net, phase_cfg={}):
            nonlocal c_dataset
            if train_phase == CNet_Init.train_phase: return CNet_Init(ad_net.c_net, c_dataset.num_train_points), c_dataset
            if train_phase == CNet_DenoiseInit.train_phase: return CNet_DenoiseInit(ad_net.c_net, c_dataset.num_train_points), c_dataset
            if train_phase == ADPINN_InitV.train_phase: return ADPINN_InitV(ad_net, v_dataset.num_train_points), v_dataset
            if train_phase == "filter":
                ad_net.c_net.to(device)
                valid_mask = create_outlier_filter_mask(ad_net.c_net, c_dataset.X_train, cfg.result_folder, phase_cfg.get("batch_size", 200_000))
                c_dataset = FilteredDCEMRIDataModule(c_dataset, valid_mask)
                c_dataset.setup()
                return None, c_dataset
            if train_phase == ADPINN_PDE_V.train_phase: return ADPINN_PDE_V(ad_net, c_dataset.num_train_points), c_dataset
            if train_phase == ADPINN_Joint.train_phase: return ADPINN_Joint(ad_net, c_dataset.num_train_points), c_dataset
            raise ValueError(f"Unknown train_phase {train_phase}")

        net = AD_Net(c_layers=[4] + [cfg.neuron_num] * cfg.hid_layer_num + [1],
                     u_layers=[3] + [cfg.neuron_num] * cfg.hid_layer_num + [3],
                     data=data, C_star=c_dataset.C_star,
                     incompressible=False, char_domain=char_domain,
                     positional_encoding=cfg.use_positional_encoding,
                     freq_nums=cfg.positional_encode_nums,
                     gamma_space=cfg.position_encode_freq_scale,
                     use_learnable_D=cfg.use_learnable_D)

    # --- 5. Training ---
    if cfg.do_training:
        pinn_model = train_all_phases(net, trainer_getter, cfg)
    else:
        print(f"Loading model from checkpoint: {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint and "train_phase" in checkpoint:
            pinn_model = trainer_getter(checkpoint.get("train_phase"), net)[0]
            pinn_model.load_state_dict(checkpoint['state_dict'], strict=True)
        else: # Fallback for older checkpoints
            pinn_model = trainer_getter("joint_data+joint_ad_pde" if model_type == 'dc' else "ad_joint", net)[0]
            net.load_state_dict(checkpoint['state_dict'], strict=False)

    # --- 6. Post-Processing and Saving Results ---
    print("Training complete. Extracting and saving results.")
    pinn_model.to(device)
    
    # Extract learned parameters
    if model_type == 'dc':
        D_learned = pinn_model.ad_dc_net.D.item()
        _, vx, vy, vz = pinn_model.ad_dc_net.v_dc_net.draw_velocity_volume()
    else: # 'ad'
        D_learned = pinn_model.ad_net.D.item()
        _, vx, vy, vz = pinn_model.v_net.draw_velocity_volume()
        
    print(f"Learned diffusivity D: {D_learned}")

    # Convert velocity to physical units (cell/min)
    vx = vx * (char_domain.V_star[0] / char_domain.pixdim[0])
    vy = vy * (char_domain.V_star[1] / char_domain.pixdim[1])
    vz = vz * (char_domain.V_star[2] / char_domain.pixdim[2])
    print(f"Velocity ranges (vx, vy, vz): ({vx.min():.3f}, {vx.max():.3f}), ({vy.min():.3f}, {vy.max():.3f}), ({vz.min():.3f}, {vz.max():.3f})")

    # Save velocity field for external analysis (e.g., MATLAB)
    save_path = f"{cfg.result_folder}/predict_velocity.mat"
    if model_type == 'dc':
        D_to_save = pinn_model.ad_dc_net.D.item() if not cfg.use_DTI else pinn_model.ad_dc_net.D_normalized.item()
    else: # 'ad'
        D_to_save = pinn_model.ad_net.D.item() if not cfg.use_DTI else pinn_model.ad_net.D_normalized.item()

    save_velocity_mat(vx, vy, vz, pixdim, D=D_to_save, use_DTI=cfg.use_DTI, path=save_path)
    print(f"Velocity and diffusivity data saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Advection-Diffusion or Darcy-Coupled PINN models.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config_path)