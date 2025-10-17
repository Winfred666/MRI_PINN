import yaml
from datetime import datetime
import os

class Train_Config:
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # output_tag should be set as name + date of yaml file
        self.result_folder = "results/" + yaml_path.split('/')[-1].replace('.yaml', '') + \
        f"_{datetime.now().strftime('%y%m%d_%H%M')}"

        # create folder if not exist
        os.makedirs(self.result_folder, exist_ok=True)

        # firstly just release all attributes in config dict, check later
        for key, value in config.items():
            setattr(self, key, value)
        
        # --- Training Control ---
        self.do_training = config.get('do_training', False)
        self.continue_training = config.get('continue_training', False)
        self.ckpt_path = config.get('ckpt_path')
        self.dcemrinp_data_path = config.get('dcemrinp_data_path')

        # --- Model Architecture ---
        self.neuron_num = config.get('neuron_num', 150)
        self.hid_layer_num = config.get('hid_layer_num', 5)
        self.c_neuron_num = config.get('c_neuron_num', 200)

        # --- Feature Engineering ---
        pe_config = config.get('positional_encoding', {})
        self.use_positional_encoding = pe_config.get('use', True)
        self.positional_encode_nums = tuple(pe_config.get('freq_nums', (10, 10, 10, 0)))
        self.position_encode_freq_scale = pe_config.get('freq_scale', 1.0)

        # --- Physics & PDE Configuration ---
        self.use_DTI = config.get('use_DTI', False)
        if self.use_DTI:
            self.dti_data_path = config.get('dti_data_path')
        
        self.use_learnable_D = config.get('use_learnable_D', False)

        # --- RBAR ---
        self.enable_rbar = config.get('enable_rbar', False)

        # --- Dataloader & Checkpointing ---
        self.reload_dataloaders_every_n_epochs = config.get('reload_dataloaders_every_n_epochs', 10)

        self.cuda_visible_devices = config.get('cuda_visible_devices', [0])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in self.cuda_visible_devices])
        
        self.ckpt_save_val_interval = config.get('ckpt_save_val_interval', 100)
        self.dataset_num_workers = config.get('dataset_num_workers', 4)

        # --- Training Phases ---
        self.phases = config.get('phases', {})

        # --- Sanity Checks ---
        if self.continue_training and not self.do_training:
            raise ValueError("'continue_training' is true, but 'do_training' is false.")
        if self.continue_training and not self.ckpt_path:
            raise ValueError("'continue_training' is true, but 'ckpt_path' is not provided.")
        if not self.do_training and not self.ckpt_path:
            raise ValueError("'do_training' is false, but 'ckpt_path' for inference is not provided.")
        if self.use_DTI and not self.dti_data_path:
            raise ValueError("'use_DTI' is true, but 'dti_data_path' is not provided.")
        
        # Also save a copy of the config file to result folder
        with open(os.path.join(self.result_folder, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
