# base class for training model that has RBA resampling
import lightning as L
from modules.c_net import C_Net
import torch

# here we do not add time-dependent scaling factor, only rba resampling
class Net_RBAResample(L.LightningModule):
    # WARNING: the weight of loss is not only for gradient when do backpropagation but also for probability when resample from datamodule.
    def __init__(self, rba_name_weight_list, num_train_points,
                 rba_learning_rate=0.1, rba_memory=0.999, distribution_shapness=2.0, base_probability = 0.5, enable_rbar = True):
        super().__init__()
        
        self.num_train_points = num_train_points
        self.rba_eta = rba_learning_rate
        self.rba_gamma = rba_memory
        self.rba_nu = distribution_shapness
        self.rba_c = base_probability

        self.rba_name_weight_list = rba_name_weight_list
        # RBA weights
        for name,_ in rba_name_weight_list:
            self.register_buffer(name, torch.ones(num_train_points))
        self.automatic_optimization = False

        self.enable_rbar = enable_rbar

    # WARNING: if c_net is None, means normal rba weighted, else Time-dependent, need c_net to calculate gradient as weight.
    # pointwise_loss_list is a list of tensors, each (N, 1) shape
    def training_step(self, c_net:C_Net|None, X_train_indice, pointwise_loss_list, Xt:torch.Tensor|None, batch_idx):
        rba_weighted_loss = 0.0
        # Critical, here use c_net to compute time-dependent scaling factor(compute once per batch)
        if c_net is None:
            time_dependent_scaling = 1.0
        else:
            c_X, c_t, c_laplacian = c_net.get_c_grad_ani_diffusion(Xt)
            c_grad = torch.cat([c_X, c_t], dim=1)
            time_dependent_scaling = c_net.get_TD_RBA_scale(Xt[1], c_grad, c_laplacian)
            time_dependent_scaling = time_dependent_scaling.detach()
        
        for i, (name, weight) in enumerate(self.rba_name_weight_list):
            if self.enable_rbar:
                old_lambda = getattr(self, name) # (num_train_points, )
                # 3. Calculate RBA-weighted loss for backpropagation

                old_lambda_part = old_lambda[X_train_indice]
                # detach all factor before compute loss, to ban grad flow or extra memory usage.
                old_lambda_part = old_lambda_part.detach()

                if batch_idx == 0:
                    self.log(f'train_{name}_rba_weight_max', old_lambda_part.max())
                # Add a small epsilon to prevent division by zero
                rba_weighted_loss += weight * (old_lambda_part * pointwise_loss_list[i] / (time_dependent_scaling + 1e-8)).mean()
            else:
                rba_weighted_loss += weight * pointwise_loss_list[i].mean()
            
            if batch_idx == 0:
                self.log(f'train_{name}_loss', pointwise_loss_list[i].mean())

        # 4. Manually optimize
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(rba_weighted_loss)
        optimizer.step()

        # 5. Update RBA weights using the unweighted pointwise loss
        if self.enable_rbar:
            pointwise_loss_list = [pl.detach().squeeze() for pl in pointwise_loss_list]
            with torch.no_grad():
                for i, (name, _) in enumerate(self.rba_name_weight_list):
                    res_max = pointwise_loss_list[i].max() + 1e-8
                    old_lambda = getattr(self, name) # (num_train_points, )
                    part_new_lambda = self.rba_gamma * old_lambda[X_train_indice] + self.rba_eta * (pointwise_loss_list[i] / res_max)
                    old_lambda[X_train_indice] = part_new_lambda
                    setattr(self, name, old_lambda)

    def set_enable_rbar(self, enable):
        self.enable_rbar = enable
    # interface for weighted sampling data modules

    def get_sample_prob_weight(self):
        # first gather all the weights
        probs = torch.zeros(self.num_train_points).to(self.device)
        if self.enable_rbar:
            for name, weight in self.rba_name_weight_list:
                old_lambda = getattr(self, name)
                sharp_lambda = old_lambda ** self.rba_nu
                sharp_lambda_mean = sharp_lambda.mean()
                probs = probs + weight * (sharp_lambda / (sharp_lambda_mean + 1e-8))
        # add base probability to avoid too small sampling probability
        return probs + self.rba_c

