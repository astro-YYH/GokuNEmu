import torch
import numpy as np
from .train_model import SimpleNN
import torch.nn as nn

act_dict = {'ReLU': nn.ReLU(), 'SiLU': nn.SiLU(), 'Tanh': nn.Tanh(), 'None': None}

class RescaledNN(SimpleNN):  # not for training, only for prediction
    def __init__(self, num_layers, hidden_size, dim_x=1, dim_y=1, activation=nn.SiLU(), center_x=None, center_y=None, device='cpu', pca_components=None, pca_mean=None, std_mean=None, std_scale=None, n_pc_zs=None):
        """
        A neural network that automatically applies zero-centering during training
        and reverses the transformation during inference.
        
        Args:
            num_layers (int): Number of hidden layers.
            hidden_size (int): Number of neurons per hidden layer.
            dim_x (int): Input feature dimension.
            dim_y (int): Output feature dimension.
            activation (nn.Module): Activation function (default: SiLU).
            center_x (torch.Tensor, optional): Mean of input data used for zero-centering.
            center_y (torch.Tensor, optional): Mean of output data used for zero-centering.
            device (str): Device for computation ('cpu' or 'cuda').
        """
        super().__init__(num_layers, hidden_size, dim_x, dim_y, activation)
        self.center_x = center_x.to(device) if center_x is not None else None
        self.center_y = center_y.to(device) if center_y is not None else None
        self.device = device  # Ensure device consistency
        self.pca_components = pca_components if pca_components is not None else None
        self.pca_mean = pca_mean if pca_mean is not None else None
        self.std_mean = std_mean.to(device) if std_mean is not None else None
        self.std_scale = std_scale.to(device) if std_scale is not None else None
        self.n_pc_zs = n_pc_zs if n_pc_zs is not None else None

    def forward(self, x):
        """
        Forward pass with input rescaling.
        """
        if self.center_x is not None:
            x = x - self.center_x  # Apply zero-centering
        y_pred = super().forward(x)  # Pass through the base network
        if self.center_y is not None:
            y_pred = y_pred + self.center_y  # Reverse zero-centering for output

        # # print shape
        # print('y_pred shape', y_pred.shape)
        # Do inverse scaling + inverse PCA if we have the info (only at inference time)
        if self.std_mean is not None and self.std_scale is not None:  # do not use this for now, we do not standardize the output in training
            y_pred = y_pred * self.std_scale + self.std_mean

        # Apply PCA inverse transformation if components and mean are provided
        # should be done for every redshift bin 
        if self.pca_components is not None and self.pca_mean is not None:
            if self.n_pc_zs is None:
                print('n_pc_zs is None, using the only PCA component')
                # convert pca_components (list of tensors) to a single tensor
                self.pca_components = torch.stack(self.pca_components, dim=0) if isinstance(self.pca_components, list) else self.pca_components
                self.pca_mean = torch.stack(self.pca_mean, dim=0) if isinstance(self.pca_mean, list) else self.pca_mean
                print('pca_components shape', self.pca_components.shape)
                y_pred = y_pred @ self.pca_components + self.pca_mean.to(self.device)  # deprecated, will be removed in the future
            else:
                # get the offset of the PCA coefficients from n_pc_zs
                offset_pc_zs = np.insert(np.cumsum(self.n_pc_zs[:-1]).astype(int), 0, 0)

                n_k = self.pca_components[0].shape[1]  # number of k bins
                n_z = len(self.n_pc_zs)
                # y_pred_new initialized to zeros in the shape of full dimension
                y_pred_new = torch.zeros(y_pred.shape[0], n_k * n_z, device=self.device)
                for i in range(n_z):
                    ik_start = i * n_k
                    ik_end = ik_start + n_k

                    # index of coefficient
                    ic_start = offset_pc_zs[i]
                    ic_end = offset_pc_zs[i] + self.n_pc_zs[i]

                    # inverse transform for this redshift bin
                    y_pred_new[:, ik_start:ik_end] = y_pred[:, ic_start:ic_end] @ self.pca_components[i] + self.pca_mean[i]

                # update y_pred
                y_pred = y_pred_new

        return y_pred
    
    # define a forward method that give the original output, without PCA transformation
    def forward_orig(self, x):
        """
        Forward pass with input rescaling, returning the original output without PCA transformation.
        """
        if self.center_x is not None:
            x = x - self.center_x  # Apply zero-centering
        y_pred = super().forward(x)  # Pass through the base network
        if self.center_y is not None:
            y_pred = y_pred + self.center_y

        return y_pred  # Return the original output without PCA transformation

    @classmethod
    def from_checkpoint(cls, path, device='cpu'):
        """
        Load a model from a saved checkpoint, including rescaling parameters.

        Args:
            path (str): Path to the saved model checkpoint.
            device (str): Device to load the model onto ('cpu' or 'cuda').

        Returns:
            RescaledNN instance
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        dim_x, dim_y = extract_input_output_dims(state_dict=checkpoint['state_dict'])

        # Activation function lookup
        activation = act_dict.get(checkpoint['activation'], nn.SiLU())  # Default to SiLU if missing

        # Convert center_x and center_y back to tensors
        # check if center_x and center_y exist
        if 'center_x' in checkpoint:
            center_x = torch.tensor(checkpoint['center_x'], dtype=torch.float32, device=device) if checkpoint['center_x'] is not None else None
            center_y = torch.tensor(checkpoint['center_y'], dtype=torch.float32, device=device) if checkpoint['center_y'] is not None else None
        else:
            center_x, center_y = None, None

        if 'mean_std' in checkpoint:
            pca_components = checkpoint['mean_std']['pca_components'] if checkpoint['mean_std']['pca_components'] is not None else None
            pca_components = [torch.tensor(pc, dtype=torch.float32, device=device) for pc in pca_components] if pca_components is not None else None
            pca_mean = checkpoint['mean_std']['pca_mean'] if checkpoint['mean_std']['pca_mean'] is not None else None
            pca_mean = [torch.tensor(pm, dtype=torch.float32, device=device) for pm in pca_mean] if pca_mean is not None else None
            std_mean =  None
            std_scale = None
            n_pc_zs = checkpoint['mean_std']['n_pc_zs'] if 'n_pc_zs' in checkpoint['mean_std'] else None
        elif 'pca_components' in checkpoint:  # backward compatibility (will be removed in the future)
            pca_components = torch.tensor(checkpoint['pca_components'], dtype=torch.float32, device=device) if checkpoint['pca_components'] is not None else None
            pca_mean = torch.tensor(checkpoint['pca_mean'], dtype=torch.float32, device=device) if checkpoint['pca_mean'] is not None else None
            std_mean = torch.tensor(checkpoint['std_mean'], dtype=torch.float32, device=device) if checkpoint['std_mean'] is not None else None
            std_scale = torch.tensor(checkpoint['std_scale'], dtype=torch.float32, device=device) if checkpoint['std_scale'] is not None else None
            n_pc_zs = None
        else:
            pca_components, pca_mean, std_mean, std_scale, n_pc_zs = None, None, None, None, None

        # Create an instance of RescaledNN
        model = cls(num_layers=checkpoint['num_layers'], hidden_size=checkpoint['hidden_size'],
                    dim_x=dim_x, dim_y=dim_y, activation=activation, center_x=center_x, center_y=center_y, device=device, pca_components=pca_components, pca_mean=pca_mean, std_mean=std_mean, std_scale=std_scale, n_pc_zs=n_pc_zs)
        
        model.load_state_dict(checkpoint['state_dict'])  # Load weights
        model.to(device)

        # load lgk
        lgk = checkpoint['lgk']

        return lgk, model


def extract_input_output_dims(state_dict):
    dim_x, dim_y = None, None
    for key in state_dict:
        if "weight" in key and "network" in key:
            if dim_x is None:
                dim_x = state_dict[key].shape[1]  # First Linear Layer (input size)
            dim_y = state_dict[key].shape[0]  # Last Linear Layer (output size)
    return dim_x, dim_y

def load_model(path, device='cpu'):
    # checkpoint = torch.load(path, weights_only=False, map_location=device)
    # dim_x, dim_y = extract_input_output_dims(state_dict=checkpoint['state_dict'])
    # # activation
    # activation = act_dict[checkpoint['activation']]
    # model = SimpleNN(num_layers=checkpoint['num_layers'], hidden_size=checkpoint['hidden_size'], dim_x=dim_x, dim_y=dim_y, activation=activation).to(device)

    # # Load the model weights
    # model.load_state_dict(checkpoint['state_dict'])
    # # load lgk
    # lgk = checkpoint['lgk']

    # center_x = checkpoint['center_x']
    # center_y = checkpoint['center_y']

    return RescaledNN.from_checkpoint(path, device=device)

    # return lgk, model

class singlefid:
    def __init__(self, path, device='cpu'):
        self.device = torch.device(device)

        # Load the saved model dictionary
        self.lgk, self.model = load_model(path, device=self.device)
        self.model.eval()

    def predict(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y = self.model(x_tensor)

        return self.lgk, y.detach().cpu().numpy()

# define doublefid: LF and HF
class doublefid:
    def __init__(self, path_LF, path_LH, device='cpu'): # k_region: 'A' or 'B', temporary solution
        self.device = torch.device(device)

        # Load the saved model dictionary
        self.lgk, self.model_LF = load_model(path_LF, device=self.device)
        _, self.model_HF = load_model(path_LH, device=self.device)
        self.model_LF.eval()
        self.model_HF.eval()
    
    def predict(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y_LF = self.model_LF(x_tensor)
        x_LH = torch.cat((x_tensor, y_LF), dim=1)
        y_HF = self.model_HF(x_LH)

        return self.lgk, y_HF.detach().cpu().numpy()
    
    def predict_LF(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y = self.model_LF(x_tensor).detach().cpu().numpy()

        return self.lgk, y
    
    # LF output in PCA space (PCs)
    def predict_LF_pc(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y_pca = self.model_LF.forward_orig(x_tensor)  # Use forward_orig to get the original output without PCA transformation

        return self.lgk, y_pca
    
    def predict_pca_in(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        # Make predictions
        _, y_LF_pca = self.predict_LF_pc(x_tensor)
        x_xy_pca = torch.cat((x_tensor, y_LF_pca), dim=1)

        y = self.model_HF(x_xy_pca)
        return self.lgk, y.detach().cpu().numpy()
    
    def predict_pca_in_LH(self, x_xy_pca):
        # Convert to tensor
        x_xy_pca_tensor = torch.tensor(x_xy_pca, dtype=torch.float32).to(self.device)
        print('x_xy_pca_tensor[0]', x_xy_pca_tensor[0])

        # Make predictions
        y = self.model_HF(x_xy_pca_tensor)

        return self.lgk, y.detach().cpu().numpy()

    
class gokunet_df_ratio:
    def __init__(self, path_LF, path_LHr, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.device = torch.device(device)

        # Load the saved model dictionary
        self.single_LF = singlefid(path_LF, device=self.device)
        self.single_LHr = singlefid(path_LHr, device=self.device)
        self.bounds = np.loadtxt(bounds_path)

    def predict_LF(self, x):
        # bounds
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        # lg to linear
        lgk, y_LF = self.single_LF.predict(x)
        # reshape y_LF(n_sample, n_k * n_z) back to (n_sample, n_z, n_k) if needed: len(lgk) = n_k < y_LF.shape[1]
        if len(lgk) < y_LF.shape[1]:
            y_LF = y_LF.reshape(y_LF.shape[0], -1, len(lgk))
        return 10**lgk, 10**y_LF
    
    def predict_LHr(self, x):
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        # lg to linear
        lgk, ratio_LH = self.single_LHr.predict(x)
        if len(lgk) < ratio_LH.shape[1]:
            ratio_LH = ratio_LH.reshape(ratio_LH.shape[0], -1, len(lgk))
        return 10**lgk, ratio_LH
    
    def predict(self, x):
        k, y_LF = self.predict_LF(x)
        _, ratio_LH = self.predict_LHr(x)

        y_H = y_LF * ratio_LH
        return k, y_H

# load L1, L2 and LF-HF models, and make predictions
class mfbox:
    def __init__(self, path_L1, path_L2, path_LH, path_LHf=None, device='cpu', stitch="XL1L2"):
        self.device = torch.device(device)

        if stitch not in ['XL1L2', 'XL', 'L']:
            raise ValueError("stitch should be one of 'XL1L2', 'XL' and 'L'")
        self.stitch = stitch

        # Load the saved model dictionary
        self.model_L1 = load_model(path_L1, device=self.device)
        self.model_L2 = load_model(path_L2, device=self.device)
        self.model_LH = load_model(path_LH, device=self.device)
        self.model_L1.eval()
        self.model_L2.eval()
        self.model_LH.eval()
        # load the final LH model if provided
        self.model_LHf = load_model(path_LHf, device=self.device) if path_LHf is not None else None

        self.lgk_L1 = np.loadtxt("./data/pre_N_L-H_stitch_z0/kf.txt") # use this for now, will be replaced later
        self.lgk_L2 = np.loadtxt("./data/pre_N_L-H_stitch_z0/kf.txt")
        self.lgk_H = np.loadtxt("./data/pre_N_L-H_stitch_z0/kf.txt")
    
    def predict(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y_L1 = self.model_L1(x_tensor)
        y_L2 = self.model_L2(x_tensor)

        # Concatenate the predictions with the input according to the stitching method
        if self.stitch == 'XL1L2':
            x_LH = torch.cat((x_tensor, y_L1, y_L2), dim=1)
        else:
            # cut and stitch L1 and L2
            middle = y_L1.shape[1] // 2
            y_L1_interp = np.array([np.interp(self.lgk_H[:middle], self.lgk_L1, y_L1[i, :].detach().numpy()) for i in range(y_L1.shape[0])])
            y_L2_interp = np.array([np.interp(self.lgk_H[middle:], self.lgk_L2, y_L2[i, :].detach().numpy()) for i in range(y_L2.shape[0])])
            # to tensor
            y_L1_interp = torch.tensor(y_L1_interp, dtype=torch.float32).to(self.device)
            y_L2_interp = torch.tensor(y_L2_interp, dtype=torch.float32).to(self.device)

            if self.stitch == 'XL':
                x_LH = torch.cat((x_tensor, y_L1_interp, y_L2_interp), dim=1)
            else: # 'L'
                x_LH = torch.cat((y_L1_interp, y_L2_interp), dim=1)

        y = self.model_LH(x_LH)

        if self.model_LHf is not None:
            x_LHf = torch.cat((x_LH, y), dim=1)
            y = self.model_LHf(x_LHf).detach().cpu().numpy()
        else:
            y = y.detach().cpu().numpy()

        return self.lgk_H, y
    
    def predict_L1(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y = self.model_L1(x_tensor).detach().cpu().numpy()

        return self.lgk_L1, y
    
    def predict_L2(self, x):
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # Make predictions
        y = self.model_L2(x_tensor).detach().cpu().numpy()

        return self.lgk_L2, y


# define gokunet model based on mfbox, normalize the input data

class gokunet(mfbox):
    def __init__(self, path_L1, path_L2, path_LH, path_LHf=None, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt", stitch="XL1L2"):
        super().__init__(path_L1, path_L2, path_LH, path_LHf, device=device, stitch=stitch)
        self.bounds = np.loadtxt(bounds_path)
    
    def predict(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y
    
    def predict_L1(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict_L1(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y
    
    def predict_L2(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict_L2(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y

    
# define gokunet_df based on doublefid
class gokunet_df(doublefid):
    def __init__(self, path_LF, path_LH, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        super().__init__(path_LF, path_LH, device=device)
        self.bounds = np.loadtxt(bounds_path)
    
    def predict(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y
    
    def predict_LF(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict_LF(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y

class gokunet_df_pca_in(doublefid):
    def __init__(self, path_LF, path_LH, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        super().__init__(path_LF, path_LH, device=device)
        self.bounds = np.loadtxt(bounds_path)
    
    def predict(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict_pca_in(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y
    
    def predict_LF(self, x):
        # Normalize the input data
        x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        lgk, y = super().predict_LF(x)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y
    
    # predict HF from LF
    def predict_LH(self, x_xy_pca):
        # Normalize the input data (only the first matched columns)
        dimx_original = self.bounds.shape[0]
        # copy the original x_xy_pca
        x_xy_pca = np.copy(x_xy_pca)
        x_xy_pca[:, :dimx_original] = (x_xy_pca[:, :dimx_original] - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        # print('x_xy_pca after normalization', x_xy_pca[0])

        lgk, y = super().predict_pca_in_LH(x_xy_pca)
        # return 10**y # Convert back to linear scale
        return 10**lgk, 10**y
    
# define gokunet-split based on gokunet_df
class gokunet_split():
    def __init__(self, path_LA, path_HA, path_LB, path_HB, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.part_A = gokunet_df(path_LA, path_HA, device=device, bounds_path=bounds_path)
        self.part_B = gokunet_df(path_LB, path_HB, device=device, bounds_path=bounds_path)

    def predict_LA(self, x):
        return self.part_A.predict_LF(x)
    
    def predict_LB(self, x):
        return self.part_B.predict_LF(x)
    
    def predict(self, x):
        k_A, y_A = self.part_A.predict(x)
        k_B, y_B = self.part_B.predict(x)
        # concatenate the results
        k = np.concatenate((k_A, k_B))
        y = np.concatenate((y_A, y_B), axis=1)
        return k, y

# class gokunet_sf
class gokunet_sf(singlefid):
    def __init__(self, path, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt", output_orig_scale=False):
        super().__init__(path, device=device)
        self.bounds = np.loadtxt(bounds_path)
        self.output_orig_scale = output_orig_scale
    
    def predict(self, x):
        # Normalize the input data
        dimx_original = self.bounds.shape[0]
        # print('before normalization',x)
        # copy the original x
        x_norm = np.copy(x)
        x_norm[:,:dimx_original] = (x_norm[:,:dimx_original] - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        # x = (x - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        # print('after normalization',x_norm)
        lgk, y = super().predict(x_norm)
        # return 10**y # Convert back to linear scale
        if self.output_orig_scale:
            print('output lg: True')
            print("one of the y values larger than 1000") if np.any(y>1000) else None
            return lgk, y
        else:
            return 10**lgk, 10**y
    
# define gokunet_alpha; A is trained separately, B is trained with range A included (L2); LH part: XL1A-HA, XL1AL2B-H(B)
class gokunet_alpha():
    def __init__(self, path_LA, path_HA, path_L2, path_LH, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.part_A = gokunet_df(path_LA, path_HA, device=device, bounds_path=bounds_path)
        self.model_L2 = gokunet_sf(path_L2, device=device, bounds_path=bounds_path)
        self.model_LH = gokunet_sf(path_LH, device=device, bounds_path=bounds_path)

    def predict_LA(self, x):
        return self.part_A.predict_LF(x)
    
    def predict_L2(self, x):
        return self.model_L2.predict(x)
    
    def predict(self, x):
        k_A, y_LA = self.part_A.predict_LF(x)
        k_A, y_A = self.part_A.predict(x)
        k_L2, y_L2 = self.model_L2.predict(x)

        lgy_LA = np.log10(y_LA) 
        lgy_L2 = np.log10(y_L2) 
        x_xL1AL2B = np.concatenate((x, lgy_LA, lgy_L2[:,lgy_L2.shape[1]//2:]), axis=1) # temporary solution 
        _, y_H_forB = self.model_LH.predict(x_xL1AL2B)
        # _, y_H_forB = self.model_LH.predict(x)
        # combine with y_A
        y = np.concatenate((y_A, y_H_forB[:, (y_H_forB.shape[1])//2:]), axis=1)
        # y = y_H_forB
        k = np.concatenate((k_A, k_L2[len(k_L2)//2:]))  # temporary solution
        return k, y
    
class gokunet_test():  # LHA ratio
    def __init__(self, path_LA, path_HA, path_L2, path_LH, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.model_LA = gokunet_sf(path_LA, device=device, bounds_path=bounds_path)
        self.model_HA = gokunet_sf(path_HA, device=device, bounds_path=bounds_path, output_orig_scale=True)
        self.model_L2 = gokunet_sf(path_L2, device=device, bounds_path=bounds_path)
        self.model_LH = gokunet_sf(path_LH, device=device, bounds_path=bounds_path, output_orig_scale=True)

    def predict_LA(self, x):
        return self.model_LA.predict(x)
    
    def predict_L2(self, x):
        return self.model_L2.predict(x)
    
    def predict(self, x):
        k_A, y_LA = self.model_LA.predict(x)
        lgk_A, ra_A = self.model_HA.predict(x)
        print('ra_A',ra_A)
        y_A = y_LA * ra_A
        k_L2, y_L2 = self.model_L2.predict(x)

        lgy_LA = np.log10(y_LA) 
        lgy_L2 = np.log10(y_L2) 
        x_xL1AL2B = np.concatenate((x, lgy_LA, lgy_L2[:,lgy_L2.shape[1]//2:]), axis=1) # temporary solution 
        _, ra_forB = self.model_LH.predict(x)
        y_H_forB = y_L2 * ra_forB
        # _, y_H_forB = self.model_LH.predict(x)
        # combine with y_A
        y = np.concatenate((y_A, y_H_forB[:, (y_H_forB.shape[1])//2:]), axis=1)
        # y = y_H_forB
        k = np.concatenate((k_A, k_L2[len(k_L2)//2:]))  # temporary solution
        return k, y
    
class gokunet_test1():  # LHA ratio
    def __init__(self, path_LA, path_HA, path_L2, path_HB, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.model_LA = gokunet_sf(path_LA, device=device, bounds_path=bounds_path)
        self.model_HA = gokunet_sf(path_HA, device=device, bounds_path=bounds_path, output_orig_scale=True)
        self.model_L2 = gokunet_sf(path_L2, device=device, bounds_path=bounds_path)
        self.model_HB = gokunet_sf(path_HB, device=device, bounds_path=bounds_path, output_orig_scale=True)

    def predict_LA(self, x):
        return self.model_LA.predict(x)
    
    def predict_L2(self, x):
        return self.model_L2.predict(x)
    
    def predict(self, x):
        k_A, y_LA = self.model_LA.predict(x)
        lgk_A, ra_A = self.model_HA.predict(x)
        print('ra_A',ra_A)
        y_A = y_LA * ra_A
        k_L2, y_L2 = self.model_L2.predict(x)

        lgy_LA = np.log10(y_LA) 
        lgy_L2 = np.log10(y_L2) 
        x_xL1AL2B = np.concatenate((x, lgy_LA, lgy_L2[:,lgy_L2.shape[1]//2:]), axis=1) # temporary solution 
        _, ra_B = self.model_HB.predict(x)
        y_B = y_L2[:, (y_L2.shape[1])//2:] * ra_B
        # _, y_H_forB = self.model_LH.predict(x)
        # combine with y_A
        y = np.concatenate((y_A, y_B), axis=1)
        # y = y_H_forB
        k = np.concatenate((k_A, k_L2[len(k_L2)//2:]))  # temporary solution
        return k, y
    
# alpha 3-step version
class gokunet_alpha3s():
    def __init__(self, path_LA, path_HAlin, path_HAnonl, path_L2, path_LHlin, path_LHnonl, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.model_LA = gokunet_sf(path_LA, device=device, bounds_path=bounds_path, output_orig_scale=True)
        self.model_HA_lin = gokunet_sf(path_HAlin, device=device, bounds_path=bounds_path, output_orig_scale=True)
        self.model_HA_nonl = gokunet_sf(path_HAnonl, device=device, bounds_path=bounds_path)
        self.model_L2 = gokunet_sf(path_L2, device=device, bounds_path=bounds_path, output_orig_scale=True)
        self.model_LH_lin = gokunet_sf(path_LHlin, device=device, bounds_path=bounds_path, output_orig_scale=True)
        self.model_LH_nonl = gokunet_sf(path_LHnonl, device=device, bounds_path=bounds_path)


    def predict_LA(self, x):
        lgk, lgy = self.model_LA.predict(x)
        return 10**lgk, 10**lgy
    
    def predict_L2(self, x):
        lgk, lgy = self.model_L2.predict(x)
        return 10**lgk, 10**lgy
    
    def predict(self, x):
        # part A
        k_A, y_LA = self.model_LA.predict(x)
        x_xLA = np.concatenate((x, y_LA), axis=1)
        _, y_HA_lin = self.model_HA_lin.predict(x_xLA)
        x_xLAlin = np.concatenate((x_xLA, y_HA_lin), axis=1)
        _, y_A = self.model_HA_nonl.predict(x_xLAlin)

        # part B
        k_L2, y_L2 = self.model_L2.predict(x)
        x_xLAB = np.concatenate((x, y_LA, y_L2[:, y_L2.shape[1]//2:]), axis=1)
        # print('x_xLAB',x_xLAB[0])
        _, y_LH_lin = self.model_LH_lin.predict(x_xLAB)
        x_xLABlin = np.concatenate((x_xLAB, y_LH_lin), axis=1)
        # print x_xLABlin all elements 
        print('x_xLABlin',x_xLABlin)
        print(x_xLABlin[0])
        _, y_H_forB = self.model_LH_nonl.predict(x_xLABlin)
        print('y_H_forB',y_H_forB)
        # concatenate the results
        k = np.concatenate((10**k_A, 10**k_L2[len(k_L2)//2:]))  # temporary solution
        y = np.concatenate((y_A, y_H_forB[:, (y_H_forB.shape[1])//2:]), axis=1)
        # y = np.concatenate((y_A, 10**y_LH_lin[:, (y_H_forB.shape[1])//2:]), axis=1)
        # y = np.concatenate((10**y_HA_lin, 10**y_LH_lin[:, (y_H_forB.shape[1])//2:]), axis=1)
        return k, y

# beta based on split
class gokunet_beta():
    def __init__(self, path_LA, path_HA, path_L2, path_HB, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.part_A = gokunet_df(path_LA, path_HA, device=device, bounds_path=bounds_path)
        self.model_L2 = gokunet_sf(path_L2, device=device, bounds_path=bounds_path)
        self.model_HB = gokunet_sf(path_HB, device=device, bounds_path=bounds_path)

    def predict_LA(self, x):
        return self.part_A.predict_LF(x)
    
    def predict_L2(self, x):
        return self.model_L2.predict(x)
    
    def predict(self, x):
        k_A, y_A = self.part_A.predict(x)
        
        k_2, y_2 = self.model_L2.predict(x)
        # print('x',x)
        lgy_2 = np.log10(y_2)  # lg values used in training
        x_xL2B = np.concatenate((x, lgy_2[:,y_2.shape[1]//2:]), axis=1) # temporary solution
        k_B, y_B = self.model_HB.predict(x_xL2B)
        # print('x_xL2B',x_xL2B)
        # concatenate the results
        k = np.concatenate((k_A, k_B))
        y = np.concatenate((y_A, y_B), axis=1)
        return k, y

# define gokunet_gamma; A is trained separately, B is trained with range A included (L2); LH part: XL1A-HA, XL2-H(B)
class gokunet_gamma():
    def __init__(self, path_LA, path_HA, path_L2, path_L2H, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.part_A = gokunet_df(path_LA, path_HA, device=device, bounds_path=bounds_path)
        self.part_B = gokunet_df(path_L2, path_L2H, device=device, bounds_path=bounds_path)

    def predict_LA(self, x):
        return self.part_A.predict_LF(x)
    
    def predict_L2(self, x):
        return self.part_B.predict_LF(x)
    
    def predict(self, x):  # combine the A result with the last half of B
        k_A, y_A = self.part_A.predict(x)
        k_B, y_B = self.part_B.predict(x)
        # concatenate the results
        k = np.concatenate((k_A, k_B[len(k_B)//2:]))  # temporary solution
        y = np.concatenate((y_A, y_B[:, len(k_B)//2:]), axis=1)
        return k, y

# LAB and LH for B
class gokunet_kappa(gokunet_gamma):
    def __init__(self, path_LA, path_HA, path_L, path_LH, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        super().__init__(path_LA, path_HA, path_L, path_LH, device=device, bounds_path=bounds_path)
    
    def predict(self, x): 
        return super().predict(x)
    
    def predict_L(self, x):
        return super().predict_L2(x)
    
    def predict_LA(self, x):
        return super().predict_LA(x)
    
# LAB for A and B, LH for B
class gokunet_lambda():
    def __init__(self, path_L, path_HA, path_LH, device='cpu', bounds_path="./data/pre_N_xL-H_stitch_z0/input_limits.txt"):
        self.model_L = gokunet_sf(path_L, device=device, bounds_path=bounds_path)
        self.model_HA = gokunet_sf(path_HA, device=device, bounds_path=bounds_path)
        self.part_B = gokunet_df(path_L, path_LH, device=device, bounds_path=bounds_path)
    
    def predict_LA(self, x):
        k_L, y_L = self.model_L.predict(x)
        k_A = k_L[:len(k_L)//2]
        y_A = y_L[:,:y_L.shape[1]//2]
        return k_A, y_A
    
    def predict_LB(self, x):
        k_L, y_L = self.model_L.predict(x)
        k_B = k_L[len(k_L)//2:]
        y_B = y_L[:,y_L.shape[1]//2:]
        return k_B, y_B
    
    def predict(self, x):
        # part A
        k_A, y_LA = self.predict_LA(x)
        x_xLA = np.concatenate((x, np.log10(y_LA)), axis=1)
        _, y_HA = self.model_HA.predict(x_xLA)

        # part B
        k_B, y_H_forB = self.part_B.predict(x)
        # concatenate the results
        k = np.concatenate((k_A, k_B[len(k_B)//2:]))  # temporary solution
        y = np.concatenate((y_HA, y_H_forB[:, (y_H_forB.shape[1])//2:]), axis=1)
        return k, y
