# import sys
import numpy as np
from .mfbox import gokunet_df_ratio
from scipy.interpolate import interp1d
from importlib.resources import files

def sigmoid_ramp(x, x_start, x_end, sharpness=10):
    """Smooth transition from 0 to 1 over [x_start, x_end] using a sigmoid."""
    x_mid = 0.5 * (x_start + x_end)
    scale = sharpness / (x_end - x_start)
    return 1 / (1 + np.exp(-scale * (x - x_mid)))

def blend_predictions(Pk1_com, Pk2_com, weights):
    return Pk1_com * (1 - weights) + Pk2_com * weights


class MatterPowerEmulator:
    """
    Emulator class to predict the matter power spectrum P(k, z) using two neural networks.
    Combines two models at different k-ranges using a redshift-dependent transition point.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize emulator by loading neural network models and preparing redshift bins and stitching indices.
        """
        # Redshift bin setup
        lna = np.linspace(0, -np.log(1 + 3), 30)
        zs_uniform = 1 / np.exp(lna) - 1
        zs_manual = np.array([0, 0.2, 0.5, 1, 2, 3])
        self.z_bins = np.unique(np.concatenate((zs_uniform, zs_manual)))

        # Load pretrained neural networks
        bounds_path = "input_limits-W.txt"
        self.emu1 = gokunet_df_ratio(
            path_LF=str(files("gokunemu").joinpath("models/L1A/best_model.pth")),
            path_LHr=str(files("gokunemu").joinpath("models/L1HAr/best_model.pth")),
            bounds_path=str(files("gokunemu").joinpath(bounds_path)),
            device=device
        )
        self.emu2 = gokunet_df_ratio(
            path_LF=str(files("gokunemu").joinpath("models/L2/best_model.pth")),
            path_LHr=str(files("gokunemu").joinpath("models/L2Hr/best_model.pth")),
            bounds_path=str(files("gokunemu").joinpath(bounds_path)),
            device=device
        )

        # Sample dummy input to extract k1 and k2 structure
        dummy_params = np.zeros((1, 10))
        k1, _ = self.emu1.predict(dummy_params)
        k2, _ = self.emu2.predict(dummy_params)

        # i_1_cut and i_2_cut
        self.i_1_cut = np.where(k1 <= k2.min())[0][-1]
        self.i_2_cut = np.where(k2 > k1.max())[0][0]
        k_com = k1[self.i_1_cut:]

        self.k = np.concatenate((k1, k2[self.i_2_cut:]))

        self.weights = sigmoid_ramp(k_com, k_com[0], k_com[-1], sharpness=4)

        # load P+F correction factor
        pf_correction_file = "pplusf_0195.npy"
        self.alpha_pf = np.load(str(files("gokunemu").joinpath(pf_correction_file)))

    def _expand_params(self, cosmo_params, Om, Ob, hubble, As, ns, w0, wa, mnu, Neff, alphas):
        if cosmo_params is None:
            return np.array([[Om, Ob, hubble, As, ns, w0, wa, mnu, Neff, alphas]])
        return np.atleast_2d(cosmo_params)


    def predict(
        self,
        cosmo_params: np.ndarray = None,
        Om: float = 0.3,
        Ob: float = 0.05,
        hubble: float = 0.7,
        As: float = 2.1e-9,
        ns: float = 0.96,
        w0: float = -1.0,
        wa: float = 0.0,
        mnu: float = 0.06,
        Neff: float = 3.044,
        alphas: float = 0.0,
        redshifts: np.ndarray = np.array([0.])
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts the matter power spectrum for given cosmological parameters and redshifts.
        Uses precomputed kAB and stitching indices to speed up repeated inference.

        Returns:
        - kAB: Combined k-array
        - Pk_obj: Array of P(k, z) predictions, shape (n_samples, len(redshifts), len(kAB))
        """
        # Expand cosmological parameters to 2D array if needed
        cosmo_params = self._expand_params(cosmo_params, Om, Ob, hubble, As, ns, w0, wa, mnu, Neff, alphas)

        n_samples = cosmo_params.shape[0]

        # Predict full redshift grid with both emulators
        _, Pk1 = self.emu1.predict(cosmo_params)
        _, Pk2 = self.emu2.predict(cosmo_params)

        # blend two predictions using sigmoidal transition
        Pk1_com = Pk1[:, :, self.i_1_cut:]
        Pk2_com = Pk2[:, :, :self.i_2_cut]
        Pk_blend = blend_predictions(Pk1_com, Pk2_com, self.weights)
        Pk = np.concatenate((Pk1[:, :, :self.i_1_cut], Pk_blend, Pk2[:, :, self.i_2_cut:]), axis=2)

        # apply P+F correction
        Pk *= self.alpha_pf

        # Interpolate log10 P(k) in redshift space to target redshifts
        log_Pk = np.log10(Pk)
        log_Pk_interp = np.array([
            interp1d(self.z_bins, log_Pk[i], kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')(redshifts)
            for i in range(n_samples)
        ])

        return self.k, 10 ** log_Pk_interp


