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
    Emulator class to predict the matter power spectrum P(k, z).
    Routes inputs between narrow- and wide-box model families and blends their L1/L2 ranges.
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

        # Load the wide- and narrow-box nonlinear emulators. Inputs in the
        # Goku-N box use N; all other inputs use W.
        self._nonlinear_emulators = {
            "W": self._load_nonlinear_emulator("W", sharpness=4, device=device),
            "N": self._load_nonlinear_emulator("N", sharpness=6, device=device),
        }
        self._n_bounds = np.loadtxt(str(files("gokunemu").joinpath("input_limits-N.txt")))

        # Preserve the existing internal attributes for compatibility.
        self.emu1 = self._nonlinear_emulators["W"]["emu1"]
        self.emu2 = self._nonlinear_emulators["W"]["emu2"]
        self.weights = self._nonlinear_emulators["W"]["weights"]
        self.k = self._nonlinear_emulators["W"]["k"]
        self.i_1_cut = self._nonlinear_emulators["W"]["i_1_cut"]
        self.i_2_cut = self._nonlinear_emulators["W"]["i_2_cut"]

        if not np.array_equal(self.k, self._nonlinear_emulators["N"]["k"]):
            raise ValueError("GokuNEmu-W and GokuNEmu-N must use the same output k grid")

        bounds_path = "input_limits-W.txt"
        self.emu_lin = gokunet_df_ratio(
            path_LF=str(files("gokunemu").joinpath("models/W_L2_linear/best_model.pth")),
            path_LHr=str(files("gokunemu").joinpath("models/W_L2Hr/best_model.pth")), # not used
            bounds_path=str(files("gokunemu").joinpath(bounds_path)),
            device=device
        )

        # load P+F correction factor
        pf_correction_file = "pplusf_0195.npy"
        self.alpha_pf = np.load(str(files("gokunemu").joinpath(pf_correction_file)))

    def _load_nonlinear_emulator(self, family, sharpness, device):
        """Load one model family and prepare its L1/L2 stitching metadata."""
        bounds_path = str(files("gokunemu").joinpath(f"input_limits-{family}.txt"))
        emu1 = gokunet_df_ratio(
            path_LF=str(files("gokunemu").joinpath(f"models/{family}_L1A/best_model.pth")),
            path_LHr=str(files("gokunemu").joinpath(f"models/{family}_L1HAr/best_model.pth")),
            bounds_path=bounds_path,
            device=device
        )
        emu2 = gokunet_df_ratio(
            path_LF=str(files("gokunemu").joinpath(f"models/{family}_L2/best_model.pth")),
            path_LHr=str(files("gokunemu").joinpath(f"models/{family}_L2Hr/best_model.pth")),
            bounds_path=bounds_path,
            device=device
        )

        dummy_params = np.zeros((1, 10))
        k1, _ = emu1.predict(dummy_params)
        k2, _ = emu2.predict(dummy_params)
        i_1_cut = np.where(k1 <= k2.min())[0][-1]
        i_2_cut = np.where(k2 > k1.max())[0][0]
        k_com = k1[i_1_cut:]

        return {
            "emu1": emu1,
            "emu2": emu2,
            "i_1_cut": i_1_cut,
            "i_2_cut": i_2_cut,
            "k": np.concatenate((k1, k2[i_2_cut:])),
            "weights": sigmoid_ramp(k_com, k_com[0], k_com[-1], sharpness=sharpness),
        }

    def _predict_nonlinear(self, emulator, cosmo_params):
        """Predict and blend the L1/L2 spectra for one emulator family."""
        _, Pk1 = emulator["emu1"].predict(cosmo_params)
        _, Pk2 = emulator["emu2"].predict(cosmo_params)
        i_1_cut = emulator["i_1_cut"]
        i_2_cut = emulator["i_2_cut"]
        Pk_blend = blend_predictions(
            Pk1[:, :, i_1_cut:],
            Pk2[:, :, :i_2_cut],
            emulator["weights"],
        )
        return np.concatenate((Pk1[:, :, :i_1_cut], Pk_blend, Pk2[:, :, i_2_cut:]), axis=2)

    def _expand_params(self, cosmo_params, Om, Ob, hubble, As, ns, w0, wa, mnu, Neff, alphas):
        if cosmo_params is None:
            return np.array([[Om, Ob, hubble, As, ns, w0, wa, mnu, Neff, alphas]])
        return np.atleast_2d(cosmo_params)


    def get_matter_power(
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

        # Route each input independently: N inside the Goku-N box, W otherwise.
        use_n = np.all(
            (cosmo_params >= self._n_bounds[:, 0]) &
            (cosmo_params <= self._n_bounds[:, 1]),
            axis=1,
        )
        Pk = None
        for family, mask in (("N", use_n), ("W", ~use_n)):
            if not np.any(mask):
                continue
            family_prediction = self._predict_nonlinear(
                self._nonlinear_emulators[family], cosmo_params[mask]
            )
            if Pk is None:
                Pk = np.empty(
                    (n_samples, *family_prediction.shape[1:]),
                    dtype=family_prediction.dtype,
                )
            Pk[mask] = family_prediction

        # apply P+F correction
        Pk *= self.alpha_pf

        # Interpolate log10 P(k) in redshift space to target redshifts
        log_Pk = np.log10(Pk)
        log_Pk_interp = np.array([
            interp1d(self.z_bins, log_Pk[i], kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')(redshifts)
            for i in range(n_samples)
        ])

        return self.k, 10 ** log_Pk_interp

    def get_matter_power_lin(
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
        Predicts the linear matter power spectrum for given cosmological parameters and redshifts.

        Returns:
        - k_lin: linear theory k-array
        - Pk_obj: Array of P(k, z) predictions, shape (n_samples, len(redshifts), len(k_lin))
        """
        # Expand cosmological parameters to 2D array if needed
        cosmo_params = self._expand_params(cosmo_params, Om, Ob, hubble, As, ns, w0, wa, mnu, Neff, alphas)

        n_samples = cosmo_params.shape[0]

        # Predict full redshift grid
        k_lin, Pk = self.emu_lin.predict_LF(cosmo_params)

        # keep only k > 0.001 h/Mpc (large errors at very large scales)
        mask = k_lin > 0.001
        k_lin = k_lin[mask]
        Pk = Pk[:, :, mask]

        # Interpolate log10 P(k) in redshift space to target redshifts
        log_Pk = np.log10(Pk)
        log_Pk_interp = np.array([
            interp1d(self.z_bins, log_Pk[i], kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')(redshifts)
            for i in range(n_samples)
        ])

        return k_lin, 10 ** log_Pk_interp
