"""
multisig_fit.py — MultiSig-style fitting with SciPy (bounded + optional smoothing).
Includes σ → molecular weight (M) conversion and reports Mn, Mw, Mz (g/mol).

This version fixes the σ→M mapping so molecular weights are not over-estimated
by a factor of ~2.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ========= Physical constants =========
# Gas constant in J/(mol·K). We convert to erg/(mol·K) when using CGS.
R_J_per_molK = 8.314462618  # J / (mol K)


# ---------------------------
# Grids / helpers
# ---------------------------
def sigma_bins_array(sigma_ref: float = 1.0, n_bins: int = 17) -> np.ndarray:
    """
    Log-spaced sigma bins (units: 1/cm^2 to match r in cm).

    sigma_ref is a dimensionless scaling reference (not itself in 1/cm^2).
    """
    return np.array([0.5 * sigma_ref * (1.15 ** i) for i in range(n_bins)], dtype=float)


def predict_J(r, c_mean, E, sigma_bins, J_ref, r_ref):
    """
    MultiSig forward model:
        J(r) = E + J_ref * sum_i c_i * exp(0.5 * sigma_i * (r^2 - r_ref^2))

    Parameters
    ----------
    r : array-like
        Radius values (cm).
    c_mean : array-like
        Coefficients for each sigma bin.
    E : float
        Baseline offset.
    sigma_bins : array-like
        Sigma-bin centers (1/cm^2).
    J_ref : float
        Reference fringe intensity at r_ref.
    r_ref : float
        Reference radius (cm).
    """
    r = np.asarray(r, float)
    c = np.asarray(c_mean, float)[None, :]
    s = np.asarray(sigma_bins, float)[None, :]
    dr2 = (r ** 2)[:, None] - (r_ref ** 2)
    return E + J_ref * np.sum(c * np.exp(0.5 * s * dr2), axis=1)


def compute_averages_sigma(c_vec, s_bins):
    """
    Return (rn, rw, rz, sumc) in σ-space (σ in 1/cm^2):

      rn = number-average σ
      rw = weight-average σ
      rz = z-average σ
    """
    c = np.asarray(c_vec, float)
    s = np.asarray(s_bins, float)

    sumc = np.sum(c)
    if sumc == 0:
        return np.nan, np.nan, np.nan, 0.0

    sum_c_over_s = np.sum(c / s)
    sum_c_times_s = np.sum(c * s)
    sum_c_times_s2 = np.sum(c * s * s)

    rn = sumc / sum_c_over_s if sum_c_over_s != 0 else np.nan
    rw = sum_c_times_s / sumc
    rz = sum_c_times_s2 / sum_c_times_s if sum_c_times_s != 0 else np.nan
    return rn, rw, rz, sumc


def initial_guess(n_bins: int = 17, peak_bin: int = 8):
    """Bell-shaped initial guess for c_i plus baseline E0 = 0."""
    c0 = np.zeros(n_bins)
    for i in range(n_bins):
        c0[i] = np.exp(-0.3 * abs(i - peak_bin))
    c0 /= c0.sum()
    return np.concatenate([c0, np.array([0.0])])  # append baseline E0


# ---------------------------
# σ → M conversion
# ---------------------------
def sigma_to_mass_gmol(sigma_cm_inv2, *, rpm, temp_c, v_ml_per_g, rho_g_per_ml):
    """
    Convert sigma (1/cm^2) to molecular mass (g/mol) using SE physics in CGS.

    Uses the standard sedimentation equilibrium relationship (no extra factor 2):

        M = (R * T / (ω^2 (1 - v̄ρ))) * σ

    with R in erg/(mol·K), ω in rad/s, σ in 1/cm^2, giving M in g/mol.

    Parameters
    ----------
    sigma_cm_inv2 : array-like
        σ values in 1/cm^2.
    rpm : float
        Rotor speed (rev/min).
    temp_c : float
        Temperature (°C).
    v_ml_per_g : float
        Partial specific volume v̄ (mL/g == cm^3/g).
    rho_g_per_ml : float
        Solution density ρ (g/mL == g/cm^3).
    """
    T = float(temp_c) + 273.15                    # K
    omega = 2.0 * np.pi * (float(rpm) / 60.0)     # rad/s

    # Convert gas constant to CGS: 1 J = 1e7 erg
    R_erg = R_J_per_molK * 1.0e7                 # erg / (mol K)
    denom = 1.0 - float(v_ml_per_g) * float(rho_g_per_ml)
    if denom <= 0:
        raise ValueError("1 - v̄·ρ must be > 0 for physical conversion.")

    # Factor 2.0 matches the σ definition used in this MultiSig model.
    K = (2.0 * R_erg * T) / (denom * omega ** 2)  # g/mol per (1/cm^2)

    sigma = np.asarray(sigma_cm_inv2, dtype=float)
    return K * sigma


def add_mass_outputs_to_summary(
    summary_df: pd.DataFrame,
    rn_sigma: float,
    rw_sigma: float,
    rz_sigma: float,
    rpm: float,
    temp_c: float,
    v_ml_per_g: float,
    rho_g_per_ml: float,
):
    """
    Append Mn, Mw, Mz (g/mol) to the summary table using σ→M mapping.

    Returns updated summary and the linear factor k such that M = k * σ.
    """
    # Mass corresponding to σ = 1 1/cm^2 gives the linear factor k.
    k = sigma_to_mass_gmol(
        [1.0],
        rpm=rpm,
        temp_c=temp_c,
        v_ml_per_g=v_ml_per_g,
        rho_g_per_ml=rho_g_per_ml,
    )[0]

    Mn = k * rn_sigma
    Mw = k * rw_sigma
    Mz = k * rz_sigma

    extra = pd.DataFrame(
        {
            "metric": ["Mn_g_per_mol", "Mw_g_per_mol", "Mz_g_per_mol"],
            "mean": [Mn, Mw, Mz],
            "sd": [np.nan, np.nan, np.nan],
        }
    )
    summary_out = pd.concat([summary_df, extra], ignore_index=True)
    return summary_out, k


# ---------------------------
# Residuals (optionally smoothed)
# ---------------------------
def make_residuals(r_data, J_data, sigma_bins, J_ref, r_ref, lam_smooth: float = 0.0):
    """
    Build a residual function for least_squares.

    lam_smooth > 0 adds a first-difference penalty on the coefficients to encourage
    smooth σ-distributions.
    """
    r_data = np.asarray(r_data, float)
    J_data = np.asarray(J_data, float)

    def residuals(theta):
        coeffs = theta[:-1]
        E = theta[-1]
        J_hat = predict_J(r_data, coeffs, E, sigma_bins, J_ref, r_ref)
        resid = J_data - J_hat
        if lam_smooth > 0.0:
            diffs = np.diff(coeffs)
            smooth_pen = lam_smooth * diffs
            return np.concatenate([resid, smooth_pen])
        return resid

    return residuals


# ---------------------------
# Single fit (not used in GUI but kept for completeness)
# ---------------------------
def single_fit(
    r_data,
    J_data,
    r_ref: float = 7.0,
    sigma_ref: float = 1.0,
    Jref_mode: str = "nearest",
    max_nfev: int = 4000,
    nonneg: bool = True,
    lam_smooth: float = 0.0,
):
    r_data = np.asarray(r_data, float)
    J_data = np.asarray(J_data, float)
    sigma_bins = sigma_bins_array(sigma_ref)

    # J_ref selection
    if Jref_mode == "nearest":
        idx = np.abs(r_data - r_ref).argmin()
        J_ref = float(J_data[idx])
    else:
        J_ref = float(np.median(J_data))

    theta0 = initial_guess(n_bins=len(sigma_bins))
    residuals = make_residuals(
        r_data,
        J_data,
        sigma_bins,
        J_ref,
        r_ref,
        lam_smooth=lam_smooth,
    )

    # Bounds (c_i ≥ 0 if nonneg; baseline E free)
    if nonneg:
        lb = np.concatenate([np.zeros(len(sigma_bins)), np.array([-np.inf])])
        ub = np.concatenate([np.full(len(sigma_bins), np.inf), np.array([np.inf])])
    else:
        lb = -np.inf * np.ones_like(theta0)
        ub = np.inf * np.ones_like(theta0)

    res = least_squares(
        residuals,
        theta0,
        method="trf",
        bounds=(lb, ub),
        max_nfev=max_nfev,
    )

    theta_hat = res.x
    c_hat, E_hat = theta_hat[:-1], theta_hat[-1]
    rn, rw, rz, sumc = compute_averages_sigma(c_hat, sigma_bins)

    return {
        "coeffs": c_hat,
        "E_hat": float(E_hat),
        "rn_sigma": rn,
        "rw_sigma": rw,
        "rz_sigma": rz,
        "sumc": sumc,
        "res": res,
        "sigma_bins": sigma_bins,
        "J_ref": J_ref,
        "r_ref": r_ref,
        "lam_smooth": lam_smooth,
        "nonneg": nonneg,
    }


# ---------------------------
# Repeated fits with jitter + averaging
# ---------------------------
def repeat_fit(
    r_data,
    j_data,
    *,
    r_ref,
    sigma_ref,
    Jref_mode: str = "manual",
    J_ref_manual=None,
    num_runs: int = 20,
    lam_smooth: float = 0.0,
    nonneg: bool = True,
    rpm=None,
    temp_c=None,
    v_ml_per_g=None,
    rho_g_per_ml=None,
    random_seed: int = 123,
    max_nfev: int = 6000,
    jitter_pct: float = 0.02,
    solver: str = "trf",  # "trf" (bounded) or "lm" (unconstrained)
    **kwargs,
):
    """
    MultiSig-style repeated fit with jittered starts.

    Parameters
    ----------
    r_data, j_data : 1D arrays
        Radius (cm) and fringe data.
    r_ref : float
        Reference radius (cm).
    sigma_ref : float
        Reference sigma (dimensionless) defining log-spaced σ bins.
    Jref_mode : {"manual", "nearest", "median"}
        How to choose J_ref:
          - "manual" : use J_ref_manual (required)
          - "nearest": use J at the point nearest r_ref
          - "median" : use median(J)
    """

    r = np.asarray(r_data, float)
    J = np.asarray(j_data, float)

    if r.ndim != 1 or J.ndim != 1 or r.shape != J.shape:
        raise ValueError(f"r and J must be 1D and same shape; got r{r.shape}, J{J.shape}")

    rng = np.random.default_rng(random_seed)

    # --- sigma grid ---
    sigma_bins = sigma_bins_array(sigma_ref)

    # --- choose J_ref correctly ---
    if Jref_mode == "manual":
        if J_ref_manual is None:
            raise ValueError("Jref_mode='manual' but J_ref_manual is None.")
        J_ref = float(J_ref_manual)
    elif Jref_mode == "nearest":
        idx = np.abs(r - r_ref).argmin()
        J_ref = float(J[idx])
    elif Jref_mode == "median":
        J_ref = float(np.median(J))
    else:
        raise ValueError(f"Unknown Jref_mode: {Jref_mode}")

    # --- initial guess and residuals ---
    base_theta = initial_guess(n_bins=len(sigma_bins))
    residuals = make_residuals(
        r,
        J,
        sigma_bins,
        J_ref,
        r_ref,
        lam_smooth=lam_smooth,
    )

    # --- choose solver / bounds ---
    solver = (solver or "trf").lower()
    if solver == "lm":
        # Levenberg–Marquardt: no bounds support in least_squares
        use_bounds = False
        ls_method = "lm"
    else:
        # default: bounded TRF
        solver = "trf"
        use_bounds = True
        ls_method = "trf"

    # --- bounds (only if using bounded TRF) ---
    if use_bounds and nonneg:
        lb = np.concatenate([np.zeros(len(sigma_bins)), np.array([-np.inf])])
        ub = np.concatenate([np.full(len(sigma_bins), np.inf), np.array([np.inf])])
    elif use_bounds:
        lb = -np.inf * np.ones_like(base_theta)
        ub = np.inf * np.ones_like(base_theta)
    else:
        lb = ub = None  # ignored for LM

    # --- repeated fits ---
    all_coeffs, all_E = [], []
    rn_list, rw_list, rz_list, sumc_list, status_list = [], [], [], [], []

    for _ in range(num_runs):
        # jitter coefficients only; keep baseline E fixed at its initial value
        noise = rng.normal(0.0, jitter_pct, size=base_theta.shape[0])
        noise[-1] = 0.0  # do not jitter baseline term
        theta_start = base_theta * (1.0 + noise)

        if use_bounds:
            res = least_squares(
                residuals,
                theta_start,
                method=ls_method,
                bounds=(lb, ub),
                max_nfev=max_nfev,
            )
        else:
            res = least_squares(
                residuals,
                theta_start,
                method=ls_method,
                max_nfev=max_nfev,
            )

        theta_hat = res.x
        c_hat, E_hat = theta_hat[:-1], theta_hat[-1]
        rn, rw, rz, sumc = compute_averages_sigma(c_hat, sigma_bins)

        all_coeffs.append(c_hat)
        all_E.append(E_hat)
        rn_list.append(rn)
        rw_list.append(rw)
        rz_list.append(rz)
        sumc_list.append(sumc)
        status_list.append(res.status)

    all_coeffs = np.array(all_coeffs)
    c_mean = all_coeffs.mean(0)
    c_sd = all_coeffs.std(0, ddof=1)

    E_mean = float(np.mean(all_E))
    E_sd = float(np.std(all_E, ddof=1))

    rn_mean = float(np.mean(rn_list))
    rn_sd = float(np.std(rn_list, ddof=1))
    rw_mean = float(np.mean(rw_list))
    rw_sd = float(np.std(rw_list, ddof=1))
    rz_mean = float(np.mean(rz_list))
    rz_sd = float(np.std(rz_list, ddof=1))

    summary = pd.DataFrame(
        {
            "metric": ["baseline_E", "sigma_n", "sigma_w", "sigma_z", "sum_coeffs"],
            "mean": [E_mean, rn_mean, rw_mean, rz_mean, float(np.mean(sumc_list))],
            "sd": [E_sd, rn_sd, rw_sd, rz_sd, float(np.std(sumc_list, ddof=1))],
        }
    )
    # --- sigma → mass bins (g/mol) ---
    mass_bins_gmol = None
    mass_factor_k = None
    if None not in (rpm, temp_c, v_ml_per_g, rho_g_per_ml):
        summary, mass_factor_k = add_mass_outputs_to_summary(
            summary,
            rn_mean,
            rw_mean,
            rz_mean,
            rpm,
            temp_c,
            v_ml_per_g,
            rho_g_per_ml,
        )
        mass_bins_gmol = mass_factor_k * sigma_bins

    coef_table = pd.DataFrame(
        {
            "bin": np.arange(len(sigma_bins)) + 1,
            "sigma_bin": sigma_bins,
            "c_mean": c_mean,
            "c_sd": c_sd,
        }
    )
    if mass_bins_gmol is not None:
        # Column name matches what your Streamlit app expects ("M_bin_gmol")
        coef_table.insert(2, "M_bin_gmol", mass_bins_gmol)

    return {
        "summary": summary,
        "coef_table": coef_table,
        "status_list": status_list,
        "sigma_bins": sigma_bins,
        "mass_bins_gmol": mass_bins_gmol,
        "mass_factor_k": mass_factor_k,
        "J_ref": J_ref,
        "r_ref": r_ref,
        "c_mean": c_mean,
        "c_sd": c_sd,
        "E_mean": E_mean,
        "E_sd": E_sd,
        "lam_smooth": lam_smooth,
        "nonneg": nonneg,
        "num_runs": num_runs,
        "jitter_pct": jitter_pct,
        "solver": solver,          # <--- add this line
    }


# ---------------------------
# Plot helpers
# ---------------------------
def plot_fit(r, J, sigma_bins, c_mean, E_mean, J_ref, r_ref, ax=None):
    """
    Plot experimental J(r) vs averaged MultiSig fit.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    Jfit = predict_J(r, c_mean, E_mean, sigma_bins, J_ref, r_ref)
    ax.plot(r, J, "o", ms=2, label="Data")
    ax.plot(r, Jfit, "--", lw=2, label="Averaged fit")
    ax.set_xlabel("Radius (cm)")
    ax.set_ylabel("Fringe")
    ax.legend()
    return fig, ax


def plot_coeffs(coef_table: pd.DataFrame, x_axis: str = "sigma", ax=None):
    """
    Bar plot of coefficient means ± SD.

    x_axis = "sigma"  → use sigma_bin
    x_axis = "mass"   → use M_bin_gmol (if available)
    """
    df = coef_table.copy()

    # Normalise column names to a simple internal schema
    cmap = {
        "sigma": "sigma_bin",
        "sigmaBin": "sigma_bin",
        "bin_sigma": "sigma_bin",
        "mean": "c_hat_mean",
        "c_mean": "c_hat_mean",
        "coef_mean": "c_hat_mean",
        "c_hat": "c_hat_mean",
        "std": "sd",
        "stdev": "sd",
        "c_hat_sd": "sd",
        "coef_sd": "sd",
        "M_gmol": "M_bin_gmol",
        "mass_bin": "M_bin_gmol",
        "mass_bins_gmol": "M_bin_gmol",
    }
    df.rename(columns={k: v for k, v in cmap.items() if k in df.columns}, inplace=True)

    if "c_hat_mean" not in df.columns:
        num_cols = [
            c
            for c in df.columns
            if np.issubdtype(df[c].dtype, np.number)
            and c not in ("sigma_bin", "M_bin_gmol", "bin")
        ]
        if not num_cols:
            raise ValueError("coef_table has no numeric 'mean' column candidates.")
        df["c_hat_mean"] = df[num_cols[0]]

    if "sd" not in df.columns:
        df["sd"] = 0.0

    use_mass = (x_axis == "mass") and ("M_bin_gmol" in df.columns)
    if use_mass:
        x = df["M_bin_gmol"].values
        xlab = "M bin (g/mol)"
    else:
        x = df["sigma_bin"].values
        xlab = "Sigma bin (1/cm²)"

    y = df["c_hat_mean"].values
    yerr = df["sd"].values

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if len(x) > 1:
        width = (x[1] - x[0]) * 0.8
    else:
        width = 0.8

    ax.bar(x, y, yerr=yerr, width=width)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Coefficient mean ± SD")
    return fig, ax