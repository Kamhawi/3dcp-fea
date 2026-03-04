# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Age-dependent material evolution laws and unit conversion helpers.

This module evaluates scalar constitutive time-laws used by
``materials.material_state.MaterialStateManager`` to update DG0 property fields.

Physics:
    The implemented laws capture early-age rheology/hardening for printable
    cementitious materials, including:
    - thixotropy-driven yield stress growth,
    - viscosity evolution,
    - transition of Poisson ratio from fresh to hardened state.

Math:
    Representative relations include:
    - tau_y(t): piecewise linear + power-law growth,
    - nu(t): exponential decay from ``nu_fresh`` to ``nu_hard``,
    - G = min(tau_y / gamma_c, E_inf / (2(1 + nu_hard))),
    - E = 2 G (1 + nu).
"""

import numpy as np


def clamp_age(t_age):
    """Clamp age values to non-negative values.

    Args:
        t_age: Scalar or array-like material age values.

    Returns:
        np.ndarray: Non-negative age array.

    Raises:
        None.
    """
    return np.maximum(np.asarray(t_age, dtype=float), 0.0)


def pa_to_mpa(value_pa):
    """Convert pressure/stress-like values from Pa to MPa.

    Args:
        value_pa: Scalar or array-like quantity in pascals.

    Returns:
        np.ndarray: Converted value in megapascals.

    Raises:
        None.
    """
    return np.asarray(value_pa, dtype=float) * 1.0e-6


def kg_m3_to_kg_mm3(rho_kg_m3):
    """Convert density from ``kg/m^3`` to ``kg/mm^3``.

    Args:
        rho_kg_m3: Scalar or array-like density in ``kg/m^3``.

    Returns:
        np.ndarray: Density in ``kg/mm^3``.

    Raises:
        None.
    """
    return np.asarray(rho_kg_m3, dtype=float) / 1.0e9


def kg_m3_to_ns2_mm4(rho_kg_m3):
    """Convert ``kg/m^3`` density to ``N*s^2/mm^4`` consistent FE units.

    Args:
        rho_kg_m3: Scalar or array-like density in ``kg/m^3``.

    Returns:
        np.ndarray: Density in ``N*s^2/mm^4``.

    Raises:
        None.

    Derivation:
      kg/m^3 -> kg/mm^3: /1e9
      kg -> N*s^2/m -> N*s^2/(1000 mm): multiply by 1e-3
      => factor = 1e-12 overall
    """
    return np.asarray(rho_kg_m3, dtype=float) * 1.0e-12


def compute_yield_stress_mpa(
    t_age, tau_0_mpa, a_thix_mpa_per_s, t_set_s, n_h, out=None
):
    """Evaluate piecewise early-age shear yield stress in MPa.

    Args:
        t_age: Material age (scalar or array-like).
        tau_0_mpa: Initial shear yield stress at deposition.
        a_thix_mpa_per_s: Linear thixotropic growth rate before ``t_set``.
        t_set_s: Setting time separating early and hardened regimes.
        n_h: Hardening exponent used after ``t_set``.
        out: Optional output buffer for in-place evaluation.

    Returns:
        np.ndarray: Shear yield stress ``tau_y`` in MPa.

    Raises:
        None.

    Math:
        For ``t < t_set``:
            tau_y = tau_0 + A_thix * t
        For ``t >= t_set``:
            tau_y = tau_set * (t / t_set)^n_h
    """
    t = clamp_age(t_age)
    t_set = max(float(t_set_s), 1.0e-9)
    tau_set = tau_0_mpa + a_thix_mpa_per_s * t_set

    tau_linear = tau_0_mpa + a_thix_mpa_per_s * t
    tau_hydration = tau_set * np.power(np.maximum(t / t_set, 1.0), float(n_h))
    if out is None:
        return np.where(t < t_set, tau_linear, tau_hydration)
    np.copyto(out, tau_hydration)
    mask = t < t_set
    out[mask] = tau_linear[mask]
    return out


def compute_poisson_ratio(t_age, nu_fresh, nu_hard, t_set_s, out=None):
    """Evaluate age-dependent Poisson ratio transition.

    Args:
        t_age: Material age (scalar or array-like).
        nu_fresh: Fresh-state Poisson ratio.
        nu_hard: Hardened-state Poisson ratio.
        t_set_s: Characteristic time scale for transition.
        out: Optional output buffer for in-place evaluation.

    Returns:
        np.ndarray: Poisson ratio values.

    Raises:
        None.

    Math:
        nu(t) = nu_hard + (nu_fresh - nu_hard) * exp(-t / t_set)
    """
    t = clamp_age(t_age)
    t_set = max(float(t_set_s), 1.0e-9)
    if out is None:
        return nu_hard + (nu_fresh - nu_hard) * np.exp(-t / t_set)
    np.divide(-t, t_set, out=out)
    np.exp(out, out=out)
    out *= float(nu_fresh - nu_hard)
    out += float(nu_hard)
    return out


def compute_shear_modulus_mpa(tau_y_mpa, gamma_c, e_inf_mpa, nu_hard, out=None):
    """Compute shear modulus with rheological estimate and elastic cap.

    Args:
        tau_y_mpa: Shear yield stress in MPa.
        gamma_c: Critical shear strain scale.
        e_inf_mpa: Long-term/hardened Young modulus cap in MPa.
        nu_hard: Hardened Poisson ratio used for cap conversion.
        out: Optional output buffer for in-place evaluation.

    Returns:
        np.ndarray: Shear modulus values in MPa.

    Raises:
        None.

    Math:
        G = min(tau_y / gamma_c, E_inf / (2(1 + nu_hard)))
    """
    gamma_c_eff = max(float(gamma_c), 1.0e-12)
    tau_arr = np.asarray(tau_y_mpa, dtype=float)
    g_from_rheology = tau_arr / gamma_c_eff
    g_cap = float(e_inf_mpa) / (2.0 * (1.0 + float(nu_hard)))
    if out is None:
        return np.minimum(g_from_rheology, g_cap)
    np.minimum(g_from_rheology, g_cap, out=out)
    return out


def compute_young_modulus_mpa(g_mpa, nu, out=None):
    """Compute Young modulus from shear modulus and Poisson ratio.

    Args:
        g_mpa: Shear modulus in MPa.
        nu: Poisson ratio.
        out: Optional output buffer for in-place evaluation.

    Returns:
        np.ndarray: Young modulus values in MPa.

    Raises:
        None.

    Math:
        E = 2 * G * (1 + nu)
    """
    g_arr = np.asarray(g_mpa, dtype=float)
    nu_arr = np.asarray(nu, dtype=float)
    if out is None:
        return 2.0 * g_arr * (1.0 + nu_arr)
    np.add(1.0, nu_arr, out=out)
    out *= g_arr
    out *= 2.0
    return out


def compute_viscosity_mpa_s(t_age, mu_p_mpa_s, t_set_s, out=None):
    """Evaluate exponential plastic viscosity growth in MPa*s.

    Args:
        t_age: Material age (scalar or array-like).
        mu_p_mpa_s: Initial viscosity parameter in MPa*s.
        t_set_s: Characteristic growth time.
        out: Optional output buffer for in-place evaluation.

    Returns:
        np.ndarray: Viscosity values in MPa*s.

    Raises:
        None.

    Math:
        eta(t) = mu_p * exp(t / t_set)
    """
    t = clamp_age(t_age)
    t_set = max(float(t_set_s), 1.0e-9)
    if out is None:
        return float(mu_p_mpa_s) * np.exp(t / t_set)
    np.divide(t, t_set, out=out)
    np.exp(out, out=out)
    out *= float(mu_p_mpa_s)
    return out


def compute_bond_quality(
    t_open_s,
    tau_0_pa,
    a_thix_pa_per_s,
    nozzle_pressure_pa,
):
    """Compute interface bond quality from pressure penetration index.

    Args:
        t_open_s: Interface opening age/time in seconds.
        tau_0_pa: Initial yield stress in Pa.
        a_thix_pa_per_s: Thixotropic growth rate in Pa/s.
        nozzle_pressure_pa: Effective nozzle/deposition pressure in Pa.

    Returns:
        np.ndarray: Bond quality factor ``beta_bond`` in ``[0, 1)``.

    Raises:
        None.

    Physics:
        Higher nozzle pressure and younger interface age increase penetration
        and bonding effectiveness.

    Math:
        phi = p_deposit / tau_sub
        beta_bond = 1 - exp(-phi)
    """
    t_open = clamp_age(t_open_s)
    # ``tau_sub`` is the substrate strength at the interface age.
    tau_sub_pa = np.asarray(tau_0_pa, dtype=float) + np.asarray(a_thix_pa_per_s, dtype=float) * t_open

    p_deposit_pa = float(nozzle_pressure_pa)

    # Guard division to avoid singularity when ``tau_sub`` is near zero.
    phi = p_deposit_pa / np.maximum(tau_sub_pa, 1.0e-12)
    return 1.0 - np.exp(-phi)
