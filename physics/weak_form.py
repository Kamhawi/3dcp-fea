# Author: Abdallah Kamhawi <Kamhawi@umich.edu>
# Package Maintainer: Abdallah Kamhawi <Kamhawi@umich.edu>

"""Weak-form construction for bulk EVP + cohesive interfaces.

This module builds the nonlinear residual and Jacobian forms used by the
monolithic solve. The algebra mirrors the original implementation.

Physics:
    The formulation combines:
    - small-strain bulk elastoviscoplasticity (with explicit ``eps_vp`` update
      outside this module),
    - activation weighting for not-yet-born material,
    - mixed-mode cohesive tractions on inter-layer interfaces,
    - SIP-like stabilization/consistency terms on selected interior facets.
"""

import ufl
from dolfinx import fem

from solver.kinematics import epsilon, sigma, sigma_from_strain


def build_evp_cohesive_weak_form(
    msh,
    V,
    materials,
    birth_time_func,
    interior_facet_tags,
    dirichlet_tags,
    cfg,
    return_ufl=False,
    alpha_facet_mode="min",
    include_facet_terms=True,
):
    """Build and compile nonlinear residual/Jacobian for the simulation.

    Args:
        msh: DOLFINx mesh.
        V: Displacement function space (DG1 vector).
        materials: MaterialStateManager containing DG0 fields and `u`.
        birth_time_func: DG0 cell birth-time field.
        interior_facet_tags: Meshtags for interior interfaces.
        dirichlet_tags: Meshtags for weak Dirichlet boundaries.
        cfg: Parsed configuration dictionary.
        include_facet_terms: When ``True`` (DG path) assemble the cohesive,
            bonded-SIPG, and jump-stabilization interior-facet (``dS``) terms.
            When ``False`` (CG path) skip every ``dS`` term and all ``("+")``/
            ``("-")`` restrictions, leaving bulk EVP + gravity + the weak
            Dirichlet ``ds(1)`` Nitsche boundary — the only consistent form on
            a continuous space where ``jump(u) = 0`` identically.

    Returns:
        tuple: ``(F_form, J_form)`` compiled DOLFINx forms.  When
            ``return_ufl=True``, returns ``(F_form, J_form, F_ufl, J_ufl)``
            with the pre-compilation UFL expressions for diagnostic use.

    Raises:
        ValueError: If ``materials.u`` has not been assigned before form build.

    Math overview:
        - Activation: alpha = alpha_min + (1-alpha_min)*0.5*(1+tanh(k(t-t_birth)))
        - Bulk stress: sigma = lambda tr(eps-eps_vp) I + 2 mu (eps-eps_vp)
        - Cohesive traction: mixed normal/tangential penalty with exponential
          softening damage and residual stiffness.
    """
    if not hasattr(materials, "u"):
        raise ValueError(
            "materials.u is required. Assign the displacement fem.Function before building forms."
        )

    u = materials.u
    v = ufl.TestFunction(V)

    activation_cfg = cfg["activation"]
    material_cfg = cfg["material"]
    interface_cfg = cfg["interface"]

    dx = ufl.Measure("dx", domain=msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=dirichlet_tags)

    mu = materials.E / (2 * (1 + materials.nu))
    lam = materials.E * materials.nu / ((1 + materials.nu) * (1 - 2 * materials.nu))

    n = ufl.FacetNormal(msh)
    h_boundary = ufl.CellDiameter(msh)
    gamma_boundary = interface_cfg["gamma_boundary_mult"] * materials.E / h_boundary

    # Smooth birth activation via tanh ramp.
    materials.t_current = fem.Constant(msh, 0.0)
    sharpness = activation_cfg["sharpness"]
    alpha_min = activation_cfg["alpha_min"]
    alpha_raw = 0.5 * (1.0 + ufl.tanh(sharpness * (materials.t_current - birth_time_func)))
    alpha = alpha_min + (1.0 - alpha_min) * alpha_raw

    sigma_u = sigma_from_strain(epsilon(u) - materials.eps_vp, lam, mu)

    # Residual form:
    # - bulk internal virtual work,
    # - cohesive traction work on inter-layer facets (DG only),
    # - SIP terms on bonded interfaces (DG only) and weak Dirichlet boundaries,
    # - body force loading.
    F = alpha * ufl.inner(sigma_u, epsilon(v)) * dx

    if include_facet_terms:
        # Facet tags are interpreted by convention:
        # dS(1) -> inter-layer cohesive, dS(2) -> intra-layer bonded terms.
        dS = ufl.Measure("dS", domain=msh, subdomain_data=interior_facet_tags)
        h = ufl.avg(ufl.CellDiameter(msh))
        gamma_bonded = interface_cfg["gamma_bonded_mult"] * ufl.avg(materials.E) / h

        if alpha_facet_mode == "min":
            alpha_facet = ufl.min_value(alpha("+"), alpha("-"))
        elif alpha_facet_mode == "max":
            alpha_facet = ufl.max_value(alpha("+"), alpha("-"))
        elif alpha_facet_mode == "avg":
            alpha_facet = 0.5 * (alpha("+") + alpha("-"))
        else:
            raise ValueError(f"Unknown alpha_facet_mode: {alpha_facet_mode!r}")
        # Inter-layer cohesive activation gate. The interface facet exists in the
        # mesh from the start (tagged before any deposition), so it must only go
        # "live" once BOTH adjacent layers are born -> min(alpha+, alpha-). Using
        # max() here was a bug: when the upper layer is still unborn (its DOFs
        # pinned to zero) max() leaves the bond fully active, spuriously clamping
        # the live top layer's surface to the frozen void above it. That clamp
        # makes each topmost layer behave "as if solid" and releases abruptly
        # when the next layer deposits, stretching the sub-top layers (they show
        # unphysical negative compression). The bond STRENGTH is set separately
        # by the cohesive law (beta_bond, sigma_y), not by this activation gate.
        cohesive_activation_mode = interface_cfg.get("cohesive_activation_mode", "min")
        if cohesive_activation_mode == "max":
            alpha_facet_cohesive = ufl.max_value(alpha("+"), alpha("-"))
        elif cohesive_activation_mode == "avg":
            alpha_facet_cohesive = 0.5 * (alpha("+") + alpha("-"))
        else:
            alpha_facet_cohesive = ufl.min_value(alpha("+"), alpha("-"))

        bonded_only = bool(interface_cfg.get("bonded_only", False))

        # ``ufl.jump(u)`` is the displacement discontinuity across an interior
        # facet. It drives cohesive opening/sliding laws on dS(1), unless the
        # bonded-DG control is requested. In that control every interior facet
        # receives the same fully bonded SIPG treatment used for intra-layer
        # facets; this isolates cohesive-interface physics from the DG space
        # and activation/discretization machinery.
        jump_u = ufl.jump(u)

        if not bonded_only:
            # Normal-sign convention. ``jump_n = <[[u]], n("+")>`` is invariant to
            # the DOLFINx +/- swap, but whether jump_n > 0 reads as OPENING or as
            # CLOSING (interpenetration) depends on the mesh's facet/cell ordering.
            # On some meshes (the non-planar cylinder among them) it lands inverted:
            # interpenetrating layers give jump_n > 0, so the contact branch
            # (min(jump_n,0)) never engages and the cohesive law accrues spurious
            # opening damage while layers sink through each other. ``cohesive_normal_flip``
            # flips the interface normal so that delta_n > 0 is reliably OPENING.
            # Default False preserves the established barrel-vault / verification
            # behavior; the cylinder sets it True.
            cohesive_normal_flip = bool(interface_cfg.get("cohesive_normal_flip", False))
            n_il = -n("+") if cohesive_normal_flip else n("+")
            jump_n_scalar = ufl.dot(jump_u, n_il)
            delta_n_open = ufl.max_value(jump_n_scalar, 0.0)
            delta_t = jump_u - jump_n_scalar * n_il
            delta_t_sq = ufl.inner(delta_t, delta_t)

            tau_0_pa = material_cfg["tau_0"]
            a_thix_pa_s = material_cfg["A_thix"]
            g_i_c = interface_cfg["G_Ic"]
            g_ii_c = interface_cfg["G_IIc"]
            nozzle_pressure_pa = interface_cfg["nozzle_pressure"]
            residual_stiffness_ratio = interface_cfg.get("residual_stiffness_ratio", 0.01)
            k_min_mult = interface_cfg.get("K_min_mult", 0.1)

            t_open_facet = abs(birth_time_func("+") - birth_time_func("-"))
            tau_sub_pa_facet = tau_0_pa + a_thix_pa_s * t_open_facet
            p_deposit_pa = nozzle_pressure_pa
            phi_facet = p_deposit_pa / ufl.max_value(tau_sub_pa_facet, 1.0e-12)
            # Bond quality from nozzle-pressure penetration index:
            #   beta = 1 - exp(-phi)
            beta_bond_facet = 1.0 - ufl.exp(-phi_facet)
            beta_bond_facet = ufl.min_value(1.0, ufl.max_value(beta_bond_facet, 1.0e-8))

            g_i_c_interface = ufl.max_value(beta_bond_facet * g_i_c, 1.0e-12)
            g_ii_c_interface = ufl.max_value(beta_bond_facet * g_ii_c, 1.0e-12)

            sigma_bond = beta_bond_facet * ufl.avg(materials.sigma_y)
            tau_bond = beta_bond_facet * ufl.avg(materials.tau_y)

            K_n = sigma_bond * sigma_bond / (2.0 * g_i_c_interface)
            K_t = tau_bond * tau_bond / (2.0 * g_ii_c_interface)
            K_min = k_min_mult * ufl.avg(materials.E) / h
            K_n = ufl.max_value(K_n, K_min)
            K_t = ufl.max_value(K_t, K_min)

            # Exponential damage driven by mode-I and mode-II displacement energies.
            delta_n_sq = delta_n_open * delta_n_open
            damage_arg_n = delta_n_sq / ufl.max_value(
                2.0 * g_i_c_interface / ufl.max_value(K_n, 1.0e-12), 1.0e-12
            )
            damage_arg_t = delta_t_sq / ufl.max_value(
                2.0 * g_ii_c_interface / ufl.max_value(K_t, 1.0e-12), 1.0e-12
            )
            damage = 1.0 - ufl.exp(-(damage_arg_n + damage_arg_t))
            damage = ufl.min_value(1.0, ufl.max_value(damage, 0.0))
            # Enforce damage irreversibility via per-cell history variable.
            damage = ufl.max_value(damage, ufl.max_value(materials.damage_max("+"), materials.damage_max("-")))

            K_n_res = residual_stiffness_ratio * K_n
            K_t_res = residual_stiffness_ratio * K_t

            # Cohesive traction law with:
            # - opening response in the normal direction,
            # - compressive contact-like penalty for negative normal jump,
            # - tangential traction with damage-softened stiffness.
            T_n = ((1.0 - damage) * K_n + damage * K_n_res) * delta_n_open + K_n * ufl.min_value(
                jump_n_scalar, 0.0
            )
            T_t = ((1.0 - damage) * K_t + damage * K_t_res) * delta_t
            T_cohesive = T_n * n_il + T_t

            # Cohesive traction virtual work on inter-layer facets.
            F += ufl.inner(alpha_facet_cohesive * T_cohesive, ufl.jump(v)) * dS(1)
            # Stiff compressive contact penalty (opt-in via gamma_contact_mult).
            # Bonded layers rest on one another, but the cohesive K_n can fall orders
            # of magnitude below the bulk E/h for low-modulus fresh material, letting
            # the layer above sink through the interface. This extra penalty on the
            # closing branch (jump_n < 0) restores a bulk-scale contact stiffness.
            # It is weighted by alpha_facet (= min activation), so it engages only
            # once BOTH layers are present and never clamps a free top surface to the
            # not-yet-deposited layer above. gamma_contact_mult = 0 (default) leaves
            # the barrel-vault formulation unchanged.
            gamma_contact_mult = interface_cfg.get("gamma_contact_mult", 0.0)
            if gamma_contact_mult > 0.0:
                K_contact = gamma_contact_mult * ufl.avg(materials.E) / h
                F += (
                    alpha_facet
                    * K_contact
                    * ufl.inner(ufl.min_value(jump_n_scalar, 0.0) * n_il, ufl.jump(v))
                    * dS(1)
                )

        # Bonded interfaces: consistent + symmetric SIP couplings.  The normal
        # DG path bonds only intra-layer facets, while the bonded-control path
        # deliberately bonds all interior facets, including former cohesive
        # inter-layer facets and generic interior facets.
        dS_bonded = (dS(0) + dS(1) + dS(2)) if bonded_only else dS(2)
        F += -ufl.inner(alpha_facet * ufl.avg(sigma_u) * n("+"), ufl.jump(v)) * dS_bonded
        F += -ufl.inner(alpha_facet * ufl.avg(sigma(v, lam, mu)) * n("+"), jump_u) * dS_bonded
        F += alpha_facet * gamma_bonded * ufl.inner(jump_u, ufl.jump(v)) * dS_bonded

        # Jump stabilization on non-cohesive interior facets only.  In the
        # bonded-control path, the former dS(1) facets are non-cohesive too.
        gamma_safe = 0.05 * ufl.avg(materials.E) / h
        dS_stab = (dS(0) + dS(1) + dS(2)) if bonded_only else (dS(0) + dS(2))
        F += alpha_facet * gamma_safe * ufl.inner(jump_u, ufl.jump(v)) * dS_stab

    # Weak Dirichlet (Nitsche-like) boundary terms on ds(1) (CG and DG).
    F += -alpha * ufl.inner(sigma_u * n, v) * ds(1)
    F += -alpha * ufl.inner(sigma(v, lam, mu) * n, u) * ds(1)
    F += alpha * gamma_boundary * ufl.inner(u, v) * ds(1)
    g_vec = ufl.as_vector([0.0, 0.0, -materials.rho_const * materials.g_const])
    F += -alpha * ufl.inner(g_vec, v) * dx

    J = ufl.derivative(F, u, ufl.TrialFunction(V))
    if return_ufl:
        return fem.form(F), fem.form(J), F, J
    return fem.form(F), fem.form(J)
