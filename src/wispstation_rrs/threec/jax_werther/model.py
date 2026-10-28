# -*- coding: utf-8 -*-
# 3C model of Groetsch et al. (2017), JAX implementation by Mortimer Werther (Eawag, 2025).
# The original Python implementation of the 3C model is distributed under LGPL-3.0-or-later.
# This copy is the 3C fit of the paper processing. The model, the weights, the parameter
# table, the lmfit L-BFGS-B options and the jax.grad Jacobian are unchanged. Plotting and the
# timing printout are removed.
"""
Python implementation of an analytic remote sensing reflectance model as described in:

    Gege, P., & Groetsch, P. (2016). A spectral model for correcting sun glint and sky glint. In Proceedings of Ocean Optics XXIII.

    Groetsch, P. M. M., Gege, P., Simis, S. G. H., Eleveld, M. A., and Peters, S. W. M.: Validation of a spectral correction procedure
    for sun and sky reflections in above-water reflectance measurements (2017). Optics Express, 25(16), A742-A761.

This code was originally implemented in Theano; here we use JAX with JIT-compilation and automatic differentiation
to speed up the evaluations.

Two fit paths give bitwise the same results:
    fit_LuEd           code path of the paper processing: cost and gradient are traced and
                       compiled for every fit.
    fit_LuEd_compiled  cost and gradient are compiled once per model, the measured ratios and the
                       weights are passed as arguments (five to ten times faster).
"""

import os

# double precision must be on before jax is imported
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)  # double precision
import jax.numpy as jnp  # noqa: E402

import lmfit as lm  # noqa: E402
from lmfit import Parameters  # noqa: E402

SPECTRA_DIR = Path(__file__).resolve().parent / "spectra"

# model grid of the paper processing (350-900 nm, 1 nm, int64)
WL = np.arange(350, 901, dtype=np.int64)

# parameter order used by the model and by the Jacobian
PARAM_ORDER = ['beta', 'alpha', 'am', 'rh', 'pressure',
               'C_chl', 'C_sm', 'C_mie', 'n_mie',
               'C_y', 'S_y', 'T_w',
               'theta_sun', 'theta_view',
               'n_w', 'rho_s', 'rho_dd', 'rho_ds', 'delta']

# lmfit L-BFGS-B options of the paper processing (the 'disp' print flag is left out)
LBFGSB_OPTIONS = {'gtol': 1e-16, 'eps': 1e-07, 'maxiter': 15000, 'ftol': 1e-16, 'maxls': 20, 'maxcor': 20}


def paper_params():
    """Base parameter set of the paper processing (theta_sun, theta_view, rh, pressure and T_w are set per measurement)."""
    init_params = Parameters()
    init_params.add('beta',      value=0.5,    vary=True,  min=0.05, max=7)
    init_params.add('alpha',     value=1.2,    vary=True,  min=0.1,  max=2.5)
    init_params.add('am',        value=1.0,    vary=False, min=1.0,  max=10)
    init_params.add('rh',        value=60,     vary=False, min=0,    max=100)
    init_params.add('pressure',  value=1013.25, vary=False, min=950,  max=1050)

    init_params.add('C_chl',     value=5,      vary=True,  min=0.01, max=100)
    init_params.add('C_sm',      value=1,      vary=True,  min=0.01, max=100)
    init_params.add('C_mie',     value=0,      vary=False, min=0,    max=100)
    init_params.add('n_mie',     value=-1,     vary=False, min=-2,   max=2)
    init_params.add('C_y',       value=0.5,    vary=True,  min=0.01, max=5)
    init_params.add('S_y',       value=0.018,  vary=False, min=0.01, max=0.03)
    init_params.add('T_w',       value=20,     vary=False, min=0,    max=35)

    init_params.add('theta_sun', value=40,     vary=False, min=0,    max=90)
    init_params.add('theta_view', value=40,    vary=False, min=0,    max=180)

    init_params.add('n_w',       value=1.34,   vary=False, min=1.33, max=1.34)
    init_params.add('rho_s',     value=0.0256, vary=False, min=0.0,  max=0.1)
    init_params.add('rho_dd',    value=0.0,    vary=True,  min=0,    max=0.1)
    init_params.add('rho_ds',    value=0.01,   vary=True,  min=0.0,  max=0.1)
    init_params.add('delta',     value=0.00,   vary=False, min=0,    max=1)
    return init_params


def paper_weights(wl):
    """Spectral weights of the paper processing: 5 up to 500 nm, 0.1 at 675-750 and 760-770 nm, 1 elsewhere."""
    weights = pd.Series(1.0, index=wl)
    weights.loc[:500] = 5
    weights.loc[675:750] = 0.1
    weights.loc[760:770] = 0.1
    return weights.values


def measured_ratios(Ls, Lu, Ed, weights, screen_nonfinite=True):
    """Ls/Ed, Lu/Ed and weights entering the cost.

    screen_nonfinite True: the ratios are 0 where Ed is not finite or zero, and wavelengths with
    values that are not finite get weight 0. False: plain ratios. Both give the same arrays for
    finite inputs with Ed above 0.
    """
    if screen_nonfinite:
        with np.errstate(divide='ignore', invalid='ignore'):
            safe_Ls_Ed = np.divide(
                Ls, Ed,
                out=np.zeros_like(Ls, dtype=float),
                where=np.isfinite(Ed) & (np.abs(Ed) > 0)
            )
            safe_Lu_over_Ed = np.divide(
                Lu, Ed,
                out=np.zeros_like(Lu, dtype=float),
                where=np.isfinite(Ed) & (np.abs(Ed) > 0)
            )

        weights_np = np.asarray(weights, dtype=float).copy()
        invalid_mask = (
            ~np.isfinite(safe_Ls_Ed)
            | ~np.isfinite(safe_Lu_over_Ed)
            | ~np.isfinite(weights_np)
        )
        weights_np[invalid_mask] = 0.0
        if not np.any(weights_np > 0):
            raise ValueError("No wavelength with finite values left.")
        return safe_Ls_Ed, safe_Lu_over_Ed, weights_np
    return Ls / Ed, Lu / Ed, weights


class rrs_model_3C(object):
    def __init__(self, downsampling=False, wl_range=False, spectra_path=SPECTRA_DIR):
        """
        Parameters:
            spectra_path -- path to input spectra files.
            downsampling -- reduce spectral resolution (False or window length for a rolling mean).
            wl_range     -- limit spectral range (False or [min_wavelength, max_wavelength]).
        """
        spectra_path = Path(spectra_path)
        # load reflectance spectra from file; filenames and skiprows values are based on the original implementation.
        a_ph = pd.read_csv(spectra_path / 'phyto.A', skiprows=11, sep='\t', index_col=0).iloc[:, 0]
        a_w = pd.read_csv(spectra_path / 'water.A', skiprows=10, sep='\t', index_col=0, encoding='cp1252').iloc[:, 0]
        daw_dT = pd.read_csv(spectra_path / 'daWdT.txt', skiprows=10, sep='\t', index_col=0, encoding='cp1252').iloc[:, 0]
        astar_y = pd.read_csv(spectra_path / 'Y.A', skiprows=11, delimiter=' ', index_col=0).loc[350:].iloc[:, 0]

        d = pd.DataFrame(index=a_ph.index)
        d.index.name = 'Wavelength, [nm]'
        d['astar_ph'] = a_ph
        d['astar_y'] = astar_y
        d['a_w'] = a_w
        d['daw_dT'] = daw_dT

        if not downsampling:
            self.spectra = d
        else:
            self.spectra = d.rolling(window=downsampling, center=True, min_periods=1).mean()

        if wl_range:
            self.spectra = self.spectra.loc[wl_range[0]:wl_range[1]].dropna()

        self.wl = np.array(self.spectra.index)
        # pre-store constant spectra as they are used repeatedly in evaluations.
        self.a_w = self.spectra['a_w'].values
        self.daw_dT = self.spectra['daw_dT'].values
        self.astar_ph = self.spectra['astar_ph'].values
        self.astar_y = self.spectra['astar_y'].values

        # parameter names in the order used by the model.
        self.param_order = list(PARAM_ORDER)

        # compile the JAX based model function
        self.model = self._compile_model()
        # cost and gradient of fit_LuEd_compiled, built on first use
        self._compiled = None

    def _compile_model(self):
        """
        JAX implementation of the 3C model as a function of input parameters.
        This function mirrors the original Theano implementation, with additional
        tolerance-based checks in the conditionals to reduce systematic differences.
        """
        @jax.jit  # JIT-compile the model function to speed up evaluations
        def model(beta, alpha, am, rh, pressure,
                  C_chl, C_sm, C_mie, n_mie,
                  C_y, S_y, T_w, theta_sun, theta_view, n_w,
                  rho_s, rho_dd, rho_ds, delta,
                  wl, a_w, daw_dT, astar_ph, astar_y, Ls_Ed):
            # set tolerance for near-boundary comparisons.
            tol = 1e-5

            # atmospheric model part
            wl_a = 550.0
            theta_sun_rad = theta_sun * jnp.pi / 180.0

            # use tolerance in the alpha-based conditionals:
            # z3 = -0.1417 * alpha + 0.82 remains the same.
            z3 = -0.1417 * alpha + 0.82
            # for values at or near 1.2, use 0.65.
            z2 = jnp.where((alpha > 1.2) | jnp.isclose(alpha, 1.2, atol=tol), 0.65, z3)
            # for values at or near 0, use 0.82.
            z1 = jnp.where((alpha < 0) | jnp.isclose(alpha, 0, atol=tol), 0.82, z2)
            theta_sun_mean = z1

            B3 = jnp.log(1 - theta_sun_mean)
            B2 = B3 * (0.0783 + B3 * (-0.3824 - 0.5874 * B3))
            B1 = B3 * (1.459 + B3 * (0.1595 + 0.4129 * B3))
            Fa = 1 - 0.5 * jnp.exp((B1 + B2 * jnp.cos(theta_sun_rad)) * jnp.cos(theta_sun_rad))

            omega_a = (-0.0032 * am + 0.972) * jnp.exp(3.06e-4 * rh)
            tau_a = beta * (wl / wl_a) ** (-alpha)

            M = 1.0 / (jnp.cos(theta_sun_rad) + 0.50572 * (90.0 + 6.07995 - theta_sun) ** (-1.6364))
            M_ = M * pressure / 1013.25

            Tr = jnp.exp(- M_ / (115.6406 * (wl / 1000.0) ** 4 - 1.335 * (wl / 1000.0) ** 2))
            Tas = jnp.exp(- omega_a * tau_a * M)

            Edd = Tr * Tas
            Edsr = 0.5 * (1 - Tr ** 0.95)
            Edsa = Tr ** 1.5 * (1 - Tas) * Fa

            Ed = Edd + Edsr + Edsa
            Edd_Ed = Edd / Ed
            Edsr_Ed = Edsr / Ed
            Edsa_Ed = Edsa / Ed
            Eds_Ed = Edsr_Ed + Edsa_Ed

            # bio-optical model part (Albert and Mobley)
            a_ph = C_chl * astar_ph
            wl_ref_y = 440.0
            # use tolerance in the S_y check, so that S_y values near -1 use the first branch.
            a_y = jnp.where(jnp.isclose(S_y, -1, atol=tol), C_y * astar_y,
                            C_y * jnp.exp(-S_y * (wl - wl_ref_y)))

            T_w_ref = 20.0
            a_w_corr = a_w + (T_w - T_w_ref) * daw_dT

            a = a_w_corr + a_ph + a_y

            bbstar_sm = 0.0086
            bbstar_mie = 0.0042
            wl_ref_mie = 500.0
            bb_sm = C_sm * bbstar_sm + C_mie * bbstar_mie * (wl / wl_ref_mie) ** n_mie

            # using isclose to check n_w already ensures proper branching.
            b1 = jnp.where(jnp.isclose(n_w, 1.34, atol=tol), 0.00144, 0.00111)
            wl_ref_water = 500.0
            S_water = -4.32
            bb_water = b1 * (wl / wl_ref_water) ** S_water
            bb = bb_water + bb_sm

            omega_b = bb / (bb + a)

            theta_sun_ss = jnp.arcsin(jnp.sin(theta_sun_rad) / n_w)
            theta_view_rad = theta_view * jnp.pi / 180.0
            theta_view_ss = jnp.arcsin(jnp.sin(theta_view_rad) / n_w)

            p_f = [0.1034, 1, 3.3586, -6.5358, 4.6638, 2.4121]
            p_frs = [0.0512, 1, 4.6659, -7.8387, 5.4571, 0.1098, 0.4021]

            f_val = p_f[0] * (p_f[1] + p_f[2] * omega_b + p_f[3] * omega_b ** 2 + p_f[4] * omega_b ** 3)
            f_val *= (1 + p_f[5] / jnp.cos(theta_sun_ss))
            R0minus = f_val * omega_b

            frs = p_frs[0] * (p_frs[1] + p_frs[2] * omega_b + p_frs[3] * omega_b ** 2 + p_frs[4] * omega_b ** 3)
            frs *= (1 + p_frs[5] / jnp.cos(theta_sun_ss)) * (1 + p_frs[6] / jnp.cos(theta_view_ss))
            Rrs0minus = frs * omega_b

            Rrs_refl = rho_s * Ls_Ed + rho_dd * Edd_Ed / jnp.pi + rho_ds * Eds_Ed / jnp.pi + delta

            gamma = 0.48
            zeta = 0.518
            Rrs = zeta * Rrs0minus / (1 - gamma * R0minus)

            Lu_Ed = Rrs + Rrs_refl

            return Rrs, Rrs_refl, Lu_Ed

        return model

    def _constants(self, wl):
        return dict(wl=jnp.array(wl), a_w=jnp.array(self.a_w), daw_dT=jnp.array(self.daw_dT),
                    astar_ph=jnp.array(self.astar_ph), astar_y=jnp.array(self.astar_y))

    def _minimize(self, cost, grad, params, verbose):
        """lmfit L-BFGS-B with the options of the paper processing and the jax.grad Jacobian as Dfun."""
        param_order = self.param_order

        def objective_from_params(lmfit_params):
            """Cost for lmfit Parameters (bridge between lmfit and the JAX objective)."""
            p = lmfit_params.valuesdict()
            pvec = jnp.array([p[name] for name in param_order])
            return float(cost(pvec))

        def objective_jac(lmfit_params):
            """Gradient of the cost (jax.grad) for the varied parameters, used as Dfun by lmfit."""
            p = lmfit_params.valuesdict()
            pvec = jnp.array([p[name] for name in param_order])
            jac_full = grad(pvec)
            # only return gradients for parameters that are varied
            free_idx = [i for i, name in enumerate(param_order) if lmfit_params[name].vary]
            return np.array(jac_full)[free_idx]

        options = dict(LBFGSB_OPTIONS)
        if verbose:
            options['disp'] = True
        reg = lm.minimize(objective_from_params,
                          params=params,
                          method='lbfgsb',
                          Dfun=objective_jac,
                          options=options)
        # final residual at the solution
        return reg, objective_from_params(reg.params)

    def _outputs(self, reg, resid_final, consts, Ls_Ed_const, Lu, Ed):
        # with the optimised parameters, compute the additional outputs using the jitted model
        p_opt = reg.params.valuesdict()
        Rrs_modelled, Rrs_refl, Lu_Ed_modelled = self.model(
                beta     = p_opt['beta'],    alpha    = p_opt['alpha'],  am      = p_opt['am'],
                rh       = p_opt['rh'],      pressure = p_opt['pressure'],
                C_chl    = p_opt['C_chl'],   C_sm     = p_opt['C_sm'],   C_mie   = p_opt['C_mie'],
                n_mie    = p_opt['n_mie'],
                C_y      = p_opt['C_y'],     S_y      = p_opt['S_y'],    T_w     = p_opt['T_w'],
                theta_sun= p_opt['theta_sun'], theta_view = p_opt['theta_view'],
                n_w      = p_opt['n_w'],
                rho_s    = p_opt['rho_s'],   rho_dd   = p_opt['rho_dd'], rho_ds  = p_opt['rho_ds'],
                delta    = p_opt['delta'],
                wl       = consts['wl'],
                a_w      = consts['a_w'],
                daw_dT   = consts['daw_dT'],
                astar_ph = consts['astar_ph'],
                astar_y  = consts['astar_y'],
                Ls_Ed   = Ls_Ed_const
        )
        Rrs_obs = np.array(Lu/Ed) - np.array(Rrs_refl)

        msg = getattr(reg, 'message', '')
        reg_min = {'final_params': {k: float(v) for k, v in p_opt.items()}, 'resid': resid_final,
                   'success': bool(getattr(reg, 'success', False)),
                   'message': msg.decode('ascii', 'replace') if isinstance(msg, bytes) else str(msg),
                   'nfev': int(getattr(reg, 'nfev', 0)), 'nit': int(getattr(reg, 'nit', -1))}
        return reg_min, np.array(Rrs_modelled), np.array(Rrs_refl), np.array(Lu_Ed_modelled), Rrs_obs

    def fit_LuEd(self, wl, Ls, Lu, Ed, params, weights, verbose=True, screen_nonfinite=True):
        """
        Fit the 3C model to a set of radiometric measurements (code path of the paper processing).

        Inputs:
            wl     -- wavelength array (numpy array)
            Ls     -- sky radiance (numpy array)
            Lu     -- upwelling radiance (numpy array)
            Ed     -- downwelling irradiance (numpy array)
            params -- lmfit Parameters object
            weights-- spectral weighting function (numpy array)
            verbose-- flag to display optimization feedback
            screen_nonfinite -- see measured_ratios

        Outputs:
            reg_min      -- dictionary with the final parameters, the cost and the optimiser status
            Rrs_modelled -- modelled remote sensing reflectance (numpy array)
            Rrs_refl     -- surface-reflected reflectance (numpy array)
            Lu_Ed_modelled -- modelled Lu/Ed spectrum (numpy array)
            Rrs_obs      -- derived remote sensing reflectance (numpy array)
        """
        # precompute constant arrays as JAX arrays; these remain the same for a given call.
        consts = self._constants(wl)
        ls_ed, lu_ed, w = measured_ratios(Ls, Lu, Ed, weights, screen_nonfinite)
        Ls_Ed_const = jnp.array(ls_ed)
        Lu_over_Ed_const = jnp.array(lu_ed)
        weights_const = jnp.array(w)

        # use the parameter order defined in __init__
        param_order = self.param_order

        # objective function mapping a vector of parameters to the residual cost
        def objective_func_vec(pvec):
            """Weighted sum of squared residuals between modelled and measured Lu/Ed for a parameter vector."""
            d = {name: pvec[i] for i, name in enumerate(param_order)}
            Rrs_modelled, Rrs_refl, Lu_Ed_modelled = self.model(
                beta     = d['beta'],    alpha    = d['alpha'],  am      = d['am'],
                rh       = d['rh'],      pressure = d['pressure'],
                C_chl    = d['C_chl'],   C_sm     = d['C_sm'],   C_mie   = d['C_mie'],
                n_mie    = d['n_mie'],
                C_y      = d['C_y'],     S_y      = d['S_y'],    T_w     = d['T_w'],
                theta_sun= d['theta_sun'], theta_view = d['theta_view'],
                n_w      = d['n_w'],
                rho_s    = d['rho_s'],   rho_dd   = d['rho_dd'], rho_ds  = d['rho_ds'],
                delta    = d['delta'],
                wl       = consts['wl'],
                a_w      = consts['a_w'],
                daw_dT   = consts['daw_dT'],
                astar_ph = consts['astar_ph'],
                astar_y  = consts['astar_y'],
                Ls_Ed    = Ls_Ed_const
            )
            resid = jnp.sum(((Lu_Ed_modelled - Lu_over_Ed_const) ** 2) * weights_const)
            return resid

        # JIT compile the cost function and its gradient
        objective_func_vec_compiled = jax.jit(objective_func_vec)
        objective_grad = jax.jit(jax.grad(objective_func_vec))

        reg, resid_final = self._minimize(objective_func_vec_compiled, objective_grad, params, verbose)
        return self._outputs(reg, resid_final, consts, Ls_Ed_const, Lu, Ed)

    def _compiled_cost(self, wl):
        """Cost and gradient compiled once for the grid ``wl``; Ls/Ed, Lu/Ed and weights are arguments."""
        key = np.asarray(wl).tobytes()
        if self._compiled is None or self._compiled[0] != key:
            consts = self._constants(wl)
            param_order = self.param_order

            def objective_func_vec(pvec, Ls_Ed, Lu_over_Ed, weights):
                """Weighted sum of squared residuals between modelled and measured Lu/Ed for a parameter vector."""
                d = {name: pvec[i] for i, name in enumerate(param_order)}
                Rrs_modelled, Rrs_refl, Lu_Ed_modelled = self.model(
                    beta     = d['beta'],    alpha    = d['alpha'],  am      = d['am'],
                    rh       = d['rh'],      pressure = d['pressure'],
                    C_chl    = d['C_chl'],   C_sm     = d['C_sm'],   C_mie   = d['C_mie'],
                    n_mie    = d['n_mie'],
                    C_y      = d['C_y'],     S_y      = d['S_y'],    T_w     = d['T_w'],
                    theta_sun= d['theta_sun'], theta_view = d['theta_view'],
                    n_w      = d['n_w'],
                    rho_s    = d['rho_s'],   rho_dd   = d['rho_dd'], rho_ds  = d['rho_ds'],
                    delta    = d['delta'],
                    wl       = consts['wl'],
                    a_w      = consts['a_w'],
                    daw_dT   = consts['daw_dT'],
                    astar_ph = consts['astar_ph'],
                    astar_y  = consts['astar_y'],
                    Ls_Ed    = Ls_Ed
                )
                resid = jnp.sum(((Lu_Ed_modelled - Lu_over_Ed) ** 2) * weights)
                return resid

            # gradient with respect to the parameter vector only, as jax.grad in the paper processing
            self._compiled = (key, consts, jax.jit(objective_func_vec), jax.jit(jax.grad(objective_func_vec)))
        return self._compiled[1:]

    def fit_LuEd_compiled(self, wl, Ls, Lu, Ed, params, weights, verbose=False, screen_nonfinite=True):
        """As fit_LuEd, with the cost and gradient compiled once per model instead of once per fit."""
        consts, cost_c, grad_c = self._compiled_cost(wl)
        ls_ed, lu_ed, w = measured_ratios(Ls, Lu, Ed, weights, screen_nonfinite)
        Ls_Ed_const = jnp.array(ls_ed)
        Lu_over_Ed_const = jnp.array(lu_ed)
        weights_const = jnp.array(w)

        def cost(pvec):
            return cost_c(pvec, Ls_Ed_const, Lu_over_Ed_const, weights_const)

        def grad(pvec):
            return grad_c(pvec, Ls_Ed_const, Lu_over_Ed_const, weights_const)

        reg, resid_final = self._minimize(cost, grad, params, verbose)
        return self._outputs(reg, resid_final, consts, Ls_Ed_const, Lu, Ed)
