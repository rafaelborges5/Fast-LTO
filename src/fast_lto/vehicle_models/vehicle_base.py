"""
Abstract interface for vehicle models used in the optimizer.

Convention
----------
state[0] = s     arc-length progress along the centerline
state[1] = d     lateral deviation from the centerline
state[2:] = ...  model-specific (e.g. psi_err, v, yaw_rate, ...)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import casadi as ca
import numpy as np

from fast_lto.utils.smooth import smoothmax


class CornerOffset(NamedTuple):
    name: str
    dx: float
    dy: float


class VehicleModel(ABC):
    """
    Abstract base class for vehicle models used in the optimizer.

    Concrete models must provide:
      - state names and input names
      - physical time-domain dynamics:  x_dot = f(x, u, kappa)
      - physical inequality constraints: g(x, u) <= 0
      - finite physical bounds for all OPTIMIZED states (excluding s) and all inputs

    The base class computes per-instance normalisation factors for the
    reduced state (everything except s) and for the inputs
    """

    def __init__(self, params: dict) -> None:
        self.params = params

        # Normalisation metadata, as NumPy and as CasADi DM.
        self._x_red_lb: Optional[np.ndarray] = None
        self._x_red_ub: Optional[np.ndarray] = None
        self._x_red_scale_np: Optional[np.ndarray] = None
        self._x_red_shift_np: Optional[np.ndarray] = None
        self._x_red_scale_dm: Optional[ca.DM] = None
        self._x_red_shift_dm: Optional[ca.DM] = None

        self._u_lb: Optional[np.ndarray] = None
        self._u_ub: Optional[np.ndarray] = None
        self._u_scale_np: Optional[np.ndarray] = None
        self._u_shift_np: Optional[np.ndarray] = None
        self._u_scale_dm: Optional[ca.DM] = None
        self._u_shift_dm: Optional[ca.DM] = None

        self._init_normalisation()

    @property
    def nx(self) -> int:
        """Number of states in the full state vector (including *s*)."""
        return len(self.get_state_names())

    @property
    def nu(self) -> int:
        """Number of inputs."""
        return len(self.get_input_names())

    @property
    def nx_reduced(self) -> int:
        """Number of states in the space-domain formulation (full minus *s*)."""
        return self.nx - 1

    def reduced_state_names(self) -> List[str]:
        """State names excluding *s* (the space-domain reduced states)."""
        return self.get_state_names()[1:]

    @abstractmethod
    def get_state_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_input_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_dynamics(
        self,
        states: Sequence[ca.MX],
        inputs: Sequence[ca.MX],
        curvature: ca.MX,
    ) -> ca.MX:
        """
        Symbolic expression for the TIME derivatives: x_dot = f(x, u, kappa).
        """
        raise NotImplementedError

    @abstractmethod
    def get_constraints(
        self,
        states: Sequence[ca.MX],
        inputs: Sequence[ca.MX],
        curvature: ca.MX,
    ) -> List[ca.MX]:
        """
        Inequality constraints g(x, u, kappa) <= 0.
        """
        raise NotImplementedError

    def get_default_params(self) -> dict:
        return {}

    def diagnostics(self, x_red: ca.MX, u: ca.MX) -> Dict[str, ca.MX]:
        """Named per-node quantities derived from the reduced state and inputs.

        These are the internals a plot wants to show — tire loads, slip angles,
        friction usage — expressed symbolically, from the same parameters and
        the same formulae the solver used. Evaluate them over a solved
        trajectory with
        :func:`fast_lto.vehicle_models.diagnostics.evaluate_diagnostics`.

        Exists so plotting code never has to re-derive the physics in NumPy:
        a second implementation drifts from the first, and then the plot you
        would use to catch the drift is the thing that drifted.

        Takes the inputs as well as the state because models disagree about
        where a quantity lives: the four-wheel model carries steering as a
        state, the dynamic bicycle as an input.

        Default: nothing. A model reports whatever it can.
        """
        return {}

    def get_corner_offsets(self) -> List[CornerOffset]:
        corners_raw = self.params.get("corners", [])
        return [c if isinstance(c, CornerOffset) else CornerOffset(*c) for c in corners_raw]

    def get_corner_constraints(
        self,
        d_phys: ca.MX,
        psi_err_phys: ca.MX,
        kappa: ca.MX,
        w_left: ca.MX,
        w_right: ca.MX,
    ) -> List[ca.MX]:
        """Corridor constraints for each corner of the car, not just its CoG.

        A corner held at ``dx`` ahead of the CoG follows a centreline that has
        bent away by ``0.5 * kappa / D_kappa * dx^2`` to second order, which is
        why an outside front corner loses room in a turn even with the CoG
        centred. Cross-checked against the NumPy form in ``warm_start`` by
        ``tests/test_corner_geometry.py``.
        """
        corners = self.get_corner_offsets()
        if not corners:
            return []

        g_list: List[ca.MX] = []
        sin_psi = ca.sin(psi_err_phys)
        cos_psi = ca.cos(psi_err_phys)
        D_kappa = 1 - kappa * d_phys

        for c in corners:
            long_proj = c.dx * cos_psi - c.dy * sin_psi
            d_corner = (
                d_phys + c.dx * sin_psi + c.dy * cos_psi - 0.5 * kappa / D_kappa * long_proj**2
            )

            if c.dy >= 0:
                g_list.append(d_corner - w_left)
            else:
                g_list.append(-w_right - d_corner)

        return g_list

    def state_bounds(self) -> Optional[Tuple[List[float], List[float]]]:
        """
        Bounds on the FULL state vector [s, d, ...] in PHYSICAL units.

        Concrete models are expected to override this and provide finite
        bounds for all OPTIMISED states
        """
        return None

    def reduced_state_bounds(self) -> Optional[Tuple[List[float], List[float]]]:
        """Bounds on the reduced state vector [d, ...] (everything except *s*)."""
        bounds = self.state_bounds()
        if bounds is None:
            return None
        lb, ub = bounds
        return lb[1:], ub[1:]

    def input_bounds(self) -> Optional[Tuple[List[float], List[float]]]:
        """Bounds on the INPUT vector in PHYSICAL units."""
        return None

    def _init_normalisation(self) -> None:
        """
        Pre-compute scaling/shift for reduced states and inputs based on
        physical bounds, mapping approximately to [-1, 1].
        """

        red_bounds = self.reduced_state_bounds()
        if red_bounds is not None:
            lb_red, ub_red = red_bounds
            lb_arr = np.asarray(lb_red, dtype=float)
            ub_arr = np.asarray(ub_red, dtype=float)

            scale = 0.5 * (ub_arr - lb_arr)
            shift = 0.5 * (ub_arr + lb_arr)

            near_zero = np.isclose(scale, 0.0)
            scale[near_zero] = 1.0
            shift[near_zero] = 0.0

            # Exactly symmetric bounds drop a term from every expression.
            symmetric = np.isclose(shift, 0.0, atol=1e-6)
            shift[symmetric] = 0.0

            self._x_red_lb = lb_arr
            self._x_red_ub = ub_arr
            self._x_red_scale_np = scale
            self._x_red_shift_np = shift
            self._x_red_scale_dm = ca.DM(scale)
            self._x_red_shift_dm = ca.DM(shift)

        in_bounds = self.input_bounds()
        if in_bounds is not None:
            lb_u, ub_u = in_bounds
            lb_arr_u = np.asarray(lb_u, dtype=float)
            ub_arr_u = np.asarray(ub_u, dtype=float)

            scale_u = 0.5 * (ub_arr_u - lb_arr_u)
            shift_u = 0.5 * (ub_arr_u + lb_arr_u)

            near_zero_u = np.isclose(scale_u, 0.0)
            scale_u[near_zero_u] = 1.0
            shift_u[near_zero_u] = 0.0

            symmetric_u = np.isclose(shift_u, 0.0, atol=1e-6)
            shift_u[symmetric_u] = 0.0

            self._u_lb = lb_arr_u
            self._u_ub = ub_arr_u
            self._u_scale_np = scale_u
            self._u_shift_np = shift_u
            self._u_scale_dm = ca.DM(scale_u)
            self._u_shift_dm = ca.DM(shift_u)

    def get_reduced_state_scaling(self) -> Tuple[Optional[ca.DM], Optional[ca.DM]]:
        """
        Return (scale, shift) for the reduced state in CasADi DM form.
        """
        return self._x_red_scale_dm, self._x_red_shift_dm

    def get_input_scaling(self) -> Tuple[Optional[ca.DM], Optional[ca.DM]]:
        """
        Return (scale, shift) for the inputs in CasADi DM form.
        """
        return self._u_scale_dm, self._u_shift_dm

    @staticmethod
    def physical_to_norm(val_phys: ca.MX, scale: ca.DM, shift: ca.DM) -> ca.MX:
        """Generic affine map from physical values to normalised space."""
        return (val_phys - shift) / scale

    @staticmethod
    def norm_to_physical(val_norm: ca.MX, scale: ca.DM, shift: ca.DM) -> ca.MX:
        """Generic affine map from normalised space back to physical."""
        return val_norm * scale + shift

    def reduced_state_phys_to_norm(self, x_red_phys: ca.MX) -> ca.MX:
        """Map reduced physical state [d, ...] -> normalised coordinates."""
        if self._x_red_scale_dm is None or self._x_red_shift_dm is None:
            return x_red_phys
        return self.physical_to_norm(x_red_phys, self._x_red_scale_dm, self._x_red_shift_dm)

    def reduced_state_norm_to_phys(self, x_red_norm: ca.MX) -> ca.MX:
        """Map reduced normreduced_state_phys_to_normalised state [d, ...] -> physical coordinates."""
        if self._x_red_scale_dm is None or self._x_red_shift_dm is None:
            return x_red_norm
        return self.norm_to_physical(x_red_norm, self._x_red_scale_dm, self._x_red_shift_dm)

    def input_phys_to_norm(self, u_phys: ca.MX) -> ca.MX:
        """Map physical inputs u -> normalised inputs."""
        if self._u_scale_dm is None or self._u_shift_dm is None:
            return u_phys
        return self.physical_to_norm(u_phys, self._u_scale_dm, self._u_shift_dm)

    def input_norm_to_phys(self, u_norm: ca.MX) -> ca.MX:
        """Map normalised inputs -> physical inputs u."""
        if self._u_scale_dm is None or self._u_shift_dm is None:
            return u_norm
        return self.norm_to_physical(u_norm, self._u_scale_dm, self._u_shift_dm)

    def reduced_state_bounds_normalized(self) -> Optional[Tuple[List[float], List[float]]]:
        """
        Bounds on the reduced state in the normalised domain.
        """
        red_bounds = self.reduced_state_bounds()
        if red_bounds is None or self._x_red_scale_np is None or self._x_red_shift_np is None:
            return None
        lb_red, ub_red = red_bounds
        lb_arr = np.asarray(lb_red, dtype=float)
        ub_arr = np.asarray(ub_red, dtype=float)
        scale = self._x_red_scale_np
        shift = self._x_red_shift_np
        lb_n = (lb_arr - shift) / scale
        ub_n = (ub_arr - shift) / scale
        return lb_n.tolist(), ub_n.tolist()

    def input_bounds_normalized(self) -> Optional[Tuple[List[float], List[float]]]:
        """
        Bounds on the inputs in the normalised domain.
        """
        in_bounds = self.input_bounds()
        if in_bounds is None or self._u_scale_np is None or self._u_shift_np is None:
            return None
        lb_u, ub_u = in_bounds
        lb_arr_u = np.asarray(lb_u, dtype=float)
        ub_arr_u = np.asarray(ub_u, dtype=float)
        scale_u = self._u_scale_np
        shift_u = self._u_shift_np
        lb_n_u = (lb_arr_u - shift_u) / scale_u
        ub_n_u = (ub_arr_u - shift_u) / scale_u
        return lb_n_u.tolist(), ub_n_u.tolist()

    def get_dynamics_normalized(
        self,
        x_red_norm: ca.MX,
        u_norm: ca.MX,
        curvature: ca.MX,
    ) -> ca.MX:
        """
        Space-domain dynamics in the normalised reduced state.
        """

        x_scale, x_shift = self.get_reduced_state_scaling()
        u_scale, u_shift = self.get_input_scaling()
        if x_scale is None or x_shift is None or u_scale is None or u_shift is None:
            raise RuntimeError("Normalisation scales/shifts are not defined. ")

        x_red_phys = self.norm_to_physical(x_red_norm, x_scale, x_shift)
        u_phys = self.norm_to_physical(u_norm, u_scale, u_shift)

        full_state = ca.vertcat(ca.MX(0), x_red_phys)
        x_dot_phys_full = self.get_dynamics(full_state, u_phys, curvature)

        s_dot = x_dot_phys_full[0]
        x_dot_phys_red = x_dot_phys_full[1:]
        s_dot_floor = float(self.params.get("eps_s_dot", 1e-3))
        s_dot_smooth_eps = float(self.params.get("smoothmax_eps", 1e-3))
        s_dot_safe = smoothmax(s_dot, ca.MX(s_dot_floor), s_dot_smooth_eps)

        x_red_dot_phys_per_s = x_dot_phys_red / s_dot_safe

        return x_red_dot_phys_per_s / x_scale

    def get_constraints_normalized(
        self,
        x_red_norm: ca.MX,
        u_norm: ca.MX,
        curvature: ca.MX,
    ) -> List[ca.MX]:
        """
        Inequality constraints g(x_norm, u_norm) <= 0 evaluated in
        physical space but taking normalised arguments.
        """
        x_scale, x_shift = self.get_reduced_state_scaling()
        u_scale, u_shift = self.get_input_scaling()
        if x_scale is None or x_shift is None or u_scale is None or u_shift is None:
            raise RuntimeError("Normalisation scales/shifts are not defined.")

        x_red_phys = self.norm_to_physical(x_red_norm, x_scale, x_shift)
        u_phys = self.norm_to_physical(u_norm, u_scale, u_shift)
        full_state = ca.vertcat(ca.MX(0), x_red_phys)
        return self.get_constraints(full_state, u_phys, curvature)
