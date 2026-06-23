"""
Interactive HTML lap visualization using Plotly.

Produces a self-contained HTML file with:
  - Track map colored by a selectable variable (speed, Fx, friction util, etc.)
  - Car marker that moves via a slider
  - Strip charts (Fx, Fz, steering, TV moment, speed) with synced cursor
  - Rich hover tooltips on the track showing all states
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def _compute_derived(sol, params):
    """Compute all derived quantities for visualization."""
    s = np.array(sol["arc_lengths"])
    path_xy = np.array(sol["path_xy"])
    v = np.array(sol.get("v", sol.get("v_long")))
    headings = np.array(sol["headings"])
    psi_err = np.array(sol["psi_err"])
    d = np.array(sol["d"])
    kappa = np.array(sol["kappa"])

    Fx_fl = np.array(sol["Fx_fl"]); Fx_fr = np.array(sol["Fx_fr"])
    Fx_rr = np.array(sol["Fx_rr"]); Fx_rl = np.array(sol["Fx_rl"])
    delta = np.array(sol["delta"])
    v_lat = np.array(sol["v_lat"])
    yaw_rate = np.array(sol["yaw_rate"])

    l_f = params["lf"]; l_r = params["lr"]
    a_l = params["a_l"]; a_r = params["a_r"]
    m = params["m"]; g_val = params["g"]
    L = l_f + l_r; W = a_l + a_r

    # Aero
    rho = params.get("rho", 1.225)
    C_l = params.get("C_l", 5.54)
    C_d = params.get("C_d", 1.58)
    C_r = params.get("C_r", 0.15)
    A_f = params.get("A_f", 1.2)
    F_down = 0.5 * rho * C_l * A_f * v**2
    F_drag = 0.5 * rho * C_d * A_f * v**2
    F_roll = m * g_val * C_r

    # Vertical loads (static + aero)
    Fw = [m*g_val*(l_r/L)*(a_r/W), m*g_val*(l_r/L)*(a_l/W),
          m*g_val*(l_f/L)*(a_l/W), m*g_val*(l_f/L)*(a_r/W)]
    Fz_fl = Fw[0] + F_down/4; Fz_fr = Fw[1] + F_down/4
    Fz_rr = Fw[2] + F_down/4; Fz_rl = Fw[3] + F_down/4

    # Slip angles
    vx_fl = v - a_l * yaw_rate; vx_fr = v + a_r * yaw_rate
    vx_rr = v + a_r * yaw_rate; vx_rl = v - a_l * yaw_rate
    vy_f = v_lat + l_f * yaw_rate; vy_r = v_lat - l_r * yaw_rate

    alpha_fl = np.arctan2(vy_f, vx_fl) - delta
    alpha_fr = np.arctan2(vy_f, vx_fr) - delta
    alpha_rr = np.arctan2(vy_r, vx_rr)
    alpha_rl = np.arctan2(vy_r, vx_rl)

    # Lateral forces
    def _pac(alpha, B, C, D):
        return D * np.sin(C * np.arctan(B * alpha))

    B_t = params.get("B_fl", 9.0); C_t = params.get("C_fl", 1.3)
    D_fl_p = params.get("D_fl", 1.2); D_fr_p = params.get("D_fr", 1.2)
    D_rr_p = params.get("D_rr", 1.2); D_rl_p = params.get("D_rl", 1.2)

    Fy_fl = -Fz_fl * _pac(alpha_fl, B_t, C_t, D_fl_p)
    Fy_fr = -Fz_fr * _pac(alpha_fr, B_t, C_t, D_fr_p)
    Fy_rr = -Fz_rr * _pac(alpha_rr, B_t, C_t, D_rr_p)
    Fy_rl = -Fz_rl * _pac(alpha_rl, B_t, C_t, D_rl_p)

    # Friction utilization
    def _util(Fx, Fy, Fz, D_val):
        cap = np.maximum(D_val * Fz, 1.0)
        return np.sqrt(Fx**2 + Fy**2) / cap * 100.0

    util_fl = _util(Fx_fl, Fy_fl, Fz_fl, D_fl_p)
    util_fr = _util(Fx_fr, Fy_fr, Fz_fr, D_fr_p)
    util_rr = _util(Fx_rr, Fy_rr, Fz_rr, D_rr_p)
    util_rl = _util(Fx_rl, Fy_rl, Fz_rl, D_rl_p)

    # Body forces
    cd = np.cos(delta); sd = np.sin(delta)
    Fx_total = (Fx_fl+Fx_fr)*cd - (Fy_fl+Fy_fr)*sd + Fx_rr+Fx_rl - F_roll - F_drag
    Fy_total = (Fx_fl+Fx_fr)*sd + (Fy_fl+Fy_fr)*cd + Fy_rr+Fy_rl
    a_long = Fx_total / m + yaw_rate * v_lat
    a_lat = Fy_total / m - yaw_rate * v

    # Yaw moments
    Mz_Fx = (Fx_fl*(-a_l*cd + l_f*sd) + Fx_fr*(a_r*cd + l_f*sd)
             + Fx_rr*a_r - Fx_rl*a_l)
    Mz_Fy = (Fy_fl*(l_f*cd + a_l*sd) + Fy_fr*(l_f*cd - a_r*sd)
             - Fy_rr*l_r - Fy_rl*l_r)
    Mz_total = Mz_Fx + Mz_Fy

    return {
        "s": s, "x": path_xy[:, 0], "y": path_xy[:, 1],
        "v": v, "v_kmh": v * 3.6, "d": d, "kappa": kappa,
        "psi_err_deg": np.degrees(psi_err),
        "delta_deg": np.degrees(delta),
        "v_lat": v_lat, "yaw_rate": yaw_rate,
        "Fx_fl": Fx_fl, "Fx_fr": Fx_fr, "Fx_rr": Fx_rr, "Fx_rl": Fx_rl,
        "Fy_fl": Fy_fl, "Fy_fr": Fy_fr, "Fy_rr": Fy_rr, "Fy_rl": Fy_rl,
        "Fz_fl": Fz_fl, "Fz_fr": Fz_fr, "Fz_rr": Fz_rr, "Fz_rl": Fz_rl,
        "alpha_fl_deg": np.degrees(alpha_fl), "alpha_fr_deg": np.degrees(alpha_fr),
        "alpha_rr_deg": np.degrees(alpha_rr), "alpha_rl_deg": np.degrees(alpha_rl),
        "util_fl": util_fl, "util_fr": util_fr,
        "util_rr": util_rr, "util_rl": util_rl,
        "a_long": a_long, "a_lat": a_lat,
        "Mz_Fx": Mz_Fx, "Mz_total": Mz_total,
        "F_down": F_down,
    }


def build_interactive(solution_path, output_path, track_csv=None):
    with Path(solution_path).open() as f:
        sol = json.load(f)
    params = sol["model_params"]
    D = _compute_derived(sol, params)
    N = len(D["s"])

    # Load cones
    cones_left = cones_right = None
    if track_csv is not None:
        from utils.track_bounds import load_boundaries
        bd = load_boundaries(Path(track_csv))
        cones_left = bd["left"]
        cones_right = bd["right"]

    # ── Build figure with subplots ──
    fig = make_subplots(
        rows=5, cols=2,
        column_widths=[0.55, 0.45],
        row_heights=[0.38, 0.15, 0.15, 0.15, 0.17],
        specs=[
            [{"rowspan": 1, "type": "xy"}, {"rowspan": 1, "type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[
            "Track (colored by speed)", "GG Diagram",
            "Per-wheel Fx [N]", "Per-wheel Fz [N]",
            "Steering [deg] & Yaw Rate [rad/s]", "Slip Angles [deg]",
            "TV Yaw Moment [Nm]", "Friction Utilization [%]",
            "Speed [m/s]", "Lateral Offset [m]",
        ],
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    WHEEL_COLORS = {"FL": "#1f77b4", "FR": "#ff7f0e", "RR": "#2ca02c", "RL": "#d62728"}

    # ── Hover template for track ──
    hover_text = []
    for i in range(N):
        hover_text.append(
            f"<b>s={D['s'][i]:.1f}m</b><br>"
            f"v={D['v'][i]:.1f} m/s ({D['v_kmh'][i]:.0f} km/h)<br>"
            f"d={D['d'][i]:.3f}m, κ={D['kappa'][i]:.4f}<br>"
            f"δ={D['delta_deg'][i]:.1f}°, ψ_err={D['psi_err_deg'][i]:.1f}°<br>"
            f"<b>Fx:</b> FL={D['Fx_fl'][i]:.0f} FR={D['Fx_fr'][i]:.0f}<br>"
            f"    RR={D['Fx_rr'][i]:.0f} RL={D['Fx_rl'][i]:.0f}<br>"
            f"<b>Fz:</b> FL={D['Fz_fl'][i]:.0f} FR={D['Fz_fr'][i]:.0f}<br>"
            f"    RR={D['Fz_rr'][i]:.0f} RL={D['Fz_rl'][i]:.0f}<br>"
            f"Mz_TV={D['Mz_Fx'][i]:.0f} Nm<br>"
            f"a_long={D['a_long'][i]:.1f} a_lat={D['a_lat'][i]:.1f} m/s²"
        )

    # ── Row 1, Col 1: Track map ──
    if cones_left is not None:
        fig.add_trace(go.Scatter(
            x=cones_left[:, 0], y=cones_left[:, 1],
            mode="markers", marker=dict(size=3, color="#2980b9", symbol="triangle-up"),
            name="Left cones", showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
    if cones_right is not None:
        fig.add_trace(go.Scatter(
            x=cones_right[:, 0], y=cones_right[:, 1],
            mode="markers", marker=dict(size=3, color="#f39c12", symbol="triangle-up"),
            name="Right cones", showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=D["x"], y=D["y"], mode="markers+lines",
        line=dict(width=1, color="rgba(200,200,200,0.3)"),
        marker=dict(
            size=5, color=D["v"], colorscale="Turbo",
            colorbar=dict(title="v [m/s]", x=0.52, len=0.35, y=0.82),
            cmin=float(D["v"].min()), cmax=float(D["v"].max()),
        ),
        text=hover_text, hoverinfo="text",
        name="Trajectory", showlegend=False,
    ), row=1, col=1)

    # Car marker (will be moved by slider)
    fig.add_trace(go.Scatter(
        x=[D["x"][0]], y=[D["y"][0]], mode="markers",
        marker=dict(size=14, color="red", symbol="circle",
                    line=dict(width=2, color="darkred")),
        name="Car", showlegend=False,
    ), row=1, col=1)

    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

    # ── Row 1, Col 2: GG diagram ──
    fig.add_trace(go.Scatter(
        x=D["a_lat"], y=D["a_long"], mode="markers",
        marker=dict(size=3, color=D["v"], colorscale="Turbo",
                    cmin=float(D["v"].min()), cmax=float(D["v"].max()),
                    showscale=False),
        name="GG", showlegend=False,
        hovertemplate="a_lat=%{x:.1f}<br>a_long=%{y:.1f}<extra></extra>",
    ), row=1, col=2)
    # Friction circle
    th = np.linspace(0, 2*np.pi, 100)
    D_g = params.get("D_fl", 1.2) * params.get("g", 9.81)
    fig.add_trace(go.Scatter(
        x=D_g*np.cos(th), y=D_g*np.sin(th), mode="lines",
        line=dict(color="red", dash="dash", width=1),
        name="D·g (static)", showlegend=True,
    ), row=1, col=2)
    fig.update_xaxes(title_text="a_lat [m/s²]", row=1, col=2)
    fig.update_yaxes(title_text="a_long [m/s²]", scaleanchor=None, row=1, col=2)

    # ── Row 2: Fx and Fz ──
    for wh, col_hex in WHEEL_COLORS.items():
        fig.add_trace(go.Scatter(
            x=D["s"], y=D[f"Fx_{wh.lower()}"], mode="lines",
            line=dict(width=1, color=col_hex), name=f"Fx {wh}",
            legendgroup="Fx", showlegend=True,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=D["s"], y=D[f"Fz_{wh.lower()}"], mode="lines",
            line=dict(width=1, color=col_hex), name=f"Fz {wh}",
            legendgroup="Fz", showlegend=True,
        ), row=2, col=2)

    # ── Row 3: Steering + yaw rate, Slip angles ──
    fig.add_trace(go.Scatter(
        x=D["s"], y=D["delta_deg"], mode="lines",
        line=dict(width=1.5, color="purple"), name="δ [deg]",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=D["s"], y=D["yaw_rate"], mode="lines",
        line=dict(width=1, color="orange", dash="dot"), name="yaw rate",
    ), row=3, col=1)

    for wh, col_hex in WHEEL_COLORS.items():
        fig.add_trace(go.Scatter(
            x=D["s"], y=D[f"alpha_{wh.lower()}_deg"], mode="lines",
            line=dict(width=1, color=col_hex), name=f"α {wh}",
            legendgroup="alpha", showlegend=True,
        ), row=3, col=2)

    # ── Row 4: TV moment, Friction utilization ──
    fig.add_trace(go.Scatter(
        x=D["s"], y=D["Mz_Fx"], mode="lines",
        line=dict(width=1.5, color="cyan"), name="Mz_Fx (TV)",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=D["s"], y=D["Mz_total"], mode="lines",
        line=dict(width=1, color="brown", dash="dot"), name="Mz total",
    ), row=4, col=1)

    for wh, col_hex in WHEEL_COLORS.items():
        fig.add_trace(go.Scatter(
            x=D["s"], y=D[f"util_{wh.lower()}"], mode="lines",
            line=dict(width=1, color=col_hex), name=f"util {wh}",
            legendgroup="util", showlegend=True,
        ), row=4, col=2)
    fig.add_hline(y=100, line=dict(color="red", dash="dash", width=0.8), row=4, col=2)

    # ── Row 5: Speed, Lateral offset ──
    fig.add_trace(go.Scatter(
        x=D["s"], y=D["v"], mode="lines",
        line=dict(width=1.5, color="#e74c3c"), name="v",
    ), row=5, col=1)

    w_left = np.array(sol["w_left"]); w_right = np.array(sol["w_right"])
    fig.add_trace(go.Scatter(
        x=D["s"], y=w_left, mode="lines",
        line=dict(width=0.8, color="gray", dash="dash"), name="w_left",
        showlegend=False,
    ), row=5, col=2)
    fig.add_trace(go.Scatter(
        x=D["s"], y=-w_right, mode="lines",
        line=dict(width=0.8, color="gray", dash="dash"), name="-w_right",
        showlegend=False,
    ), row=5, col=2)
    fig.add_trace(go.Scatter(
        x=D["s"], y=D["d"], mode="lines",
        line=dict(width=1.5, color="black"), name="d (CoG)",
    ), row=5, col=2)

    # ── Vertical cursor lines on strip charts (one per strip subplot) ──
    cursor_traces = []
    strip_positions = [(2,1),(2,2),(3,1),(3,2),(4,1),(4,2),(5,1),(5,2)]
    for r, c in strip_positions:
        yaxis_range = fig.get_subplot(r, c).yaxis.range if hasattr(fig.get_subplot(r, c).yaxis, 'range') else None
        trace = go.Scatter(
            x=[D["s"][0], D["s"][0]], y=[-1e6, 1e6],
            mode="lines", line=dict(color="red", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        )
        fig.add_trace(trace, row=r, col=c)
        cursor_traces.append(len(fig.data) - 1)

    # Car marker trace index
    car_trace_idx = None
    for idx, trace in enumerate(fig.data):
        if trace.name == "Car":
            car_trace_idx = idx
            break

    # ── Slider for scrubbing ──
    stride = max(1, N // 200)  # ~200 slider steps
    slider_indices = list(range(0, N, stride))
    if slider_indices[-1] != N - 1:
        slider_indices.append(N - 1)

    steps = []
    for i in slider_indices:
        # Build update args: move car marker + cursor lines
        updates_x = [[D["x"][i]]]
        updates_y = [[D["y"][i]]]

        args = {
            "x": [[D["x"][i]]] + [[D["s"][i], D["s"][i]]] * len(cursor_traces),
            "y": [[D["y"][i]]] + [[-1e6, 1e6]] * len(cursor_traces),
        }
        trace_indices = [car_trace_idx] + cursor_traces

        step = dict(
            method="restyle",
            args=[args, trace_indices],
            label=f"{D['s'][i]:.0f}",
        )
        steps.append(step)

    slider = dict(
        active=0,
        currentvalue=dict(prefix="s = ", suffix=" m", font=dict(size=13)),
        pad=dict(t=30),
        steps=steps,
    )

    # ── Layout ──
    profiling = sol.get("profiling", {})
    lap_time = profiling.get("lap_time_s", 0)

    fig.update_layout(
        title=dict(
            text=f"Four-Wheel LTO — Maisach Track (lap time: {lap_time:.2f}s)",
            font=dict(size=16),
        ),
        height=1600,
        width=1300,
        sliders=[slider],
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.02,
            xanchor="center", x=0.5, font=dict(size=9),
        ),
        template="plotly_white",
        hovermode="closest",
    )

    # Label shared x-axes
    for r in [2, 3, 4, 5]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="s [m]" if r == 5 else "", row=r, col=c)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True)
    print(f"Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sol_path = REPO / "data" / "solutions" / "track_boundary_maisach_four_wheel_euler.json"
    csv_path = REPO / "data" / "tracks" / "track_boundary_maisach.csv"
    out_path = REPO / "ocp_plots" / "maisach_extras" / "maisach_interactive.html"

    build_interactive(sol_path, out_path, track_csv=csv_path)
