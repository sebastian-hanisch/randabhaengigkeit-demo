"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import tdp_constants as C

AXIS_LIM = 5.0


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="top", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _level_text(q: float) -> str:
    return f"{100 * q:.3f}".rstrip("0").rstrip(".").replace(".", ",") + " %"


def exponent(q: float) -> float:
    """Niveau q als Zehnerexponent e mit q = 1 − 10^(−e) (0,99 -> 2)."""
    return -math.log10(1.0 - q)


def build_scatter(x, y, c: float, color: str, name: str, height: int = 360):
    """Streudiagramm der normierten Nachfrage zweier Standorte; Wochen, in denen beide über der Schwelle c liegen, sind hervorgehoben, die Schwelle ist als Gerade eingezeichnet."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    both = (x > c) & (y > c)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[~both].tolist(), y=y[~both].tolist(), mode="markers", name="übrige Wochen", marker=dict(size=4, color=color, opacity=0.35), hovertemplate="A %{x:.2f}, B %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x[both].tolist(), y=y[both].tolist(), mode="markers", name="beide über der Schwelle", marker=dict(size=9, color="#d62728", line=dict(width=1, color="#ffffff")),
                             hovertemplate="A %{x:.2f}, B %{y:.2f}<extra>beide oben</extra>"))
    fig.add_vline(x=c, line=dict(color=C.COLOR_EDGE, dash="dash", width=1.2))
    fig.add_hline(y=c, line=dict(color=C.COLOR_EDGE, dash="dash", width=1.2))
    fig.update_xaxes(title_text="Standort A (normiert)", range=[-AXIS_LIM, AXIS_LIM])
    fig.update_yaxes(title_text="Standort B (normiert)", range=[-AXIS_LIM, AXIS_LIM], scaleanchor="x")
    fig.update_layout(title=dict(text=name, x=0.5, font=dict(size=14)))
    fig = _base(fig, height)
    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
    return fig


def _level_axis(fig, levels: list[float]):
    ticks = [e for e in (1, 2, 3, 4, 5)]
    fig.update_xaxes(title_text="Niveau q der Schwelle (Quantil der Ränder)", tickvals=ticks, ticktext=[_level_text(1 - 10.0 ** (-e)) for e in ticks], range=[exponent(levels[0]) - 0.15, exponent(levels[-1]) + 0.15])


def build_cond_curves(curves: dict, nus: tuple[int, ...], mark_level: float | None = None, height: int = 380):
    """P(Y > q | X > q) gegen das Niveau q: Gauß-Kopula und t-Kopulas, mit λ als gepunkteter Waagerechten und dem gewählten Niveau als Senkrechten."""
    xs = [exponent(q) for q in curves["levels"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=curves["gauss"], mode="lines+markers", name="Gauß-Kopula", line=dict(color=C.COLOR_GAUSS, width=3.5), hovertemplate="Niveau %{customdata}: %{y:.3f}<extra>Gauß</extra>",
                             customdata=[_level_text(q) for q in curves["levels"]]))
    for nu in nus:
        color = C.NU_COLORS.get(nu, C.COLOR_T)
        fig.add_trace(go.Scatter(x=xs, y=curves["t"][nu], mode="lines+markers", name=f"t-Kopula, ν = {nu}", line=dict(color=color, width=3), hovertemplate="Niveau %{customdata}: %{y:.3f}<extra>t(" + str(nu) + ")</extra>",
                                 customdata=[_level_text(q) for q in curves["levels"]]))
        fig.add_trace(go.Scatter(x=[xs[0], xs[-1]], y=[curves["lambda"][nu]] * 2, mode="lines", name=f"λ, ν = {nu}", line=dict(color=color, width=1.2, dash="dot"), showlegend=False, hoverinfo="skip"))
    if mark_level is not None:
        fig.add_vline(x=exponent(mark_level), line=dict(color=C.COLOR_EDGE, width=1.5, dash="dash"))
    _level_axis(fig, curves["levels"])
    fig.update_yaxes(title_text="P(B oben | A oben)", range=[0, 1], rangemode="tozero")
    return _base(fig, height)


def build_gain(rows: list[dict], nu: int, height: int = 340):
    """Pooling-Gewinn gegen ρ: Gauß-Kopula und t-Kopula mit nu Freiheitsgraden (Quantil der Summe zum mittleren Niveau)."""
    sub = sorted((r for r in rows if r["nu"] == nu), key=lambda r: r["rho"])
    rhos = [r["rho"] for r in sub]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rhos, y=[100 * r["gauss"]["gain"] for r in sub], mode="lines+markers", name="Gauß-Kopula", line=dict(color=C.COLOR_GAUSS, width=3), hovertemplate="ρ = %{x}: %{y:.1f} %<extra>Gauß</extra>"))
    fig.add_trace(go.Scatter(x=rhos, y=[100 * r["t"]["gain"] for r in sub], mode="lines+markers", name=f"t-Kopula, ν = {nu}", line=dict(color=C.COLOR_T, width=3), hovertemplate="ρ = %{x}: %{y:.1f} %<extra>t</extra>"))
    fig.update_xaxes(title_text="Korrelationsparameter ρ")
    fig.update_yaxes(title_text="Pooling-Gewinn (%)", rangemode="tozero")
    return _base(fig, height)


def build_exceed(rows: list[dict], nu: int, height: int = 340):
    """Wahrscheinlichkeit, dass die Summe die Schranke überschreitet, gegen ρ (logarithmisch)."""
    sub = sorted((r for r in rows if r["nu"] == nu and r["rho"] > 0), key=lambda r: r["rho"])
    rhos = [r["rho"] for r in sub]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rhos, y=[r["gauss"]["exceed"] for r in sub], mode="lines+markers", name="Gauß-Kopula", line=dict(color=C.COLOR_GAUSS, width=3), hovertemplate="ρ = %{x}: %{y:.5f}<extra>Gauß</extra>"))
    fig.add_trace(go.Scatter(x=rhos, y=[r["t"]["exceed"] for r in sub], mode="lines+markers", name=f"t-Kopula, ν = {nu}", line=dict(color=C.COLOR_T, width=3), hovertemplate="ρ = %{x}: %{y:.5f}<extra>t</extra>"))
    fig.update_xaxes(title_text="Korrelationsparameter ρ")
    fig.update_yaxes(title_text=f"P(Summe > {C.SUM_BOUND:g})", type="log")
    return _base(fig, height)


def build_rank_bars(cell: dict, height: int = 320):
    """Pearson, Spearman und Kendall beider Kopulas bei gleichem ρ (gruppierte Balken; die Unterschiede sind klein)."""
    names = [("pearson", "Pearson"), ("spearman", "Spearman"), ("kendall", "Kendall")]
    fig = go.Figure()
    for copula, label in (("gauss", "Gauß-Kopula"), ("t", f"t-Kopula, ν = {cell['nu']}")):
        vals = [cell[copula][k] for k, _ in names]
        fig.add_trace(go.Bar(x=[n for _, n in names], y=vals, name=label, marker_color=C.COPULA_COLORS[copula], text=[f"{v:.3f}".replace(".", ",") for v in vals], textposition="outside",
                             hovertemplate="%{x}: %{y:.3f}<extra>" + label + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Maß", range=[0, 0.8])
    return _base(fig, height)


def build_estimate(rows: list[dict], share: float, height: int = 340):
    """Rangschätzer aus n Wochen: Mittel ± Streuung für Daten der Gauß- und der t-Kopula, mit den exakten Werten (Rauten) beim Niveau 1 − k/n."""
    sub = sorted((r for r in rows if r["share"] == share), key=lambda r: r["n"])
    labels = [f"{r['n']} Wochen" for r in sub]
    fig = go.Figure()
    for copula, label in (("gauss", "Daten der Gauß-Kopula"), ("t", "Daten der t-Kopula")):
        fig.add_trace(go.Scatter(x=labels, y=[r["mean"][copula] for r in sub], mode="markers", name=label, marker=dict(size=11, color=C.COPULA_COLORS[copula]),
                                 error_y=dict(type="data", array=[r["sd"][copula] for r in sub], visible=True, thickness=2, width=6), hovertemplate="%{x}: %{y:.2f}<extra>" + label + "</extra>"))
        fig.add_trace(go.Scatter(x=labels, y=[r["exact"][copula] for r in sub], mode="markers", name="exakt", marker=dict(size=12, symbol="diamond-open", color=C.COPULA_COLORS[copula], line=dict(width=2)),
                                 showlegend=False, hovertemplate="%{x}: exakt %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text=f"Länge der Reihe (Anteil am Rand {100 * share:g} %)", type="category")
    fig.update_yaxes(title_text="Rangschätzer (Mittel ± Streuung)", rangemode="tozero")
    return _base(fig, height)


def build_accuracy(rows: list[dict], height: int = 340):
    """Anteil richtiger Zuordnungen (Gauß oder t) gegen die Länge der Reihe, je ein Anteil am Rand."""
    fig = go.Figure()
    shares = sorted({r["share"] for r in rows})
    palette = {s: c for s, c in zip(shares, ("#9d9d9d", "#4c78a8", "#e45756"))}
    for share in shares:
        sub = sorted((r for r in rows if r["share"] == share), key=lambda r: r["n"])
        fig.add_trace(go.Scatter(x=[r["n"] for r in sub], y=[100 * r["accuracy"] for r in sub], mode="lines+markers", name=f"{100 * share:g} % der Wochen am Rand", line=dict(color=palette[share], width=3),
                                 hovertemplate="%{x} Wochen: %{y:.1f} %<extra>" + f"{100 * share:g} %" + "</extra>"))
    ns = sorted({r["n"] for r in rows})
    fig.add_hline(y=50, line=dict(color=C.COLOR_EDGE, dash="dot", width=1.2), annotation_text="Raten", annotation_position="bottom right")
    fig.update_xaxes(title_text="Länge der Reihe in Wochen", type="log", tickvals=ns, ticktext=[str(n) for n in ns])
    fig.update_yaxes(title_text="richtig erkannt (%)", range=[40, 101])
    return _base(fig, height)


def build_nu_curves(table: dict, height: int = 340):
    """Bedingte Wahrscheinlichkeit der t-Kopula gegen ν bei einigen Niveaus (durchgezogen), λ (gepunktet) und die Gauß-Kopula (Waagerechte je Niveau)."""
    nus = [r["nu"] for r in table["rows"]]
    palette = ("#e45756", "#f58518", "#54a24b")
    fig = go.Figure()
    for i, q in enumerate(table["levels"]):
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(x=nus, y=[r["cond"][i] for r in table["rows"]], mode="lines+markers", name=f"t-Kopula, Niveau {_level_text(q)}", line=dict(color=color, width=3),
                                 hovertemplate="ν = %{x}: %{y:.3f}<extra>Niveau " + _level_text(q) + "</extra>"))
        fig.add_trace(go.Scatter(x=[nus[0], nus[-1]], y=[table["gauss"][i]] * 2, mode="lines", name=f"Gauß, Niveau {_level_text(q)}", line=dict(color=color, width=1.5, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=nus, y=[r["lambda"] for r in table["rows"]], mode="lines+markers", name="λ der t-Kopula", line=dict(color=C.COLOR_EDGE, width=2, dash="dot"), hovertemplate="ν = %{x}: λ = %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Freiheitsgrade ν", tickvals=nus)
    fig.update_yaxes(title_text="P(B oben | A oben)", rangemode="tozero")
    return _base(fig, height)
