"""Auswertung: Einzeldatensatz, exakte Kurven, Zellen des vorgerechneten Rasters, Zugriff auf das Raster und alle Verdikte als reine Funktionen."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import prs_scenario as S
import tdp_algorithm as K
import tdp_constants as C

GRID_PATH = Path(__file__).resolve().parent / "data" / "tail_grid.json"


# --- Einzeldatensatz -------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Settings:
    rho: float = C.DEFAULT_RHO
    nu: int = C.DEFAULT_NU
    n: int = C.DEFAULT_N
    level: float = C.DEFAULT_LEVEL
    share: int = C.DEFAULT_SHARE
    seed: int = C.DEFAULT_SEED

    def key(self) -> tuple:
        return (self.rho, self.nu, self.n, self.level, self.share, self.seed)

    def __eq__(self, other) -> bool:
        return isinstance(other, Settings) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())


@dataclass(frozen=True, eq=False)
class Analysis:
    settings: Settings
    xg: np.ndarray
    yg: np.ndarray
    xt: np.ndarray
    yt: np.ndarray
    c: float                   # Normal-Quantil zum Niveau
    quadrant: dict             # Kopula -> (Wochen mit X > c, davon mit Y > c)
    exact: dict                # Kopula -> P(Y > c | X > c), exakt
    lam: float                 # Randabhängigkeitskoeffizient der t-Kopula
    measures: dict             # Kopula -> {"pearson", "spearman", "kendall"}
    k: int                     # Wochen am Rand für den Rangschätzer
    tail_level: float          # 1 − k/n
    tail_est: dict             # Kopula -> Rangschätzer
    tail_exact: dict           # Kopula -> exakte bedingte Wahrscheinlichkeit beim Niveau 1 − k/n
    sums: dict                 # Kopula -> {"quantiles", "exceed", "gain"}


def analyse(s: Settings) -> Analysis:
    data_seed, sum_seed = S.substream_seeds(s.seed, 2)
    xg, yg, xt, yt = K.draw_both(s.n, s.rho, s.nu, S.VRng(data_seed))
    c = K.norm_ppf(s.level)
    quadrant = {"gauss": K.quadrant_count(xg, yg, c), "t": K.quadrant_count(xt, yt, c)}
    exact = {"gauss": K.cond_exceed_gauss(s.level, s.rho), "t": K.cond_exceed_t(s.level, s.rho, s.nu)}
    measures = {name: {"pearson": K.pearson_r(x, y), "spearman": K.spearman_r(x, y), "kendall": K.kendall_tau(x, y)} for name, (x, y) in (("gauss", (xg, yg)), ("t", (xt, yt)))}
    k = K.tail_k(s.n, s.share / 100.0)
    tail_level = 1.0 - k / s.n
    tail_exact = {"gauss": K.cond_exceed_gauss(tail_level, s.rho), "t": K.cond_exceed_t(tail_level, s.rho, s.nu)}
    st, yt_sum = K.draw("t", C.LIVE_SUM_PAIRS, s.rho, s.nu, S.VRng(sum_seed))
    sums = {"gauss": gauss_sums(s.rho), "t": K.sum_stats(st, yt_sum)}
    return Analysis(s, xg, yg, xt, yt, c, quadrant, exact, K.lambda_t(s.rho, s.nu), measures, k, tail_level,
                    {"gauss": K.tail_rate(xg, yg, k), "t": K.tail_rate(xt, yt, k)}, tail_exact, sums)


def gauss_sums(rho: float) -> dict:
    """Summenquantile, Überschreitung der Schranke und Pooling-Gewinn der Gauß-Kopula, exakt (Summe ~ N(0, 2 + 2 rho))."""
    q = [K.gauss_sum_quantile(rho, p) for p in C.SUM_LEVELS]
    mid = len(C.SUM_LEVELS) // 2
    return {"quantiles": q, "exceed": K.gauss_sum_exceed(rho, C.SUM_BOUND), "gain": K.pooling_gain(q[mid], C.SUM_LEVELS[mid])}


# --- Exakte Kurven (live, Quadratur) -------------------------------------------------------------------------------------------------------

CURVE_EXPONENTS = tuple(np.round(np.arange(1.0, 5.0001, 0.25), 2))          # Niveau = 1 − 10^(−e), von 0,9 bis 0,99999


def curve_levels() -> list[float]:
    return [1.0 - 10.0 ** (-float(e)) for e in CURVE_EXPONENTS]


def cond_curves(rho: float, nus: tuple[int, ...] = C.COND_NUS) -> dict:
    """E1: exakte bedingte Wahrscheinlichkeit gegen das Niveau für die Gauß-Kopula und t-Kopulas mit den gegebenen Freiheitsgraden."""
    levels = curve_levels()
    out = {"levels": levels, "gauss": [K.cond_exceed_gauss(q, rho) for q in levels]}
    out["t"] = {nu: [K.cond_exceed_t(q, rho, nu) for q in levels] for nu in nus}
    out["lambda"] = {nu: K.lambda_t(rho, nu) for nu in nus}
    return out


def nu_table(rho: float, nus: tuple[int, ...] = (3, 4, 5, 7, 10, 15, 20, 30), levels: tuple[float, ...] = (0.99, 0.999, 0.9999)) -> dict:
    """E5: bedingte Wahrscheinlichkeit der t-Kopula bei einigen Niveaus und λ für wachsende Freiheitsgrade, daneben die Gauß-Kopula."""
    return {"levels": list(levels), "gauss": [K.cond_exceed_gauss(q, rho) for q in levels],
            "rows": [{"nu": nu, "cond": [K.cond_exceed_t(q, rho, nu) for q in levels], "lambda": K.lambda_t(rho, nu)} for nu in nus]}


# --- Zellen des vorgerechneten Rasters -------------------------------------------------------------------------------------------------------

def cell_seed(kind: str, rho: float, nu: int, s: int, extra: int = 0) -> int:
    """Eigener Seed je Zelle und Plan-Seed s (verschieden für Art, rho, nu und eine Zusatzzahl)."""
    kinds = ("sum", "rank", "est", "probe")
    return s * 1_000_003 + int(round(rho * 100)) * 7919 + nu * 1013 + extra * 101 + kinds.index(kind) * 100_000_007


def sum_cell(rho: float, nu: int, pairs: int = C.SUM_PAIRS, seeds=C.GRID_SEEDS) -> dict:
    """E2: Summenquantile, Überschreitung der Schranke und Pooling-Gewinn der t-Kopula (Mittel über die Seeds mit je `pairs` Paaren) neben den exakten Zahlen der Gauß-Kopula."""
    quantiles, exceed = [], []
    for s in seeds:
        x, y = K.draw("t", pairs, rho, nu, S.VRng(cell_seed("sum", rho, nu, s)))
        st = K.sum_stats(x, y)
        quantiles.append(st["quantiles"])
        exceed.append(st["exceed"])
    q = np.mean(quantiles, axis=0)
    mid = len(C.SUM_LEVELS) // 2
    return {"rho": rho, "nu": nu, "pairs": pairs * len(seeds), "gauss": gauss_sums(rho),
            "t": {"quantiles": [float(v) for v in q], "exceed": float(np.mean(exceed)), "gain": K.pooling_gain(float(q[mid]), C.SUM_LEVELS[mid])}}


def rank_cell(rho: float, nu: int, pairs: int = C.RANK_PAIRS, tau_pairs: int = C.RANK_TAU_PAIRS, seeds=C.GRID_SEEDS) -> dict:
    """E3: Pearson, Spearman und Kendall beider Kopulas bei gleichem rho (Mittel über die Seeds; Kendall aus den ersten `tau_pairs` Paaren)."""
    out = {"rho": rho, "nu": nu, "pairs": pairs * len(seeds), "tau_formula": K.tau_from_rho(rho)}
    for copula in C.COPULAS:
        rows = []
        for s in seeds:
            x, y = K.draw(copula, pairs, rho, nu, S.VRng(cell_seed("rank", rho, nu, s)))
            rows.append([K.pearson_r(x, y), K.spearman_r(x, y), K.kendall_tau(x[:tau_pairs], y[:tau_pairs])])
        m = np.mean(rows, axis=0)
        out[copula] = {"pearson": float(m[0]), "spearman": float(m[1]), "kendall": float(m[2])}
    return out


def estimate_cell(n: int, share: float, rho: float = C.EST_RHO, nu: int = C.EST_NU, runs_per_seed: int = C.EST_RUNS, seeds=C.GRID_SEEDS) -> dict:
    """E4: Rangschätzer des Randes aus n Wochen beider Kopulas: Mittel und Streuung über die Datensätze, exakte Werte beim Niveau 1 − k/n und Anteil richtiger Zuordnungen
    (Schwelle in der Mitte der beiden exakten Werte; ausgewogen über beide Kopulas)."""
    k = K.tail_k(n, share)
    level = 1.0 - k / n
    exact = {"gauss": K.cond_exceed_gauss(level, rho), "t": K.cond_exceed_t(level, rho, nu)}
    vals = {"gauss": [], "t": []}
    for s in seeds:
        rng = S.VRng(cell_seed("est", rho, nu, s, extra=n + int(round(share * 1000))))
        for _ in range(runs_per_seed):
            xg, yg, xt, yt = K.draw_both(n, rho, nu, rng)
            vals["gauss"].append(K.tail_rate(xg, yg, k))
            vals["t"].append(K.tail_rate(xt, yt, k))
    g, t = np.array(vals["gauss"]), np.array(vals["t"])
    mid = 0.5 * (exact["gauss"] + exact["t"])
    return {"n": n, "share": share, "k": k, "level": level, "runs": len(g), "exact": exact, "mid": mid,
            "mean": {"gauss": float(g.mean()), "t": float(t.mean())}, "sd": {"gauss": float(g.std(ddof=1)), "t": float(t.std(ddof=1))},
            "correct": {"gauss": float((g <= mid).mean()), "t": float((t > mid).mean())}, "accuracy": float(0.5 * ((g <= mid).mean() + (t > mid).mean()))}


def grid_probe() -> dict:
    """Kleine feste Zellen und ein fester Einzeldatensatz, die der Test nachrechnet: ändert sich Erzeuger oder Schätzer, passen sie nicht mehr zum gespeicherten Raster."""
    a = analyse(Settings(0.5, 3, 200, 0.99, 5, 11))
    return {"sum": sum_cell(0.5, 3, 20_000, seeds=(7, 8)), "rank": rank_cell(0.5, 5, 5_000, 1_000, seeds=(7,)), "estimate": estimate_cell(120, 0.05, runs_per_seed=30, seeds=(7, 8)),
            "quadrant": {k: list(v) for k, v in a.quadrant.items()}, "exact": a.exact, "measures": a.measures, "tail_est": a.tail_est, "k": a.k}


# --- Zugriff auf das Raster ------------------------------------------------------------------------------------------------------------------

def load_grid(path: Path = GRID_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find(rows: list[dict], **keys) -> dict | None:
    for row in rows:
        if all(row.get(k) == v or (isinstance(v, float) and isinstance(row.get(k), (int, float)) and abs(row[k] - v) < 1e-9) for k, v in keys.items()):
            return row
    return None


# --- Verdikte (reine Funktionen: Art, Text) -----------------------------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{100 * v:.1f} %".replace(".", ",")


def _num(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def _level(q: float) -> str:
    return f"{100 * q:.3f}".rstrip("0").rstrip(".").replace(".", ",") + " %"


def _int(v: int) -> str:
    return f"{v:,}".replace(",", " ")


def _ratio(a: float, b: float) -> str:
    return "unbestimmt" if b <= 0 else "das " + f"{a / b:.1f}".replace(".", ",") + "-Fache"


def verdict_main(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: gleiche Korrelationszahl, gleiche Ränder, wie oft stehen beide Standorte zugleich oben, und was sagt die exakte Rechnung?"""
    s = a.settings
    (og, bg), (ot, bt) = a.quadrant["gauss"], a.quadrant["t"]
    eg, et = a.exact["gauss"], a.exact["t"]
    text = (f"Beim Niveau {_level(s.level)} (Schwelle {_num(a.c, 2)}) liegt die bedingte Wahrscheinlichkeit, dass der zweite Standort auch oben steht, wenn der erste oben steht, "
            f"exakt bei **{_num(eg)}** (Gauß-Kopula) und **{_num(et)}** (t-Kopula, ν = {s.nu}), also {_ratio(et, eg)}. "
            f"In diesen {s.n} Wochen standen {og} bzw. {ot} Wochen beim ersten Standort oben, davon beim zweiten ebenfalls {bg} bzw. {bt}.")
    if min(og, ot) < 10:
        text += " Die Zählung ist bei so wenigen Wochen oben nur grob: die exakte Zahl ist die belastbare."
    return ("warning" if et > 1.5 * eg else "info"), text


def verdict_measures(a: Analysis) -> tuple[str, str]:
    """Rangmaße: dieselbe Funktion von rho bei Kendall, kleine Abweichung bei Pearson und Spearman."""
    m = a.measures
    text = (f"Gauß-Kopula: Pearson **{_num(m['gauss']['pearson'])}**, Spearman **{_num(m['gauss']['spearman'])}**, Kendall **{_num(m['gauss']['kendall'])}**; "
            f"t-Kopula: Pearson **{_num(m['t']['pearson'])}**, Spearman **{_num(m['t']['spearman'])}**, Kendall **{_num(m['t']['kendall'])}**. "
            f"Kendalls τ ist bei jeder elliptischen Kopula dieselbe Funktion von ρ ({_num(K.tau_from_rho(a.settings.rho))}); keine der drei Zahlen zeigt den Rand.")
    return "info", text


def verdict_sums(a: Analysis) -> tuple[str, str]:
    """Summe beider Standorte: Überschreitung der Schranke und Pooling-Gewinn bei gleicher Korrelationszahl."""
    g, t = a.sums["gauss"], a.sums["t"]
    text = (f"Summe beider Standorte: Schranke {C.SUM_BOUND:g} wird bei der Gauß-Kopula mit Wahrscheinlichkeit **{_num(g['exceed'], 5)}** überschritten, bei der t-Kopula mit **{_num(t['exceed'], 5)}** "
            f"({_ratio(t['exceed'], g['exceed'])}). Das Quantil der Summe zum Niveau {_level(C.SUM_LEVELS[1])} ist {_num(g['quantiles'][1], 2)} gegen {_num(t['quantiles'][1], 2)}; "
            f"der Pooling-Gewinn gegen getrennte Bestände sinkt von **{_pct(g['gain'])}** auf **{_pct(t['gain'])}** (t-Werte aus {_int(C.LIVE_SUM_PAIRS)} simulierten Paaren, daher mit Rauschen von etwa ± 1 Prozentpunkt beim Pooling-Gewinn).")
    return "warning", text


def verdict_tail(a: Analysis) -> tuple[str, str]:
    """Rangschätzer des Randes aus den Daten dieses Laufs gegen die exakten Werte beim Niveau 1 − k/n."""
    s = a.settings
    text = (f"Rangschätzer mit k = {a.k} Wochen am Rand (Niveau {_level(a.tail_level)}): Gauß-Daten **{_num(a.tail_est['gauss'], 2)}**, t-Daten **{_num(a.tail_est['t'], 2)}**. "
            f"Exakt beim selben Niveau: {_num(a.tail_exact['gauss'], 2)} bzw. {_num(a.tail_exact['t'], 2)}; der Randabhängigkeitskoeffizient λ der t-Kopula (ν = {s.nu}) ist {_num(a.lam)}. "
            "Der Schätzer misst die bedingte Wahrscheinlichkeit beim gewählten Niveau, nicht λ.")
    return "info", text


def verdict_curves(curves: dict, nu: int) -> tuple[str, str]:
    """E1: Verhältnis der bedingten Wahrscheinlichkeiten beim höchsten Niveau und Abfall der Gauß-Kurve."""
    g, t = curves["gauss"], curves["t"][nu]
    q = curves["levels"]
    mid = min(range(len(q)), key=lambda i: abs(q[i] - 0.999))
    text = (f"Bei Niveau {_level(q[mid])}: Gauß **{_num(g[mid])}**, t(ν = {nu}) **{_num(t[mid])}**, also {_ratio(t[mid], g[mid])}. "
            f"Die Gauß-Kurve fällt von {_num(g[0])} (Niveau {_level(q[0])}) auf {_num(g[-1])} (Niveau {_level(q[-1])}) und erreicht null erst im Grenzwert; "
            f"die t-Kurve fällt von {_num(t[0])} auf {_num(t[-1])} und nähert sich λ = **{_num(curves['lambda'][nu])}** von oben.")
    return "warning", text


def verdict_sum_grid(rows: list[dict], rho: float, nu: int) -> tuple[str, str]:
    """E2: Überschreitung der Schranke und Pooling-Gewinn für eine Zelle des Rasters."""
    c = find(rows, rho=rho, nu=nu)
    g, t = c["gauss"], c["t"]
    return "warning", (f"ρ = {_num(rho, 1)}, t-Kopula mit ν = {nu}: Schranke {C.SUM_BOUND:g} wird mit **{_num(t['exceed'], 5)}** überschritten (Gauß: {_num(g['exceed'], 5)}, {_ratio(t['exceed'], g['exceed'])}); "
                       f"Pooling-Gewinn **{_pct(t['gain'])}** gegen {_pct(g['gain'])}; Quantil der Summe zum Niveau {_level(C.SUM_LEVELS[2])}: {_num(t['quantiles'][2], 2)} gegen {_num(g['quantiles'][2], 2)}.")


def verdict_rank_grid(rows: list[dict], rho: float) -> tuple[str, str]:
    """E3: Rangmaße beider Kopulas bei gleichem rho; Kendall gleich, Pearson und Spearman bei der t-Kopula etwas kleiner."""
    cells = sorted((r for r in rows if r["rho"] == rho), key=lambda r: r["nu"])
    low = cells[0]
    text = (f"ρ = {_num(rho, 1)}, t-Kopula mit ν = {low['nu']}: Pearson {_num(low['t']['pearson'])} gegen {_num(low['gauss']['pearson'])} (Gauß), Spearman {_num(low['t']['spearman'])} gegen {_num(low['gauss']['spearman'])}, "
            f"Kendall {_num(low['t']['kendall'])} gegen {_num(low['gauss']['kendall'])} (Formel {_num(low['tau_formula'])}). "
            "Die Zahlen unterscheiden sich um wenige Hundertstel; **bei der Kopula mit dem größeren Randrisiko sind Pearson und Spearman kleiner, Kendall ist gleich**.")
    return "info", text


def verdict_estimate(rows: list[dict], n: int, share: float) -> tuple[str, str]:
    """E4: Mittel und Streuung des Rangschätzers beider Kopulas für n Wochen und einen Anteil am Rand, mit dem Anteil richtiger Zuordnungen."""
    c = find(rows, n=n, share=share)
    text = (f"n = {n} Wochen ({n / 52:.0f} Jahre), k = {c['k']}: Rangschätzer Gauß-Daten **{_num(c['mean']['gauss'], 2)}** ± {_num(c['sd']['gauss'], 2)}, t-Daten **{_num(c['mean']['t'], 2)}** ± {_num(c['sd']['t'], 2)} "
            f"(exakt {_num(c['exact']['gauss'], 2)} und {_num(c['exact']['t'], 2)}). Die richtige Kopula wird in **{_pct(c['accuracy'])}** der Fälle erkannt (50 % wäre Raten).")
    return ("warning" if c["accuracy"] < 0.8 else "success"), text


def verdict_nu(table: dict) -> tuple[str, str]:
    """E5: kleines λ heißt nicht kleines Risiko bei erreichbaren Niveaus: bedingte Wahrscheinlichkeit der Kopula mit den meisten Freiheitsgraden gegen Gauß."""
    last = table["rows"][-1]
    mid = 1 if len(table["levels"]) > 1 else 0
    return "warning", (f"ν = {last['nu']}: λ ist nur **{_num(last['lambda'], 4)}**, aber beim Niveau {_level(table['levels'][mid])} liegt die bedingte Wahrscheinlichkeit bei **{_num(last['cond'][mid])}** "
                       f"gegen {_num(table['gauss'][mid])} bei der Gauß-Kopula ({_ratio(last['cond'][mid], table['gauss'][mid])}).")
