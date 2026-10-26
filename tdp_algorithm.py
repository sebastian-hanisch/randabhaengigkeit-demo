"""Kern: Gauß- und t-Kopula mit Normal-Rändern, bedingte Überschreitungswahrscheinlichkeit (exakt per Quadratur), Randabhängigkeitskoeffizient, Summe und Pooling-Gewinn, Rangschätzer des Randes.

Ohne scipy zur Laufzeit: die t-Verteilungsfunktion (ganzzahlige Freiheitsgrade) steht als endliche Summe (Abramowitz/Stegun 26.7.3 und 26.7.4) vektorisiert da, das Normal-Quantil
als Acklam-Näherung mit einem Newton-Schritt. Zufall, Pearson und Spearman kommen aus den wörtlichen Kopien von pearson-demo (prs_*.py)."""

import math

import numpy as np

import tdp_constants as C
from prs_algorithm import norm_cdf, norm_ppf, pearson_r, spearman_r  # noqa: F401  (Wiederverwendung der Kopie)

_erfc = np.frompyfunc(math.erfc, 1, 1)
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)

# Koeffizienten der Näherung nach Acklam (relativer Fehler unter 1,2·10⁻⁹), danach ein Newton-Schritt
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
_CC = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_P_LOW = 0.02425


# --- Normal- und t-Verteilung, vektorisiert --------------------------------------------------------------------------------------------------

def erfc_vec(x) -> np.ndarray:
    return np.asarray(_erfc(np.asarray(x, dtype=float)), dtype=float)


def norm_cdf_vec(x) -> np.ndarray:
    return 0.5 * erfc_vec(-np.asarray(x, dtype=float) / _SQRT2)


def norm_sf_vec(x) -> np.ndarray:
    return 0.5 * erfc_vec(np.asarray(x, dtype=float) / _SQRT2)


def norm_ppf_vec(p) -> np.ndarray:
    """Quantilfunktion der Standardnormalverteilung für ein Feld von Wahrscheinlichkeiten in (0, 1): Acklam-Näherung, ein Newton-Schritt auf dem unteren Rand (Spiegelung für p > 1/2)."""
    p = np.asarray(p, dtype=float)
    if p.size and (np.min(p) <= 0.0 or np.max(p) >= 1.0):
        raise ValueError("p muss in (0, 1) liegen")
    upper = p > 0.5
    pp = np.where(upper, 1.0 - p, p)
    x = np.empty_like(pp)
    tail = pp < _P_LOW
    mid = ~tail
    q = np.sqrt(-2.0 * np.log(pp[tail]))
    x[tail] = (((((_CC[0] * q + _CC[1]) * q + _CC[2]) * q + _CC[3]) * q + _CC[4]) * q + _CC[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    q = pp[mid] - 0.5
    r = q * q
    x[mid] = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    e = norm_cdf_vec(x) - pp                      # ein Newton-Schritt: aus 10⁻⁹ wird Rundungsgenauigkeit (quadratische Konvergenz)
    x = x - e * _SQRT2PI * np.exp(0.5 * x * x)
    return np.where(upper, -x, x)


def t_cdf(x, nu: int) -> np.ndarray:
    """Verteilungsfunktion der Student-Verteilung mit ganzzahligen Freiheitsgraden nu als endliche Summe in θ = arctan(x/√nu)."""
    if nu < 1 or int(nu) != nu:
        raise ValueError("nu muss eine ganze Zahl ab 1 sein")
    x = np.asarray(x, dtype=float)
    th = np.arctan(x / math.sqrt(nu))
    s, c = np.sin(th), np.cos(th)
    if nu % 2 == 1:
        total = np.zeros_like(x)
        power, coef = c.copy(), 1.0                      # k = 0: cos θ
        for k in range(nu // 2):                         # bei ungeradem nu sind es (nu − 1)/2 Summanden nach dem ersten
            if k > 0:
                coef *= (2.0 * k) / (2.0 * k + 1.0)
                power = power * c * c
            total = total + coef * power
        return 0.5 + (th + s * total) / math.pi
    total = np.ones_like(x)
    power, coef = np.ones_like(x), 1.0
    for k in range(1, nu // 2):
        coef *= (2.0 * k - 1.0) / (2.0 * k)
        power = power * c * c
        total = total + coef * power
    return 0.5 + 0.5 * s * total


def t_sf(x, nu: int) -> np.ndarray:
    return t_cdf(-np.asarray(x, dtype=float), nu)


def t_pdf(x, nu: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    logc = math.lgamma((nu + 1) / 2.0) - math.lgamma(nu / 2.0) - 0.5 * math.log(nu * math.pi)
    return np.exp(logc - 0.5 * (nu + 1) * np.log1p(x * x / nu))


def t_ppf(p: float, nu: int) -> float:
    """Quantil der Student-Verteilung durch Bisektion auf der kleineren Randwahrscheinlichkeit (genau auch bei p nahe 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p muss in (0, 1) liegen")
    if p == 0.5:
        return 0.0
    ps = min(p, 1.0 - p)
    lo, hi = 0.0, 1.0
    while float(t_cdf(-hi, nu)) > ps:
        lo, hi = hi, hi * 2.0
        if hi > 1e15:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(t_cdf(-mid, nu)) > ps:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * hi:
            break
    x = 0.5 * (lo + hi)
    return x if p > 0.5 else -x


# --- Erzeuger ----------------------------------------------------------------------------------------------------------------------------

def _corr_normals(n: int, rho: float, rng) -> tuple[np.ndarray, np.ndarray]:
    z1 = rng.normal(n)
    z2 = rho * z1 + math.sqrt(1.0 - rho * rho) * rng.normal(n)
    return z1, z2


def t_to_normal(t, nu: int) -> np.ndarray:
    """Randtransformation einer t(nu)-verteilten Größe auf eine Standardnormal-Größe (Kopula bleibt): auf dem unteren Rand gerechnet und gespiegelt, damit kleine Wahrscheinlichkeiten genau bleiben."""
    t = np.asarray(t, dtype=float)
    z = norm_ppf_vec(np.maximum(t_cdf(-np.abs(t), nu), 1e-300))
    return np.where(t < 0, z, -z)


def draw(copula: str, n: int, rho: float, nu: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """n Paare mit Standardnormal-Rändern und Gauß-Kopula ("gauss") bzw. t-Kopula mit nu Freiheitsgraden ("t"), Korrelationsparameter rho."""
    if copula not in C.COPULAS:
        raise ValueError(f"unbekannte Kopula: {copula}")
    z1, z2 = _corr_normals(n, rho, rng)
    if copula == "gauss":
        return z1, z2
    w = np.sqrt(nu / (rng.normal((nu, n)) ** 2).sum(axis=0))
    return t_to_normal(z1 * w, nu), t_to_normal(z2 * w, nu)


def draw_both(n: int, rho: float, nu: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gauß- und t-Kopula aus denselben Normalen (gleiche Ränder, gleiches rho); entspricht draw("gauss", …) und draw("t", …) mit gleichem Zufallsstrom."""
    z1, z2 = _corr_normals(n, rho, rng)
    w = np.sqrt(nu / (rng.normal((nu, n)) ** 2).sum(axis=0))
    return z1, z2, t_to_normal(z1 * w, nu), t_to_normal(z2 * w, nu)


# --- Quadratur ---------------------------------------------------------------------------------------------------------------------------

_GL_X, _GL_W = np.polynomial.legendre.leggauss(C.QUAD_NODES)


def integrate_unit(f, edges=C.QUAD_EDGES) -> float:
    """Zusammengesetzte Gauß-Legendre-Regel auf [0, 1) mit den Teilpunkten `edges`; f nimmt ein Feld und gibt ein Feld zurück."""
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        w = 0.5 * (b - a) * _GL_X + 0.5 * (b + a)
        total += 0.5 * (b - a) * float(np.sum(_GL_W * f(w)))
    return total


def _check_level(level: float) -> None:
    if not 0.5 < level < 1.0:
        raise ValueError("das Niveau muss zwischen 0,5 und 1 liegen")


def cond_exceed_gauss(level: float, rho: float) -> float:
    """P(Y > c | X > c) bei Gauß-Kopula, c = Normal-Quantil zum Niveau `level`, exakt per Quadratur (Substitution x = c/(1 − w))."""
    _check_level(level)
    c = norm_ppf(level)
    s = math.sqrt(1.0 - rho * rho)

    def f(w):
        x = c / (1.0 - w)
        return np.exp(-0.5 * x * x) / _SQRT2PI * norm_sf_vec((c - rho * x) / s) * c / (1.0 - w) ** 2

    return integrate_unit(f) / (1.0 - level)


def cond_exceed_t(level: float, rho: float, nu: int) -> float:
    """P(Y > c | X > c) bei t(nu)-Kopula mit Parameter rho und demselben Niveau auf der Kopulaskala, exakt per Quadratur."""
    _check_level(level)
    c = t_ppf(level, nu)
    s = math.sqrt((1.0 - rho * rho) / (nu + 1.0))

    def f(w):
        x = c / (1.0 - w)
        return t_pdf(x, nu) * t_sf((c - rho * x) / (s * np.sqrt(nu + x * x)), nu + 1) * c / (1.0 - w) ** 2

    return integrate_unit(f) / (1.0 - level)


def cond_exceed(copula: str, level: float, rho: float, nu: int = 3) -> float:
    return cond_exceed_gauss(level, rho) if copula == "gauss" else cond_exceed_t(level, rho, nu)


def lambda_t(rho: float, nu: int) -> float:
    """Randabhängigkeitskoeffizient der t-Kopula: 2 · t_{nu+1}(−√((nu + 1)(1 − rho)/(1 + rho)))."""
    return 2.0 * float(t_cdf(-math.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)), nu + 1))


# --- Summe und Pooling ---------------------------------------------------------------------------------------------------------------------

def gauss_sum_quantile(rho: float, p: float) -> float:
    """Quantil der Summe zweier Standardnormalgrößen mit Korrelation rho: Summe ~ N(0, 2 + 2 rho)."""
    return norm_ppf(p) * math.sqrt(2.0 + 2.0 * rho)


def gauss_sum_exceed(rho: float, s: float) -> float:
    return float(norm_sf_vec(s / math.sqrt(2.0 + 2.0 * rho)))


def pooling_gain(sum_quantile: float, p: float) -> float:
    """Gewinn gegen zwei getrennte Bestände: 1 − Quantil der Summe / (2 · Normal-Quantil zum selben Niveau)."""
    return 1.0 - sum_quantile / (2.0 * norm_ppf(p))


def sum_stats(x, y, levels=C.SUM_LEVELS, bound: float = C.SUM_BOUND) -> dict:
    """Summenquantile, Überschreitung einer Schranke und Pooling-Gewinn (beim mittleren Niveau) aus Paaren."""
    s = np.asarray(x, dtype=float) + np.asarray(y, dtype=float)
    q = [float(v) for v in np.quantile(s, levels)]
    return {"quantiles": q, "exceed": float((s > bound).mean()), "gain": pooling_gain(q[len(q) // 2], levels[len(q) // 2])}


# --- Rangmaße und Schätzer des Randes -----------------------------------------------------------------------------------------------------

def ranks(v) -> np.ndarray:
    """Ränge 0 bis n − 1 (ohne Behandlung von Gleichständen: die Daten sind stetig)."""
    return np.argsort(np.argsort(np.asarray(v, dtype=float), kind="stable"), kind="stable")


def tail_rate(x, y, k: int) -> float:
    """Rangschätzer: Anteil der k größten X-Ränge, deren Y-Rang ebenfalls unter den k größten steht (schätzt P(Y groß | X groß) beim Niveau 1 − k/n)."""
    n = len(x)
    if len(y) != n:
        raise ValueError("x und y müssen gleich lang sein")
    if not 1 <= k <= n:
        raise ValueError("k muss zwischen 1 und n liegen")
    return float(np.sum((ranks(x) >= n - k) & (ranks(y) >= n - k)) / k)


def quadrant_count(x, y, c: float) -> tuple[int, int]:
    """Zahl der Paare mit X > c und davon mit zugleich Y > c."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return int((x > c).sum()), int(((x > c) & (y > c)).sum())


def kendall_tau(x, y) -> float:
    """Kendalls τ (ohne Gleichstände): (konkordante − diskordante Paare) / Paare."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < 2 or len(y) != n:
        raise ValueError("mindestens zwei gleich lange Reihen nötig")
    total = 0.0
    for i in range(n - 1):
        total += float(np.sum(np.sign(x[i + 1:] - x[i]) * np.sign(y[i + 1:] - y[i])))
    return 2.0 * total / (n * (n - 1))


def tau_from_rho(rho: float) -> float:
    """Kendalls τ jeder elliptischen Kopula mit Parameter rho: 2/π · arcsin(rho)."""
    return 2.0 / math.pi * math.asin(rho)


def tail_k(n: int, share: float) -> int:
    """Zahl k der Wochen am oberen Rand bei einem Anteil `share` (mindestens 3, höchstens n)."""
    return min(n, max(3, int(round(n * share))))
