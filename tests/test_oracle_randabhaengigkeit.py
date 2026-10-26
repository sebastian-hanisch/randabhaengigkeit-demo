"""Orakel (ORACLE-PLAYBOOK): jedes Verfahren gegen einen anderen Rechenweg.

* t-Verteilungsfunktion, -Dichte und -Quantil gegen `scipy.stats.t`, Normal-Quantil gegen `scipy.stats.norm`.
* Bedingte Wahrscheinlichkeit der Gauß-Kopula gegen `scipy.stats.multivariate_normal.cdf` (Inklusion–Exklusion) und gegen `scipy.integrate.quad`;
  der t-Fall gegen `scipy.stats.multivariate_t.cdf`, gegen `quad` und gegen eine Simulation (Brute-Force-Zählung des Quadranten).
* Randabhängigkeitskoeffizient als Grenzwert der bedingten Wahrscheinlichkeit beim Niveau gegen 1.
* Kendalls τ und Spearman gegen scipy, Rangschätzer gegen eine Schleifenfassung, Summenquantile gegen die Formel (Gauß) und eine unabhängige Simulation.
Die Bibliotheken stehen nur im Test-venv; das Orakel braucht sie nicht zur Laufzeit."""

import math

import numpy as np
import pytest

import prs_scenario as S
import tdp_algorithm as K

scipy_stats = pytest.importorskip("scipy.stats")
scipy_integrate = pytest.importorskip("scipy.integrate")


@pytest.mark.parametrize("nu", [1, 2, 3, 4, 5, 10, 11, 30])
def test_t_distribution_matches_scipy(nu):
    x = np.linspace(-30, 30, 1201)
    assert K.t_cdf(x, nu) == pytest.approx(scipy_stats.t.cdf(x, nu), abs=1e-12)
    assert K.t_pdf(x, nu) == pytest.approx(scipy_stats.t.pdf(x, nu), abs=1e-12)
    tail = -np.logspace(0, 2, 30)
    keep = scipy_stats.t.cdf(tail, nu) > 1e-9          # weiter draußen ist die Summenform durch Auslöschung nur noch absolut genau
    assert K.t_cdf(tail, nu)[keep] == pytest.approx(scipy_stats.t.cdf(tail, nu)[keep], rel=1e-6)


@pytest.mark.parametrize("nu", [3, 10])
@pytest.mark.parametrize("p", [0.6, 0.9, 0.99, 0.999, 0.99999, 0.2, 0.001])
def test_t_ppf_matches_scipy(nu, p):
    assert K.t_ppf(p, nu) == pytest.approx(scipy_stats.t.ppf(p, nu), rel=1e-9)


def test_norm_ppf_vec_matches_scipy_over_the_whole_range():
    p = np.concatenate([np.logspace(-14, -1, 60), np.linspace(0.02, 0.98, 100), 1 - np.logspace(-12, -1, 50)])
    p = p[(p > 0) & (p < 1)]
    assert K.norm_ppf_vec(p) == pytest.approx(scipy_stats.norm.ppf(p), abs=1e-11, rel=1e-11)
    assert K.norm_cdf_vec(np.linspace(-8, 8, 50)) == pytest.approx(scipy_stats.norm.cdf(np.linspace(-8, 8, 50)), abs=1e-15, rel=1e-12)


def test_the_scalar_normal_functions_of_the_copy_match_the_vector_versions():
    for p in (0.001, 0.3, 0.5, 0.97, 0.99999):
        assert K.norm_ppf(p) == pytest.approx(float(K.norm_ppf_vec([p])[0]), abs=1e-12)


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.9])
@pytest.mark.parametrize("level", [0.9, 0.99, 0.999, 0.99999])
def test_gauss_conditional_matches_the_bivariate_normal_cdf(rho, level):
    """P(X > c, Y > c) = 1 − 2Φ(c) + C(c, c); geteilt durch P(X > c) = 1 − level."""
    c = scipy_stats.norm.ppf(level)
    both = 1.0 - 2.0 * level + scipy_stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([c, c])
    assert K.cond_exceed_gauss(level, rho) == pytest.approx(both / (1.0 - level), rel=2e-3 if level > 0.9999 else 1e-5)


@pytest.mark.parametrize("rho", [0.3, 0.7])
@pytest.mark.parametrize("level", [0.95, 0.999])
def test_gauss_conditional_matches_scipy_quad(rho, level):
    c = scipy_stats.norm.ppf(level)
    s = math.sqrt(1 - rho * rho)
    num, _ = scipy_integrate.quad(lambda x: scipy_stats.norm.pdf(x) * scipy_stats.norm.sf((c - rho * x) / s), c, np.inf, epsabs=1e-14, epsrel=1e-12)
    assert K.cond_exceed_gauss(level, rho) == pytest.approx(num / (1 - level), rel=1e-9)


@pytest.mark.parametrize("nu", [3, 5, 10, 30])
@pytest.mark.parametrize("rho", [0.3, 0.5, 0.9])
@pytest.mark.parametrize("level", [0.9, 0.99, 0.999, 0.99999])
def test_t_conditional_matches_scipy_quad(nu, rho, level):
    c = scipy_stats.t.ppf(level, nu)
    s = math.sqrt((1 - rho * rho) / (nu + 1))
    num, _ = scipy_integrate.quad(lambda x: scipy_stats.t.pdf(x, nu) * scipy_stats.t.sf((c - rho * x) / (s * math.sqrt(nu + x * x)), nu + 1), c, np.inf, epsabs=1e-14, epsrel=1e-12)
    assert K.cond_exceed_t(level, rho, nu) == pytest.approx(num / (1 - level), rel=1e-8)


@pytest.mark.parametrize("nu,rho,level", [(3, 0.5, 0.9), (5, 0.7, 0.95), (4, 0.3, 0.9)])
def test_t_conditional_matches_the_multivariate_t_cdf(nu, rho, level):
    mvt = pytest.importorskip("scipy.stats").multivariate_t
    c = scipy_stats.t.ppf(level, nu)
    dist = mvt([0, 0], [[1, rho], [rho, 1]], df=nu, seed=1)
    both = 1.0 - 2.0 * level + dist.cdf([c, c])
    # Die Verteilungsfunktion von scipy ist numerisch integriert (Fehler bis ~10⁻⁴ in der Wahrscheinlichkeit): verglichen wird deshalb die gemeinsame Wahrscheinlichkeit, nicht das durch 1 − Niveau geteilte Verhältnis.
    assert K.cond_exceed_t(level, rho, nu) * (1.0 - level) == pytest.approx(both, abs=3e-4)


@pytest.mark.parametrize("copula,nu,level,rho", [("gauss", 3, 0.95, 0.5), ("t", 3, 0.95, 0.5), ("t", 5, 0.9, 0.7)])
def test_conditional_matches_a_brute_force_count_on_simulated_pairs(copula, nu, level, rho):
    x, y = K.draw(copula, 400_000, rho, nu, S.VRng(21))
    c = float(scipy_stats.norm.ppf(level))
    over = int((x > c).sum())
    hits = sum(1 for a, b in zip(x[x > c], y[x > c]) if b > c)
    est = hits / over
    exact = K.cond_exceed(copula, level, rho, nu)
    se = math.sqrt(exact * (1 - exact) / over)
    assert abs(est - exact) < 4.5 * se, (est, exact, se)


def test_the_simulated_pairs_have_standard_normal_margins():
    x, y = K.draw("t", 100_000, 0.5, 3, S.VRng(5))
    for v in (x, y):
        assert scipy_stats.kstest(v, "norm").statistic < 0.01
        assert abs(v.mean()) < 0.02 and abs(v.std() - 1) < 0.02


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("nu", [3, 5])
def test_the_conditional_probability_of_the_t_copula_converges_to_lambda(rho, nu):
    lam = K.lambda_t(rho, nu)
    assert K.cond_exceed_t(1 - 1e-9, rho, nu) == pytest.approx(lam, abs=2e-3)
    assert K.cond_exceed_t(0.99, rho, nu) > lam                      # von oben
    assert lam == pytest.approx(2 * float(scipy_stats.t.cdf(-math.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)), abs=1e-12)


def test_the_gauss_copula_has_no_tail_dependence_but_decays_slowly():
    vals = [K.cond_exceed_gauss(q, 0.5) for q in (0.9, 0.99, 0.999, 0.9999, 0.99999)]
    assert all(a > b for a, b in zip(vals, vals[1:])) and vals[-1] < 0.015 and vals[2] > 0.05


def test_kendall_and_spearman_match_scipy():
    x, y = K.draw("t", 1500, 0.6, 4, S.VRng(9))
    assert K.kendall_tau(x, y) == pytest.approx(scipy_stats.kendalltau(x, y)[0], abs=1e-12)
    assert K.spearman_r(x, y) == pytest.approx(scipy_stats.spearmanr(x, y)[0], abs=1e-12)
    assert K.pearson_r(x, y) == pytest.approx(float(np.corrcoef(x, y)[0, 1]), abs=1e-12)


@pytest.mark.parametrize("copula,nu", [("gauss", 3), ("t", 3), ("t", 10)])
def test_kendalls_tau_is_the_same_function_of_rho_for_every_elliptical_copula(copula, nu):
    x, y = K.draw(copula, 6000, 0.5, nu, S.VRng(31))
    assert K.kendall_tau(x, y) == pytest.approx(K.tau_from_rho(0.5), abs=0.02)


def test_tail_rate_matches_a_loop_version():
    x, y = K.draw("t", 300, 0.5, 3, S.VRng(13))
    n = len(x)
    order_x, order_y = sorted(range(n), key=lambda i: x[i]), sorted(range(n), key=lambda i: y[i])
    for k in (3, 15, 60):
        top_x, top_y = set(order_x[n - k:]), set(order_y[n - k:])
        assert K.tail_rate(x, y, k) == pytest.approx(len(top_x & top_y) / k, abs=1e-15)


def test_gauss_sum_quantile_matches_a_simulation():
    x, y = K.draw("gauss", 400_000, 0.5, 3, S.VRng(41))
    s = x + y
    assert float(np.quantile(s, 0.99)) == pytest.approx(K.gauss_sum_quantile(0.5, 0.99), abs=0.04)
    assert float((s > 6).mean()) == pytest.approx(K.gauss_sum_exceed(0.5, 6.0), abs=0.0003)


def test_the_t_sum_has_a_heavier_tail_than_the_gauss_sum_at_equal_rho():
    xg, yg, xt, yt = K.draw_both(300_000, 0.5, 3, S.VRng(43))
    g, t = K.sum_stats(xg, yg), K.sum_stats(xt, yt)
    assert t["exceed"] > 1.8 * g["exceed"]
    assert t["gain"] < g["gain"]
    assert t["quantiles"][2] > g["quantiles"][2]
