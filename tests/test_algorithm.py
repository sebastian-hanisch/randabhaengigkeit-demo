"""Einheiten von Hand (DEMO-PLAYBOOK §1b): jede Funktion an einer Mini-Instanz, deren Ergebnis ohne Rechner folgt."""

import math

import numpy as np
import pytest

import prs_scenario as S
import tdp_algorithm as K
import tdp_constants as C


# --- t-Verteilung ---------------------------------------------------------------------------------------------------------------------

def test_t_cdf_with_one_degree_of_freedom_is_the_cauchy_distribution():
    for x in (-5.0, -1.0, 0.0, 0.7, 1.0, 40.0):
        assert float(K.t_cdf(x, 1)) == pytest.approx(0.5 + math.atan(x) / math.pi, abs=1e-14)
    assert float(K.t_cdf(1.0, 1)) == pytest.approx(0.75, abs=1e-14)


def test_t_cdf_with_two_degrees_of_freedom_has_a_closed_form():
    for x in (-3.0, 0.0, math.sqrt(2.0), 2.5):
        assert float(K.t_cdf(x, 2)) == pytest.approx(0.5 + x / (2.0 * math.sqrt(2.0 + x * x)), abs=1e-14)
    assert float(K.t_cdf(math.sqrt(2.0), 2)) == pytest.approx(0.5 + math.sqrt(2.0) / 4.0, abs=1e-14)


def test_t_cdf_with_three_degrees_of_freedom():
    """F(x) = 1/2 + (1/π)(θ + sin θ cos θ) mit θ = arctan(x/√3); bei x = √3 ist θ = π/4, sin θ cos θ = 1/2: F = 1/2 + (π/4 + 1/2)/π = 3/4 + 1/(2π)."""
    assert float(K.t_cdf(math.sqrt(3.0), 3)) == pytest.approx(0.75 + 1.0 / (2.0 * math.pi), abs=1e-14)


@pytest.mark.parametrize("nu", [1, 2, 3, 4, 7, 10, 30])
def test_t_cdf_is_symmetric_and_monotone(nu):
    x = np.linspace(-8, 8, 161)
    f = K.t_cdf(x, nu)
    assert f == pytest.approx(1.0 - K.t_cdf(-x, nu), abs=1e-14)
    assert np.all(np.diff(f) > 0) and float(K.t_cdf(0.0, nu)) == pytest.approx(0.5, abs=1e-15)


def test_t_cdf_rejects_non_integer_or_nonpositive_degrees():
    for bad in (0, -2, 2.5):
        with pytest.raises(ValueError):
            K.t_cdf(0.0, bad)


def test_t_ppf_inverts_t_cdf():
    for nu in (3, 5, 10):
        for p in (0.001, 0.2, 0.5, 0.9, 0.99, 0.99999):
            assert float(K.t_cdf(K.t_ppf(p, nu), nu)) == pytest.approx(p, rel=1e-8)
    assert K.t_ppf(0.5, 4) == 0.0 and K.t_ppf(0.3, 6) == pytest.approx(-K.t_ppf(0.7, 6), abs=1e-12)
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            K.t_ppf(bad, 3)


def test_t_ppf_of_cauchy_is_the_tangent():
    assert K.t_ppf(0.75, 1) == pytest.approx(1.0, abs=1e-10)       # tan(π/4)


# --- Normalverteilung -------------------------------------------------------------------------------------------------------------------

def test_norm_ppf_vec_known_values_and_symmetry():
    z = K.norm_ppf_vec([0.5, 0.975, 0.025, 0.8413447460685429])
    assert z == pytest.approx([0.0, 1.959963984540054, -1.959963984540054, 1.0], abs=1e-12)
    p = np.array([1e-4, 0.001, 0.3, 0.7, 0.999, 1 - 1e-4])           # 1 − p ist in Gleitkommazahlen nur bis ~10⁻¹⁶ genau: bei 10⁻⁹ wäre der Spiegelvergleich schon ungenau
    assert K.norm_ppf_vec(p) == pytest.approx(-K.norm_ppf_vec(1 - p), abs=1e-12)


def test_norm_ppf_vec_rejects_the_endpoints():
    for bad in ([0.0], [1.0], [0.2, 1.2]):
        with pytest.raises(ValueError):
            K.norm_ppf_vec(bad)


def test_norm_cdf_vec_and_sf_add_to_one():
    x = np.linspace(-6, 6, 25)
    assert K.norm_cdf_vec(x) + K.norm_sf_vec(x) == pytest.approx(np.ones_like(x), abs=1e-15)
    assert float(K.norm_cdf_vec(0.0)) == 0.5


# --- Bedingte Wahrscheinlichkeit ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("level", [0.9, 0.99, 0.99999])
def test_gauss_copula_without_dependence_gives_one_minus_level(level):
    assert K.cond_exceed_gauss(level, 0.0) == pytest.approx(1.0 - level, rel=1e-9)


def test_gauss_copula_with_almost_perfect_dependence_gives_almost_one():
    assert K.cond_exceed_gauss(0.99, 0.99999) > 0.995
    assert K.cond_exceed_t(0.99, 0.99999, 3) > 0.995


def test_the_t_copula_approaches_the_gauss_copula_for_many_degrees_of_freedom():
    """Bei ν = 30 liegen die Zahlen noch merklich auseinander (0,1520 gegen 0,1294 bei q = 0,99), bei ν = 1000 nicht mehr."""
    assert abs(K.cond_exceed_t(0.99, 0.5, 1000) - K.cond_exceed_gauss(0.99, 0.5)) < 0.005
    assert abs(K.cond_exceed_t(0.99, 0.5, 30) - K.cond_exceed_gauss(0.99, 0.5)) > 0.02


def test_cond_exceed_dispatches_and_checks_the_level():
    assert K.cond_exceed("gauss", 0.99, 0.5) == K.cond_exceed_gauss(0.99, 0.5)
    assert K.cond_exceed("t", 0.99, 0.5, 5) == K.cond_exceed_t(0.99, 0.5, 5)
    for bad in (0.5, 1.0, 0.2, 1.5):
        with pytest.raises(ValueError):
            K.cond_exceed_gauss(bad, 0.5)
        with pytest.raises(ValueError):
            K.cond_exceed_t(bad, 0.5, 3)


def test_lambda_t_known_values():
    """ρ = 0,5, ν = 3: 2 · t₄(−√(4 · 0,5/1,5)) = 2 · t₄(−√(4/3)) = 0,3125 (Quadratur im Ergebnis der Vormessung 0,3125); ρ → 1 gibt 1, ρ = −1 + ε gibt fast 0."""
    assert K.lambda_t(0.5, 3) == pytest.approx(0.3125, abs=1e-4)
    assert K.lambda_t(0.999999, 3) > 0.99
    assert K.lambda_t(-0.99, 3) < 1e-4
    assert K.lambda_t(0.5, 3) > K.lambda_t(0.5, 10) > K.lambda_t(0.5, 30) > 0


# --- Erzeuger -----------------------------------------------------------------------------------------------------------------------------

def test_draw_is_deterministic_and_the_two_copulas_differ_only_in_the_dependence():
    a = K.draw("t", 200, 0.5, 3, S.VRng(7))
    b = K.draw("t", 200, 0.5, 3, S.VRng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    g = K.draw("gauss", 200, 0.5, 3, S.VRng(7))
    assert not np.array_equal(g[0], a[0])


def test_draw_both_equals_the_two_single_draws_with_the_same_stream():
    xg, yg, xt, yt = K.draw_both(300, 0.6, 5, S.VRng(11))
    g = K.draw("gauss", 300, 0.6, 5, S.VRng(11))
    t = K.draw("t", 300, 0.6, 5, S.VRng(11))
    assert np.array_equal(xg, g[0]) and np.array_equal(yg, g[1]) and np.array_equal(xt, t[0]) and np.array_equal(yt, t[1])


def test_draw_rejects_an_unknown_copula():
    with pytest.raises(ValueError):
        K.draw("clayton", 10, 0.5, 3, S.VRng(1))


def test_draw_with_rho_one_gives_identical_columns():
    for copula in C.COPULAS:
        x, y = K.draw(copula, 50, 1.0, 3, S.VRng(3))
        assert np.allclose(x, y, atol=1e-12)


# --- Summe und Pooling ------------------------------------------------------------------------------------------------------------------

def test_gauss_sum_quantile_hand_cases():
    """ρ = 0: Summe ~ N(0, 2), Quantil z·√2; ρ = 1: Summe = 2X, Quantil 2z."""
    z = 2.3263478740408408
    assert K.gauss_sum_quantile(0.0, 0.99) == pytest.approx(z * math.sqrt(2.0), abs=1e-9)
    assert K.gauss_sum_quantile(1.0, 0.99) == pytest.approx(2.0 * z, abs=1e-9)
    assert K.gauss_sum_exceed(0.0, 0.0) == pytest.approx(0.5, abs=1e-15)


def test_pooling_gain_is_zero_for_perfect_dependence_and_positive_otherwise():
    assert K.pooling_gain(K.gauss_sum_quantile(1.0, 0.999), 0.999) == pytest.approx(0.0, abs=1e-9)
    assert K.pooling_gain(K.gauss_sum_quantile(0.0, 0.999), 0.999) == pytest.approx(1.0 - 1.0 / math.sqrt(2.0), abs=1e-9)


def test_sum_stats_on_a_tiny_sample():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    st = K.sum_stats(x, x, levels=(0.5, 0.9), bound=10.0)
    assert st["quantiles"] == pytest.approx([10.0, 18.0]) and st["exceed"] == pytest.approx(5 / 11)     # Summen 0, 2, …, 20; über 10: 12, 14, 16, 18, 20


# --- Rangmaße und Schätzer ------------------------------------------------------------------------------------------------------------------

def test_kendall_tau_on_the_hand_instance(four_weeks):
    x, y = four_weeks
    assert K.kendall_tau(x, y) == pytest.approx(1.0 / 3.0, abs=1e-15)
    assert K.kendall_tau(x, x) == pytest.approx(1.0) and K.kendall_tau(x, -x) == pytest.approx(-1.0)


def test_tau_from_rho_hand_values():
    assert K.tau_from_rho(0.5) == pytest.approx(1.0 / 3.0, abs=1e-15)
    assert K.tau_from_rho(0.0) == 0.0 and K.tau_from_rho(1.0) == pytest.approx(1.0, abs=1e-15)


def test_ranks_of_a_small_vector():
    assert list(K.ranks([30.0, 10.0, 20.0, 40.0])) == [2, 0, 1, 3]


def test_tail_rate_on_the_hand_instance(five_weeks):
    x, y = five_weeks
    assert K.tail_rate(x, y, 2) == pytest.approx(0.5)
    assert K.tail_rate(x, y, 3) == pytest.approx(1.0)
    assert K.tail_rate(x, y, 5) == pytest.approx(1.0)
    assert K.tail_rate(x, y, 1) == pytest.approx(0.0)       # größtes x in Woche 5, größtes y in Woche 3


def test_tail_rate_validates_its_arguments(five_weeks):
    x, y = five_weeks
    for k in (0, 6):
        with pytest.raises(ValueError):
            K.tail_rate(x, y, k)
    with pytest.raises(ValueError):
        K.tail_rate(x, y[:4], 2)


def test_quadrant_count_on_the_hand_instance(five_weeks):
    x, y = five_weeks
    assert K.quadrant_count(x, y, 2.5) == (3, 3)           # x > 2,5: Wochen 3, 4, 5; y dort 5, 3, 4, alle über 2,5
    assert K.quadrant_count(x, y, 3.5) == (2, 1)           # x > 3,5: Wochen 4, 5; y dort 3, 4: nur die zweite über 3,5


def test_quadrant_count_counts_only_values_strictly_above_the_threshold():
    """Gleichstände zählen nicht: x = (1, 2, 3), y = (2, 3, 2), c = 2: nur die dritte Woche hat x > 2, und dort ist y = 2 nicht über 2."""
    assert K.quadrant_count([1.0, 2.0, 3.0], [2.0, 3.0, 2.0], 2.0) == (1, 0)
    assert K.quadrant_count([1.0, 2.0, 3.0], [2.0, 3.0, 2.5], 2.0) == (1, 1)


def test_tail_k_rounds_and_clamps():
    assert K.tail_k(500, 0.05) == 25 and K.tail_k(50, 0.01) == 3 and K.tail_k(2, 0.5) == 2 and K.tail_k(260, 0.05) == 13
