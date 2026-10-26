"""Randfälle (vor dem ersten Mutationstest): Reglergrenzen, rho = 0, sehr hohe Niveaus, kleinste und größte Reihe."""

import math

import numpy as np
import pytest

import prs_scenario as S
import tdp_algorithm as K
import tdp_constants as C
import tdp_evaluation as ev


@pytest.mark.parametrize("rho", [C.RHO_MIN, C.RHO_MAX])
@pytest.mark.parametrize("nu", [C.NU_MIN, C.NU_MAX])
@pytest.mark.parametrize("level", [C.LEVELS[0], C.LEVELS[-1]])
def test_the_exact_conditional_is_a_probability_at_every_slider_corner(rho, nu, level):
    for v in (K.cond_exceed_gauss(level, rho), K.cond_exceed_t(level, rho, nu)):
        assert 0.0 < v < 1.0 and math.isfinite(v)


def test_rho_zero_gauss_is_independence_but_the_t_copula_is_not():
    assert K.cond_exceed_gauss(0.99, 0.0) == pytest.approx(0.01, rel=1e-9)
    assert K.cond_exceed_t(0.99, 0.0, 3) > 0.05                                   # die t-Kopula mit ρ = 0 ist abhängig (gemeinsamer Schwankungsfaktor)
    assert K.lambda_t(0.0, 3) > 0.1


def test_analyse_at_the_smallest_and_largest_series():
    small = ev.analyse(ev.Settings(0.5, 3, C.N_MIN, 0.99, C.SHARE_MAX, 1))
    assert small.k == 10 and len(small.xg) == C.N_MIN                              # 20 % von 50 Wochen
    tiny_share = ev.analyse(ev.Settings(0.5, 3, C.N_MIN, 0.99, C.SHARE_MIN, 1))
    assert tiny_share.k == 3                                                       # mindestens 3 Wochen am Rand
    big = ev.analyse(ev.Settings(0.95, 30, C.N_MAX, 0.99999, 1, 1))
    assert len(big.xg) == C.N_MAX and big.k == 50 and np.all(np.isfinite(big.xt)) and np.all(np.isfinite(big.yt))


def test_no_infinite_values_even_for_extreme_degrees_of_freedom_and_many_pairs():
    for nu in (C.NU_MIN, C.NU_MAX):
        x, y = K.draw("t", 200_000, 0.95, nu, S.VRng(5))
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and abs(x).max() < 8.0


def test_tail_rate_at_the_extremes_of_k():
    x = np.arange(10.0)
    assert K.tail_rate(x, x, 10) == 1.0 and K.tail_rate(x, x, 1) == 1.0 and K.tail_rate(x, -x, 1) == 0.0 and K.tail_rate(x, -x, 5) == 0.0


def test_kendall_tau_needs_two_equally_long_series():
    for args in ([1.0], [1.0, 2.0]):
        with pytest.raises(ValueError):
            K.kendall_tau(args, [1.0, 2.0, 3.0])
    assert K.kendall_tau([1.0, 2.0], [1.0, 2.0]) == 1.0


def test_pooling_gain_stays_between_zero_and_the_independence_bound_for_the_gauss_copula():
    for rho in (0.0, 0.5, 0.95):
        g = K.pooling_gain(K.gauss_sum_quantile(rho, 0.999), 0.999)
        assert 0.0 <= g <= 1.0 - 1.0 / math.sqrt(2.0) + 1e-12


def test_integration_has_no_hidden_dependence_on_the_panel_split():
    """Mehr Teilpunkte ändern das Ergebnis nicht (die Regel ist ausreichend fein)."""
    finer = tuple(sorted(set(C.QUAD_EDGES) | {0.01, 0.03, 0.07, 0.15, 0.25, 0.42, 0.58, 0.72, 0.85, 0.93, 0.97, 0.99, 0.999}))
    for level in (0.9, 0.99999):
        c = K.norm_ppf(level)

        def f(w, c=c):
            x = c / (1.0 - w)
            return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi) * K.norm_sf_vec((c - 0.5 * x) / math.sqrt(0.75)) * c / (1.0 - w) ** 2

        assert K.integrate_unit(f) == pytest.approx(K.integrate_unit(f, finer), rel=1e-10)
