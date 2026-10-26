"""Eigenschaften: Symmetrie, Invarianz der Rangmaße gegen monotone Umformungen (und Nicht-Invarianz von Pearson), Monotonie der exakten Wahrscheinlichkeiten."""

import numpy as np
import pytest

import prs_scenario as S
import tdp_algorithm as K


def test_the_conditional_probability_is_symmetric_in_the_two_series():
    """Die Zählung P(X > c, Y > c) ändert sich nicht, wenn man X und Y vertauscht."""
    x, y = K.draw("t", 2000, 0.5, 3, S.VRng(3))
    assert K.quadrant_count(x, y, 1.0)[1] == K.quadrant_count(y, x, 1.0)[1]
    assert K.tail_rate(x, y, 40) == K.tail_rate(y, x, 40)
    assert K.kendall_tau(x, y) == pytest.approx(K.kendall_tau(y, x), abs=1e-15)


@pytest.mark.parametrize("transform", [np.exp, lambda v: v ** 3, lambda v: 5.0 * v + 2.0])
def test_rank_measures_and_the_rank_estimator_do_not_change_under_monotone_transformations(transform):
    x, y = K.draw("t", 1500, 0.6, 4, S.VRng(7))
    tx, ty = transform(x), transform(y)
    assert K.spearman_r(tx, ty) == pytest.approx(K.spearman_r(x, y), abs=1e-12)
    assert K.kendall_tau(tx, ty) == pytest.approx(K.kendall_tau(x, y), abs=1e-12)
    assert K.tail_rate(tx, ty, 60) == K.tail_rate(x, y, 60)


def test_pearson_changes_under_a_monotone_transformation():
    x, y = K.draw("gauss", 3000, 0.6, 3, S.VRng(7))
    assert abs(K.pearson_r(np.exp(x), np.exp(y)) - K.pearson_r(x, y)) > 0.05


def test_the_conditional_probability_rises_with_rho_and_falls_with_the_level():
    for copula, nu in (("gauss", 3), ("t", 3), ("t", 10)):
        by_rho = [K.cond_exceed(copula, 0.99, rho, nu) for rho in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert by_rho == sorted(by_rho)
        by_level = [K.cond_exceed(copula, q, 0.5, nu) for q in (0.9, 0.95, 0.99, 0.999, 0.9999)]
        assert by_level == sorted(by_level, reverse=True)


def test_the_t_copula_lies_above_the_gauss_copula_at_every_checked_point_and_ordered_by_nu():
    for rho in (0.3, 0.5, 0.9):
        for q in (0.9, 0.99, 0.999, 0.99999):
            g = K.cond_exceed_gauss(q, rho)
            ts = [K.cond_exceed_t(q, rho, nu) for nu in (3, 5, 10, 30)]
            assert ts == sorted(ts, reverse=True) and ts[-1] > g, (rho, q)


def test_lambda_falls_with_nu_and_rises_with_rho():
    assert [K.lambda_t(0.5, nu) for nu in (3, 5, 10, 30)] == sorted([K.lambda_t(0.5, nu) for nu in (3, 5, 10, 30)], reverse=True)
    lam = [K.lambda_t(rho, 5) for rho in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert lam == sorted(lam)


def test_the_gauss_pooling_gain_falls_with_rho():
    gains = [K.pooling_gain(K.gauss_sum_quantile(rho, 0.999), 0.999) for rho in (0.0, 0.3, 0.5, 0.7, 0.9, 0.99)]
    assert gains == sorted(gains, reverse=True) and gains[-1] < 0.01


def test_the_margins_do_not_depend_on_the_copula():
    """Beide Welten haben Standardnormal-Ränder: Mittel, Streuung und Schiefe stimmen überein (4 · 10⁵ Werte)."""
    xg, _, xt, _ = K.draw_both(400_000, 0.5, 3, S.VRng(9))
    for v in (xg, xt):
        assert abs(v.mean()) < 0.01 and abs(v.std() - 1.0) < 0.01 and abs(((v - v.mean()) ** 3).mean()) < 0.03
        assert abs(((v / v.std()) ** 4).mean() - 3.0) < 0.1                       # Kurtosis 3 wie bei der Normalverteilung
