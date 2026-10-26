"""Auswertung: Einzeldatensatz, exakte Kurven, Zellen des Rasters (klein gerechnet), Zugriff auf das Raster."""

import numpy as np
import pytest

import prs_scenario as S
import tdp_algorithm as K
import tdp_constants as C
import tdp_evaluation as ev


def test_settings_equality_and_hash_use_the_values():
    assert ev.Settings() == ev.Settings() and hash(ev.Settings()) == hash(ev.Settings())
    assert ev.Settings(rho=0.6) != ev.Settings() and ev.Settings() != "x"
    assert len({ev.Settings(), ev.Settings(), ev.Settings(seed=1)}) == 2


def test_analyse_is_deterministic_and_consistent_with_the_algorithm():
    s = ev.Settings(0.5, 3, 300, 0.95, 5, 7)
    a, b = ev.analyse(s), ev.analyse(s)
    assert np.array_equal(a.xg, b.xg) and np.array_equal(a.yt, b.yt) and a.exact == b.exact
    assert a.c == pytest.approx(1.6448536269514722, abs=1e-9)
    assert a.quadrant["gauss"] == K.quadrant_count(a.xg, a.yg, a.c) and a.quadrant["t"] == K.quadrant_count(a.xt, a.yt, a.c)
    assert a.exact["gauss"] == K.cond_exceed_gauss(0.95, 0.5) and a.exact["t"] == K.cond_exceed_t(0.95, 0.5, 3)
    assert a.lam == K.lambda_t(0.5, 3) and a.k == 15 and a.tail_level == pytest.approx(0.95)
    assert a.tail_est["gauss"] == K.tail_rate(a.xg, a.yg, 15) and a.tail_exact["t"] == K.cond_exceed_t(0.95, 0.5, 3)
    assert a.measures["gauss"]["kendall"] == pytest.approx(K.kendall_tau(a.xg, a.yg), abs=1e-15)


def test_analyse_uses_the_same_pairs_as_draw_both():
    s = ev.Settings(0.4, 5, 120, 0.9, 10, 3)
    a = ev.analyse(s)
    data_seed = S.substream_seeds(3, 2)[0]
    xg, yg, xt, yt = K.draw_both(120, 0.4, 5, S.VRng(data_seed))
    assert np.array_equal(a.xg, xg) and np.array_equal(a.yt, yt)


def test_gauss_sums_are_exact():
    g = ev.gauss_sums(0.5)
    assert g["quantiles"] == pytest.approx([K.gauss_sum_quantile(0.5, p) for p in C.SUM_LEVELS])
    assert g["exceed"] == pytest.approx(K.gauss_sum_exceed(0.5, C.SUM_BOUND)) and g["gain"] == pytest.approx(K.pooling_gain(g["quantiles"][1], 0.999))


def test_curve_levels_run_from_ninety_percent_to_the_fifth_nine():
    levels = ev.curve_levels()
    assert levels[0] == pytest.approx(0.9) and levels[-1] == pytest.approx(0.99999) and levels == sorted(levels) and len(levels) == 17


def test_cond_curves_have_the_expected_shape():
    c = ev.cond_curves(0.5)
    assert len(c["gauss"]) == len(c["levels"]) == len(c["t"][3])
    assert all(a > b for a, b in zip(c["gauss"], c["gauss"][1:]))                                  # Gauß fällt streng
    assert all(a > b for a, b in zip(c["t"][3], c["t"][3][1:])) and c["t"][3][-1] > c["lambda"][3]  # t fällt und bleibt über λ
    assert all(c["t"][3][i] > c["t"][10][i] > c["gauss"][i] for i in range(len(c["levels"])))
    assert c["lambda"] == {nu: K.lambda_t(0.5, nu) for nu in C.COND_NUS}


def test_nu_table_rows_and_ordering():
    t = ev.nu_table(0.5)
    assert [r["nu"] for r in t["rows"]] == [3, 4, 5, 7, 10, 15, 20, 30] and len(t["gauss"]) == len(t["levels"]) == 3
    lam = [r["lambda"] for r in t["rows"]]
    assert all(a > b for a, b in zip(lam, lam[1:]))
    for i in range(3):
        col = [r["cond"][i] for r in t["rows"]]
        assert all(a > b for a, b in zip(col, col[1:])) and col[-1] > t["gauss"][i]


def test_cell_seed_differs_for_every_input():
    base = ev.cell_seed("sum", 0.5, 3, 101)
    others = {ev.cell_seed("rank", 0.5, 3, 101), ev.cell_seed("sum", 0.7, 3, 101), ev.cell_seed("sum", 0.5, 5, 101), ev.cell_seed("sum", 0.5, 3, 102), ev.cell_seed("sum", 0.5, 3, 101, extra=1)}
    assert base not in others and len(others) == 5


def test_sum_cell_small_run():
    cell = ev.sum_cell(0.5, 3, pairs=20_000, seeds=(7, 8))
    assert cell["pairs"] == 40_000 and cell["gauss"] == ev.gauss_sums(0.5)
    t = cell["t"]
    assert t["quantiles"][0] < t["quantiles"][1] < t["quantiles"][2] and 0 < t["exceed"] < 0.01
    assert cell == ev.sum_cell(0.5, 3, pairs=20_000, seeds=(7, 8))


def test_rank_cell_small_run():
    cell = ev.rank_cell(0.5, 5, pairs=4_000, tau_pairs=800, seeds=(7,))
    assert cell["tau_formula"] == pytest.approx(1 / 3) and cell["pairs"] == 4_000
    for copula in C.COPULAS:
        assert 0.4 < cell[copula]["pearson"] < 0.6 and 0.4 < cell[copula]["spearman"] < 0.6 and 0.2 < cell[copula]["kendall"] < 0.45


def test_estimate_cell_small_run():
    cell = ev.estimate_cell(120, 0.05, runs_per_seed=40, seeds=(7, 8))
    assert cell["k"] == 6 and cell["level"] == pytest.approx(0.95) and cell["runs"] == 80
    assert cell["exact"]["gauss"] == K.cond_exceed_gauss(0.95, C.EST_RHO) and cell["mid"] == pytest.approx(0.5 * (cell["exact"]["gauss"] + cell["exact"]["t"]))
    assert cell["accuracy"] == pytest.approx(0.5 * (cell["correct"]["gauss"] + cell["correct"]["t"])) and 0.0 <= cell["accuracy"] <= 1.0
    assert cell["mean"]["gauss"] < cell["mean"]["t"]


def test_estimate_cell_is_exactly_reproducible():
    a = ev.estimate_cell(100, 0.1, runs_per_seed=20, seeds=(3,))
    assert a == ev.estimate_cell(100, 0.1, runs_per_seed=20, seeds=(3,))
    assert a != ev.estimate_cell(100, 0.1, runs_per_seed=20, seeds=(4,))


def test_find_matches_floats_and_returns_none_when_missing():
    rows = [{"rho": 0.5, "nu": 3}, {"rho": 0.7, "nu": 5}]
    assert ev.find(rows, rho=0.7, nu=5) == rows[1] and ev.find(rows, rho=0.9) is None and ev.find(rows, rho=0.5 + 1e-12, nu=3) == rows[0]


def test_load_grid_returns_none_for_a_missing_or_broken_file(tmp_path):
    assert ev.load_grid(tmp_path / "gibt-es-nicht.json") is None
    bad = tmp_path / "kaputt.json"
    bad.write_text("{", encoding="utf-8")
    assert ev.load_grid(bad) is None
