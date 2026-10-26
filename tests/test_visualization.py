"""Abbildungen: jede wird gebaut, hat die erwarteten Spuren und Achsen."""

import numpy as np
import pytest

import tdp_constants as C
import tdp_evaluation as ev
import tdp_visualization as V


@pytest.fixture(scope="module")
def grid():
    return ev.load_grid()


def test_every_chart_builds_and_locks_its_axes(grid):
    a = ev.analyse(ev.Settings(0.5, 3, 200, 0.99, 5, 3))
    curves = ev.cond_curves(0.5)
    figs = [V.build_scatter(a.xg, a.yg, a.c, C.COLOR_GAUSS, "Gauß"), V.build_cond_curves(curves, (3,), mark_level=0.99), V.build_cond_curves(curves, C.COND_NUS), V.build_gain(grid["sum"], 3),
            V.build_exceed(grid["sum"], 3), V.build_rank_bars(ev.find(grid["rank"], rho=0.5, nu=3)), V.build_estimate(grid["estimate"], 0.05), V.build_accuracy(grid["estimate"]),
            V.build_nu_curves(ev.nu_table(0.5))]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and len(fig.data) >= 1


def test_scatter_marks_exactly_the_weeks_above_the_threshold_in_both_series():
    x, y = np.array([0.0, 2.0, 3.0, 2.5]), np.array([0.0, 3.0, 1.0, 2.6])
    fig = V.build_scatter(x, y, 2.4, "#123456", "Test")
    assert [t.name for t in fig.data] == ["übrige Wochen", "beide über der Schwelle"]
    assert list(fig.data[1].x) == [2.0 + 0.5] and list(fig.data[1].y) == [2.6]
    assert len(fig.data[0].x) == 3 and fig.layout.title.text == "Test"
    assert {s.x0 for s in fig.layout.shapes if s.x0 == s.x1} == {2.4} and {s.y0 for s in fig.layout.shapes if s.y0 == s.y1} == {2.4}
    assert list(fig.layout.xaxis.range) == [-V.AXIS_LIM, V.AXIS_LIM]


def test_exponent_of_a_level():
    assert V.exponent(0.99) == pytest.approx(2.0) and V.exponent(0.9) == pytest.approx(1.0) and V.exponent(0.99999) == pytest.approx(5.0)


def test_cond_curve_chart_has_gauss_and_one_line_plus_lambda_per_nu():
    curves = ev.cond_curves(0.5)
    fig = V.build_cond_curves(curves, (3, 10), mark_level=0.999)
    assert [t.name for t in fig.data if t.showlegend is not False] == ["Gauß-Kopula", "t-Kopula, ν = 3", "t-Kopula, ν = 10"]
    assert len(fig.data) == 5 and list(fig.data[0].y) == pytest.approx(curves["gauss"]) and list(fig.data[2].y) == [curves["lambda"][3]] * 2
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].x0 == pytest.approx(3.0)
    assert list(fig.layout.xaxis.tickvals) == [1, 2, 3, 4, 5] and fig.layout.xaxis.ticktext[1] == "99 %" and fig.layout.xaxis.ticktext[4] == "99,999 %"
    assert fig.layout.xaxis.range[1] > 5.0                                                   # der letzte Teilstrich liegt innerhalb der Achse
    assert not V.build_cond_curves(curves, (3,)).layout.shapes


def test_gain_and_exceed_charts_use_the_cells_of_one_nu(grid):
    fig = V.build_gain(grid["sum"], 5)
    assert [t.name for t in fig.data] == ["Gauß-Kopula", "t-Kopula, ν = 5"] and list(fig.data[0].x) == list(C.SUM_RHOS)
    assert fig.data[1].y[2] == pytest.approx(100 * ev.find(grid["sum"], rho=0.5, nu=5)["t"]["gain"])
    ex = V.build_exceed(grid["sum"], 5)
    assert list(ex.data[0].x) == [r for r in C.SUM_RHOS if r > 0] and ex.layout.yaxis.type == "log"          # ρ = 0 hat keine sichtbare Wahrscheinlichkeit auf der Log-Achse


def test_rank_bars_group_three_measures_for_two_copulas(grid):
    cell = ev.find(grid["rank"], rho=0.7, nu=3)
    fig = V.build_rank_bars(cell)
    assert [t.name for t in fig.data] == ["Gauß-Kopula", "t-Kopula, ν = 3"] and list(fig.data[0].x) == ["Pearson", "Spearman", "Kendall"]
    assert list(fig.data[1].y) == pytest.approx([cell["t"]["pearson"], cell["t"]["spearman"], cell["t"]["kendall"]]) and fig.layout.barmode == "group"


def test_estimate_chart_shows_mean_with_spread_and_the_exact_value(grid):
    fig = V.build_estimate(grid["estimate"], 0.05)
    assert [t.name for t in fig.data][::2] == ["Daten der Gauß-Kopula", "Daten der t-Kopula"] and list(fig.data[0].x) == [f"{n} Wochen" for n in C.EST_NS]
    first = ev.find(grid["estimate"], n=260, share=0.05)
    assert fig.data[0].y[0] == pytest.approx(first["mean"]["gauss"]) and fig.data[0].error_y.array[0] == pytest.approx(first["sd"]["gauss"]) and fig.data[1].y[0] == pytest.approx(first["exact"]["gauss"])


def test_accuracy_chart_has_one_line_per_share_and_a_reference_at_fifty(grid):
    fig = V.build_accuracy(grid["estimate"])
    assert len(fig.data) == 3 and [t.name for t in fig.data] == ["2 % der Wochen am Rand", "5 % der Wochen am Rand", "10 % der Wochen am Rand"]
    assert list(fig.data[1].x) == list(C.EST_NS) and fig.data[1].y[0] == pytest.approx(100 * ev.find(grid["estimate"], n=260, share=0.05)["accuracy"])
    assert any(s.y0 == 50 for s in fig.layout.shapes)


def test_nu_chart_has_two_lines_per_level_and_lambda():
    table = ev.nu_table(0.5)
    fig = V.build_nu_curves(table)
    assert len(fig.data) == 2 * len(table["levels"]) + 1 and fig.data[-1].name == "λ der t-Kopula"
    assert list(fig.data[0].x) == [r["nu"] for r in table["rows"]] and list(fig.data[1].y) == [table["gauss"][0]] * 2
