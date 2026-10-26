"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def four_weeks():
    """Vier Wochen, zwei Standorte: x = (1, 2, 3, 4), y = (2, 1, 4, 3). Paare (i, j): (1,2) diskordant, (1,3) (1,4) (2,3) (2,4) konkordant, (3,4) diskordant:
    4 konkordante und 2 diskordante von 6, also Kendalls τ = (4 − 2)/6 = 1/3."""
    return np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 1.0, 4.0, 3.0])


@pytest.fixture
def five_weeks():
    """Fünf Wochen: x = (1, 2, 3, 4, 5), y = (1, 2, 5, 3, 4). Die zwei größten x-Werte stehen in den Wochen 4 und 5 (Index 3, 4), die zwei größten y-Werte in den Wochen 3 und 5 (Index 2, 4):
    gemeinsam nur Woche 5, der Rangschätzer mit k = 2 ist also 1/2; mit k = 3 stehen die Wochen 3, 4, 5 oben in beiden Reihen, er ist 1."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1.0, 2.0, 5.0, 3.0, 4.0])
