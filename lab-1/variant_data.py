"""Input data of lab 1, variant 23 (table "Завдання 1", p. 24 of the guide)."""

from fractions import Fraction

import numpy as np

VARIANT_NUMBER = 23
POLYNOMIAL_DEGREE = 2

# Exact values for the hand least-squares calculation: x_n = 0.1 * n for n = 1..8.
SAMPLE_X_EXACT = tuple(Fraction(sample_number, 10) for sample_number in range(1, 9))
SAMPLE_Y_EXACT = tuple(Fraction(value) for value in (1, 2, 2, 3, 3, 4, 4, 5))

# Same samples as float64 arrays, for the spline, the library checks and the plot.
SAMPLE_X = np.array(SAMPLE_X_EXACT, dtype=np.float64)
SAMPLE_Y = np.array(SAMPLE_Y_EXACT, dtype=np.float64)
