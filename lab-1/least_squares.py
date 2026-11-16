"""Least-squares smoothing polynomial p(x) = a + b * x + c * x^2, formulas (4)-(10) of the guide."""

import numpy as np


def compute_power_sums(sample_x, sample_y, polynomial_degree):
    """Return the sums of x powers and of x powers times y used in the normal equations (9)-(10).

    x_power_sums[power] is the sum of x_n^power (power 0 just counts the points, N).
    weighted_y_sums[power] is the sum of x_n^power * y_n.
    Fraction inputs give exact sums, floats give float sums.
    """
    x_power_sums = [
        sum(x_value**power for x_value in sample_x)
        for power in range(2 * polynomial_degree + 1)
    ]
    weighted_y_sums = [
        sum(x_value**power * y_value for x_value, y_value in zip(sample_x, sample_y))
        for power in range(polynomial_degree + 1)
    ]
    return x_power_sums, weighted_y_sums


def build_normal_equations(x_power_sums, weighted_y_sums, polynomial_degree):
    """Build the matrix and the right-hand side of the normal equations (10) from the power sums."""
    system_matrix = [
        [x_power_sums[row_index + column_index] for column_index in range(polynomial_degree + 1)]
        for row_index in range(polynomial_degree + 1)
    ]
    right_hand_side = list(weighted_y_sums)
    return system_matrix, right_hand_side


def compute_determinant(square_matrix):
    """Determinant by expansion along the first row. Slow for big matrices, fine for 3x3, and exact for Fraction."""
    if len(square_matrix) == 1:
        return square_matrix[0][0]
    determinant = 0
    for column_index, leading_value in enumerate(square_matrix[0]):
        minor = [row[:column_index] + row[column_index + 1:] for row in square_matrix[1:]]
        sign = -1 if column_index % 2 else 1
        determinant += sign * leading_value * compute_determinant(minor)
    return determinant


def solve_by_cramer(system_matrix, right_hand_side):
    """Solve the system by Cramer's rule, the way it is done by hand in the report.

    Column determinant k is the determinant with column k replaced by the right-hand side;
    coefficient k is that determinant divided by the main one.
    Returns (coefficients, main_determinant, column_determinants).
    """
    main_determinant = compute_determinant(system_matrix)
    column_determinants = []
    for column_index in range(len(system_matrix)):
        replaced_matrix = [
            row[:column_index] + [right_hand_side[row_index]] + row[column_index + 1:]
            for row_index, row in enumerate(system_matrix)
        ]
        column_determinants.append(compute_determinant(replaced_matrix))
    coefficients = [column_determinant / main_determinant for column_determinant in column_determinants]
    return coefficients, main_determinant, column_determinants


def evaluate_polynomial(coefficients, x_values):
    """Value of the polynomial (4) at x_values. Coefficients go from the constant term up."""
    x_values = np.asarray(x_values, dtype=np.float64)
    polynomial_values = np.zeros_like(x_values)
    for power, coefficient in enumerate(coefficients):
        polynomial_values += float(coefficient) * x_values**power
    return polynomial_values


def compute_residual_sum_of_squares(coefficients, sample_x, sample_y):
    """Sum of squared deviations S from formula (6), the value the least-squares method makes smallest."""
    residuals = evaluate_polynomial(coefficients, sample_x) - np.asarray(sample_y, dtype=np.float64)
    return float(np.sum(residuals**2))
