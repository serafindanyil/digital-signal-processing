"""Natural cubic spline by formulas (12)-(27) of the guide.

Between every two neighbouring points the spline is its own cubic polynomial (12).
"Natural" means the second derivative is zero at the first and the last point.
"""

import numpy as np


def compute_steps(sample_x):
    """Steps between neighbouring x values, h_n = x_{n+1} - x_n (formula (13))."""
    return np.diff(np.asarray(sample_x, dtype=np.float64))


def build_second_derivative_system(sample_x, sample_y):
    """Build the tridiagonal system (26) for the second derivatives at the inner points.

    There is one equation per inner point. Each one links the second derivative at the point
    with its two neighbours, and its right-hand side is 6 times the change of slope there.
    The end values are zero for a natural spline, so they drop out of the first and last rows.
    Returns (sub_diagonal, main_diagonal, super_diagonal, right_hand_side).
    """
    steps = compute_steps(sample_x)
    slopes = np.diff(np.asarray(sample_y, dtype=np.float64)) / steps
    left_steps = steps[:-1]
    right_steps = steps[1:]
    sub_diagonal = left_steps[1:]
    main_diagonal = 2.0 * (left_steps + right_steps)
    super_diagonal = right_steps[:-1]
    right_hand_side = 6.0 * (slopes[1:] - slopes[:-1])
    return sub_diagonal, main_diagonal, super_diagonal, right_hand_side


def solve_tridiagonal(sub_diagonal, main_diagonal, super_diagonal, right_hand_side):
    """Solve a tridiagonal system by the sweep (Thomas) method: go down eliminating, then come back up."""
    unknown_count = len(main_diagonal)
    modified_super = np.zeros(unknown_count)
    modified_right = np.zeros(unknown_count)
    modified_super[0] = super_diagonal[0] / main_diagonal[0] if unknown_count > 1 else 0.0
    modified_right[0] = right_hand_side[0] / main_diagonal[0]
    for row_index in range(1, unknown_count):
        pivot = main_diagonal[row_index] - sub_diagonal[row_index - 1] * modified_super[row_index - 1]
        if row_index < unknown_count - 1:
            modified_super[row_index] = super_diagonal[row_index] / pivot
        modified_right[row_index] = (
            right_hand_side[row_index] - sub_diagonal[row_index - 1] * modified_right[row_index - 1]
        ) / pivot
    solution = np.zeros(unknown_count)
    solution[-1] = modified_right[-1]
    for row_index in range(unknown_count - 2, -1, -1):
        solution[row_index] = modified_right[row_index] - modified_super[row_index] * solution[row_index + 1]
    return solution


def compute_natural_second_derivatives(sample_x, sample_y):
    """Second derivatives y'' at all N points: zero at both ends, system (26) for the rest."""
    interior_second_derivatives = solve_tridiagonal(*build_second_derivative_system(sample_x, sample_y))
    return np.concatenate(([0.0], interior_second_derivatives, [0.0]))


def compute_spline_coefficients(sample_x, sample_y, second_derivatives):
    """Coefficients a, b, c, d of the N-1 cubic pieces, formulas (20)-(23) of the guide.

    a is the sample value at the left end of the piece and c is half the second derivative there.
    b and d also need the step h_n and the second derivative at the right end.
    """
    sample_y = np.asarray(sample_y, dtype=np.float64)
    steps = compute_steps(sample_x)
    constant_terms = sample_y[:-1].copy()
    linear_terms = np.diff(sample_y) / steps - steps * (second_derivatives[1:] + 2.0 * second_derivatives[:-1]) / 6.0
    quadratic_terms = second_derivatives[:-1] / 2.0
    cubic_terms = np.diff(second_derivatives) / (6.0 * steps)
    return constant_terms, linear_terms, quadratic_terms, cubic_terms


def evaluate_spline(sample_x, spline_coefficients, x_values):
    """Value of the spline (27): find the interval each x falls into, then use formula (12) there."""
    sample_x = np.asarray(sample_x, dtype=np.float64)
    x_values = np.asarray(x_values, dtype=np.float64)
    constant_terms, linear_terms, quadratic_terms, cubic_terms = spline_coefficients
    interval_indices = np.clip(np.searchsorted(sample_x, x_values, side="right") - 1, 0, len(sample_x) - 2)
    offsets = x_values - sample_x[interval_indices]
    return (
        constant_terms[interval_indices]
        + linear_terms[interval_indices] * offsets
        + quadratic_terms[interval_indices] * offsets**2
        + cubic_terms[interval_indices] * offsets**3
    )
