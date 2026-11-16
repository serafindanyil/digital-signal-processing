"""Lab 1, variant 23: least-squares polynomial and natural cubic spline, by hand and by library."""

from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import numpy as np
from scipy.interpolate import CubicSpline

import cubic_spline
import least_squares
import plotting
from variant_data import (
    POLYNOMIAL_DEGREE,
    SAMPLE_X,
    SAMPLE_X_EXACT,
    SAMPLE_Y,
    SAMPLE_Y_EXACT,
    VARIANT_NUMBER,
)

FIGURES_FOLDER = Path(__file__).resolve().parent / "figures"
DENSE_POINT_COUNT = 701
COEFFICIENT_NAMES = ("a", "b", "c", "d")


# ---------- calculations ----------

def solve_polynomial_by_hand():
    """Least-squares polynomial by formulas (4)-(10) and Cramer's rule. Fractions keep it exact."""
    x_power_sums, weighted_y_sums = least_squares.compute_power_sums(SAMPLE_X_EXACT, SAMPLE_Y_EXACT, POLYNOMIAL_DEGREE)
    system_matrix, right_hand_side = least_squares.build_normal_equations(x_power_sums, weighted_y_sums, POLYNOMIAL_DEGREE)
    coefficients, main_determinant, column_determinants = least_squares.solve_by_cramer(system_matrix, right_hand_side)
    return {
        "x_power_sums": x_power_sums,
        "weighted_y_sums": weighted_y_sums,
        "system_matrix": system_matrix,
        "right_hand_side": right_hand_side,
        "main_determinant": main_determinant,
        "column_determinants": column_determinants,
        "coefficients": coefficients,
        "fitted_values": least_squares.evaluate_polynomial(coefficients, SAMPLE_X),
        "residual_sum_of_squares": least_squares.compute_residual_sum_of_squares(coefficients, SAMPLE_X, SAMPLE_Y),
    }


def solve_polynomial_by_library():
    """Same polynomial from np.polyfit. It returns the highest power first, so the order is flipped to a, b, c."""
    return np.polyfit(SAMPLE_X, SAMPLE_Y, POLYNOMIAL_DEGREE)[::-1]


def solve_spline_by_hand():
    """Natural cubic spline by formulas (12)-(27) of the guide."""
    sub_diagonal, main_diagonal, super_diagonal, right_hand_side = cubic_spline.build_second_derivative_system(
        SAMPLE_X, SAMPLE_Y)
    system_matrix = np.diag(main_diagonal) + np.diag(sub_diagonal, -1) + np.diag(super_diagonal, 1)
    second_derivatives = cubic_spline.compute_natural_second_derivatives(SAMPLE_X, SAMPLE_Y)
    coefficients = cubic_spline.compute_spline_coefficients(SAMPLE_X, SAMPLE_Y, second_derivatives)
    return {
        "steps": cubic_spline.compute_steps(SAMPLE_X),
        "system_matrix": system_matrix,
        "right_hand_side": right_hand_side,
        "second_derivatives": second_derivatives,
        "coefficients": coefficients,
    }


def solve_spline_by_library():
    """Same spline from scipy CubicSpline with bc_type="natural" (zero second derivative at both ends)."""
    natural_spline = CubicSpline(SAMPLE_X, SAMPLE_Y, bc_type="natural")
    # CubicSpline.c keeps the cubic term in row 0, so reverse the rows to get a, b, c, d.
    natural_coefficients = tuple(natural_spline.c[::-1])
    return {
        "natural_spline": natural_spline,
        "coefficients": natural_coefficients,
        "second_derivatives": natural_spline(SAMPLE_X, 2),
    }


def _dense_grid():
    """Dense grid from the first to the last sample, used to compare and draw the curves."""
    return np.linspace(SAMPLE_X[0], SAMPLE_X[-1], DENSE_POINT_COUNT)


def compare_polynomials(polynomial_hand, polynomial_library):
    """Largest difference between the hand polynomial and np.polyfit: in the coefficients and along the curve."""
    dense_x = _dense_grid()
    hand_coefficients = np.array([float(value) for value in polynomial_hand["coefficients"]])
    return {
        "polynomial_coefficients": float(np.max(np.abs(hand_coefficients - polynomial_library))),
        "polynomial_values": float(np.max(np.abs(
            least_squares.evaluate_polynomial(hand_coefficients, dense_x)
            - np.polyval(polynomial_library[::-1], dense_x)))),
    }


def compare_splines(spline_hand, spline_library):
    """Largest differences between the hand spline and CubicSpline, plus the spline error at the samples."""
    dense_x = _dense_grid()
    hand_spline_values = cubic_spline.evaluate_spline(SAMPLE_X, spline_hand["coefficients"], dense_x)
    return {
        "spline_second_derivatives": float(np.max(np.abs(
            spline_hand["second_derivatives"] - spline_library["second_derivatives"]))),
        "spline_coefficients": float(np.max(np.abs(
            np.array(spline_hand["coefficients"]) - np.array(spline_library["coefficients"])))),
        "spline_values": float(np.max(np.abs(hand_spline_values - spline_library["natural_spline"](dense_x)))),
        "spline_at_samples": float(np.max(np.abs(
            cubic_spline.evaluate_spline(SAMPLE_X, spline_hand["coefficients"], SAMPLE_X) - SAMPLE_Y))),
    }


def build_figures(polynomial_hand, polynomial_library, spline_hand, spline_library):
    """Build the figure for task item 3. Returns a dict that maps the file name to the figure."""
    dense_x = _dense_grid()
    return {
        "function_polynomial_spline.png": plotting.plot_function_polynomial_spline(
            SAMPLE_X, SAMPLE_Y, dense_x, polynomial_hand["coefficients"],
            least_squares.evaluate_polynomial(polynomial_hand["coefficients"], dense_x),
            np.polyval(polynomial_library[::-1], dense_x),
            cubic_spline.evaluate_spline(SAMPLE_X, spline_hand["coefficients"], dense_x),
            spline_library["natural_spline"](dense_x)),
    }


def save_figures(figures):
    """Save every figure to lab-1/figures/ at 180 dpi and return the paths."""
    FIGURES_FOLDER.mkdir(exist_ok=True)
    saved_paths = []
    for file_name, figure in figures.items():
        figure_path = FIGURES_FOLDER / file_name
        figure.savefig(figure_path, dpi=180)
        saved_paths.append(figure_path)
    return saved_paths


# ---------- printing ----------

def _format_float(value):
    """Float with 6 decimals. Round-off noise below 5e-7 prints as 0.000000 instead of -0.000000."""
    return f"{0.0 if abs(value) < 5e-7 else value:.6f}"


def format_polynomial(coefficients):
    """Polynomial as one line of text for the printout, constant term first."""
    power_suffixes = ("", "·x", "·x²", "·x³")
    return " + ".join(f"{_format_float(float(coefficient))}{power_suffixes[power]}"
                      for power, coefficient in enumerate(coefficients))


def _format_fraction(value):
    """Exact fraction next to its 6-decimal value."""
    return f"{str(value):>12} = {float(value):12.6f}"


def print_variant_data():
    print(f"Variant {VARIANT_NUMBER}, N = {len(SAMPLE_X)}")
    print("n   :", "  ".join(f"{sample_number:>4d}" for sample_number in range(1, len(SAMPLE_X) + 1)))
    print("x_n :", "  ".join(f"{value:4.1f}" for value in SAMPLE_X))
    print("y_n :", "  ".join(f"{value:4.1f}" for value in SAMPLE_Y))


def print_polynomial_by_hand(polynomial_hand):
    print("Sums for the normal equations (exact):")
    print(f"  N        = {polynomial_hand['x_power_sums'][0]}")
    for power in range(1, len(polynomial_hand["x_power_sums"])):
        print(f"  Σx^{power}      = {_format_fraction(polynomial_hand['x_power_sums'][power])}")
    for power, weighted_sum in enumerate(polynomial_hand["weighted_y_sums"]):
        print(f"  Σx^{power}·y    = {_format_fraction(weighted_sum)}")
    print("System (10):")
    for row, right_value in zip(polynomial_hand["system_matrix"], polynomial_hand["right_hand_side"]):
        print("  " + "  ".join(f"{float(value):10.6f}" for value in row) + f"  | {float(right_value):10.6f}")
    print("Cramer's rule:")
    print(f"  Δ   = {_format_fraction(polynomial_hand['main_determinant'])}")
    for name, column_determinant, coefficient in zip(
            COEFFICIENT_NAMES, polynomial_hand["column_determinants"], polynomial_hand["coefficients"]):
        print(f"  Δ{name}  = {_format_fraction(column_determinant)}   →   {name} = Δ{name}/Δ = {_format_fraction(coefficient)}")
    print(f"P(x) = {format_polynomial(polynomial_hand['coefficients'])}")
    print("P(x_n) :", "  ".join(f"{value:.6f}" for value in polynomial_hand["fitted_values"]))
    print(f"Residual sum of squares S = {polynomial_hand['residual_sum_of_squares']:.6f}")


def print_polynomial_by_library(polynomial_library, polynomial_comparison):
    print(f"np.polyfit(x, y, {POLYNOMIAL_DEGREE}) →", "  ".join(
        f"{name} = {_format_float(value)}" for name, value in zip(COEFFICIENT_NAMES, polynomial_library)))
    print(f"max |hand − polyfit| (coefficients) = {polynomial_comparison['polynomial_coefficients']:.3e}")
    print(f"max |hand − polyfit| (curve)        = {polynomial_comparison['polynomial_values']:.3e}")


def print_spline_by_hand(spline_hand):
    print("Steps h_n:", "  ".join(f"{value:.1f}" for value in spline_hand["steps"]))
    print("System (26) for y″_2 … y″_7 (y″_1 = y″_8 = 0):")
    for row, right_value in zip(spline_hand["system_matrix"], spline_hand["right_hand_side"]):
        print("  " + "  ".join(f"{value:4.1f}" for value in row) + f"  | {right_value:8.2f}")
    print("y″_n :", "  ".join(_format_float(value) for value in spline_hand["second_derivatives"]))
    print(f"{'interval':>12} {'a_n':>10} {'b_n':>12} {'c_n':>14} {'d_n':>14}")
    constant_terms, linear_terms, quadratic_terms, cubic_terms = spline_hand["coefficients"]
    for interval_index in range(len(constant_terms)):
        print(f"[{SAMPLE_X[interval_index]:.1f}; {SAMPLE_X[interval_index + 1]:.1f}]"
              f" {constant_terms[interval_index]:10.6f} {linear_terms[interval_index]:12.6f}"
              f" {quadratic_terms[interval_index]:14.6f} {cubic_terms[interval_index]:14.6f}")


def print_spline_by_library(spline_library, spline_comparison):
    print('CubicSpline(x, y, bc_type="natural") second derivatives:')
    print("  " + "  ".join(_format_float(value) for value in spline_library["second_derivatives"]))
    print(f"max |hand − CubicSpline| (y″)           = {spline_comparison['spline_second_derivatives']:.3e}")
    print(f"max |hand − CubicSpline| (a, b, c, d)   = {spline_comparison['spline_coefficients']:.3e}")
    print(f"max |hand − CubicSpline| (curve)        = {spline_comparison['spline_values']:.3e}")
    print(f"max |spline(x_n) − y_n|                 = {spline_comparison['spline_at_samples']:.3e}")


def print_summary(polynomial_hand, polynomial_comparison, spline_comparison):
    exact_coefficients = ", ".join(
        f"{name} = {value}" for name, value in zip(COEFFICIENT_NAMES, polynomial_hand["coefficients"]))
    print(f"P(x) = {format_polynomial(polynomial_hand['coefficients'])}  ({exact_coefficients})")
    comparison = {**polynomial_comparison, **spline_comparison}
    largest_difference = max(comparison[key] for key in (
        "polynomial_coefficients", "polynomial_values", "spline_second_derivatives",
        "spline_coefficients", "spline_values"))
    print(f"Largest |hand − library| difference: {largest_difference:.3e}")
    print(f"Spline passes through all samples: max error {spline_comparison['spline_at_samples']:.3e}")


def main():
    polynomial_hand = solve_polynomial_by_hand()
    polynomial_library = solve_polynomial_by_library()
    spline_hand = solve_spline_by_hand()
    spline_library = solve_spline_by_library()
    polynomial_comparison = compare_polynomials(polynomial_hand, polynomial_library)
    spline_comparison = compare_splines(spline_hand, spline_library)

    sections = (
        ("VARIANT DATA", print_variant_data, ()),
        ("1. LEAST-SQUARES POLYNOMIAL BY HAND (formulas 4–10)", print_polynomial_by_hand, (polynomial_hand,)),
        ("1. CHECK WITH np.polyfit", print_polynomial_by_library, (polynomial_library, polynomial_comparison)),
        ("2. NATURAL CUBIC SPLINE BY HAND (formulas 12–27)", print_spline_by_hand, (spline_hand,)),
        ("2. CHECK WITH scipy CubicSpline", print_spline_by_library, (spline_library, spline_comparison)),
        ("SUMMARY", print_summary, (polynomial_hand, polynomial_comparison, spline_comparison)),
    )
    for title, print_section, arguments in sections:
        print(f"\n=== {title} ===")
        print_section(*arguments)

    saved_paths = save_figures(build_figures(polynomial_hand, polynomial_library, spline_hand, spline_library))
    print("\nFigures:", ", ".join(str(path.relative_to(FIGURES_FOLDER.parent.parent)) for path in saved_paths))


if __name__ == "__main__":
    main()
