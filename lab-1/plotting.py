"""Figure for task item 3. It returns the matplotlib figure and leaves saving or showing to the caller."""

import matplotlib.pyplot as plt


def _format_number(value):
    """Number with 6 decimals and a decimal comma, as written in the report."""
    return f"{value:.6f}".replace(".", ",")


def plot_function_polynomial_spline(sample_x, sample_y, dense_x, coefficients, hand_polynomial_values,
                                    library_polynomial_values, hand_spline_values, library_spline_values):
    """Plot the samples, the polynomial and the spline on one figure.

    Hand results are solid lines. The np.polyfit and CubicSpline curves are dashed and drawn on top,
    so if the two calculations agree, the dashed line lies right over the solid one.
    """
    figure, axes = plt.subplots(figsize=(9, 5.5))
    axes.plot(sample_x, sample_y, "-", color="gray", linewidth=1, alpha=0.7, label="Функція y(x) (ламана через точки)")
    axes.plot(sample_x, sample_y, "o", color="black", markersize=7, label="Вихідні дані (xₙ, yₙ)", zorder=3)

    power_suffixes = ("", "x", "x²", "x³")
    polynomial_terms = " + ".join(f"{_format_number(float(coefficient))}{power_suffixes[power]}"
                                  for power, coefficient in enumerate(coefficients))
    axes.plot(dense_x, hand_polynomial_values, color="tab:blue", linewidth=2.5,
              label=f"Поліном P(x) = {polynomial_terms}")
    axes.plot(dense_x, library_polynomial_values, "--", color="tab:orange", linewidth=1.5,
              label=f"np.polyfit(x, y, {len(coefficients) - 1})")
    axes.plot(dense_x, hand_spline_values, color="tab:green", linewidth=2.5, label="Звичайний кубічний сплайн")
    axes.plot(dense_x, library_spline_values, "--", color="tab:red", linewidth=1.5,
              label='CubicSpline(bc_type="natural")')

    axes.set_title("Функція, апроксимуючий поліном та сплайн")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return figure
