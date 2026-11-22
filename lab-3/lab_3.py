"""Lab 3, variant 23: amplitude spectrum of a sampled signal by the DFT, by hand and by np.fft."""

# %% Imports and variant data
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

variant_number = 23
sample_numbers = np.arange(10)
sample_values = np.array([0.0, -2.0, -5.0, -3.0, 0.3, 2.0, 1.2, -0.6, -1.2, 0.0])
sample_count = len(sample_values)

# Variants 15-30 use Ta = 0.4 s (fa = 2.5 Hz, NTa = 4 s). The table row nTa = 0...0.9 is printed
# for Ta = 0.1 s, so the sample times are recomputed from the sampling period of our variant.
sampling_period = 0.4
sampling_frequency = 1 / sampling_period
observation_time = sample_count * sampling_period
frequency_resolution = 1 / observation_time  # formula (44) of the guide
sample_times = sample_numbers * sampling_period


def clean(value):
    """Round away float noise, so a tiny negative number prints as 0.000000, not -0.000000."""
    return np.round(value, 10) + 0.0


print(f"Variant {variant_number}, N = {sample_count}")
print("n:      ", sample_numbers)
print("nTa, s: ", sample_times)
print("f(nTa): ", sample_values)
print(f"Ta = {sampling_period} s, fa = {sampling_frequency} Hz, NTa = {observation_time} s,"
      f" delta f = {frequency_resolution} Hz")

# %% Task 1: DFT by hand, formulas (36)-(39) of the guide
# Spectral lines are taken only at the N discrete frequencies of formulas (37) and (38).
harmonic_numbers = np.arange(sample_count)
line_frequencies = harmonic_numbers / observation_time  # f_k = k / (N * Ta), formula (38)
angular_frequencies = 2 * np.pi * line_frequencies  # w_k = 2 * pi * k / (N * Ta), formula (37)

# Formula (39): the sum of f(nTa) * exp(-j * w_k * nTa). As in the worked example of the guide,
# the exponent is split by Euler's formula into a cosine sum (real part) and a sine sum (imaginary part).
spectrum_real_parts = np.zeros(sample_count)
spectrum_imaginary_parts = np.zeros(sample_count)
for harmonic_index in range(sample_count):
    for sample_index in range(sample_count):
        phase = angular_frequencies[harmonic_index] * sample_times[sample_index]
        spectrum_real_parts[harmonic_index] += sample_values[sample_index] * np.cos(phase)
        spectrum_imaginary_parts[harmonic_index] -= sample_values[sample_index] * np.sin(phase)

hand_spectrum = spectrum_real_parts + 1j * spectrum_imaginary_parts
dft_magnitudes = np.sqrt(spectrum_real_parts ** 2 + spectrum_imaginary_parts ** 2)
# Formula (36): the DFT is multiplied by Ta, so it matches the ordinary Fourier transform in size and units.
amplitude_spectrum = sampling_period * dft_magnitudes
# Formula (47): the complex Fourier coefficients are the DFT divided by N.
fourier_coefficients = hand_spectrum / sample_count

print(f"{'k':>2} {'f_k, Hz':>8} {'w_k, rad/s':>11} {'Re F_d':>11} {'Im F_d':>11}"
      f" {'|F_d|':>10} {'|F(jw_k)|':>10} {'|c_k|':>9}")
for harmonic_index in range(sample_count):
    print(f"{harmonic_index:>2} {line_frequencies[harmonic_index]:8.2f}"
          f" {angular_frequencies[harmonic_index]:11.6f}"
          f" {clean(spectrum_real_parts[harmonic_index]):11.6f}"
          f" {clean(spectrum_imaginary_parts[harmonic_index]):11.6f}"
          f" {dft_magnitudes[harmonic_index]:10.6f} {amplitude_spectrum[harmonic_index]:10.6f}"
          f" {abs(fourier_coefficients[harmonic_index]):9.6f}")

maximum_index = np.argmax(amplitude_spectrum[: sample_count // 2 + 1])
print(f"maximum |F(jw_k)| = {amplitude_spectrum[maximum_index]:.6f} at k = {maximum_index}"
      f" (f = {line_frequencies[maximum_index]:.2f} Hz) and its mirror k = {sample_count - maximum_index}")

# %% Task 1 check: np.fft.fft
library_spectrum = np.fft.fft(sample_values)
library_frequencies = np.fft.fftfreq(sample_count, d=sampling_period)
library_amplitude_spectrum = sampling_period * np.abs(library_spectrum)
spectrum_difference = np.max(np.abs(hand_spectrum - library_spectrum))
amplitude_difference = np.max(np.abs(amplitude_spectrum - library_amplitude_spectrum))

print(f"{'k':>2} {'Re fft':>11} {'Im fft':>11} {'Ta*|fft|':>10}")
for harmonic_index in range(sample_count):
    print(f"{harmonic_index:>2} {clean(library_spectrum[harmonic_index].real):11.6f}"
          f" {clean(library_spectrum[harmonic_index].imag):11.6f}"
          f" {library_amplitude_spectrum[harmonic_index]:10.6f}")
print("np.fft.fftfreq, Hz:", library_frequencies)
print(f"max |F_d hand - fft| = {spectrum_difference:.1e}")
print(f"max |F hand - F fft| = {amplitude_difference:.1e}")

# %% Checks: inverse DFT, Parseval, symmetry
# Formula (41): the inverse DFT must give back the samples.
restored_values = np.zeros(sample_count, dtype=complex)
for sample_index in range(sample_count):
    for harmonic_index in range(sample_count):
        phase = angular_frequencies[harmonic_index] * sample_times[sample_index]
        restored_values[sample_index] += hand_spectrum[harmonic_index] * np.exp(1j * phase)
restored_values = restored_values / sample_count
hand_restore_error = np.max(np.abs(restored_values - sample_values))
library_restore_error = np.max(np.abs(np.fft.ifft(library_spectrum) - sample_values))

# Parseval: the signal energy equals the spectrum energy divided by N.
# The power spectrum is the DFT times its complex conjugate.
signal_energy = np.sum(sample_values ** 2)
power_spectrum = (hand_spectrum * np.conj(hand_spectrum)).real
spectrum_energy = np.sum(power_spectrum) / sample_count

# A real signal has a mirrored spectrum: line N - k is the complex conjugate of line k.
symmetry_error = np.max(np.abs(hand_spectrum[1:] - np.conj(hand_spectrum[1:][::-1])))

print(f"max |f restored by hand - f| = {hand_restore_error:.1e}")
print(f"max |np.fft.ifft - f|       = {library_restore_error:.1e}")
print(f"sum f^2 = {signal_energy:.6f}, sum |F_d|^2 / N = {spectrum_energy:.6f}")
print(f"max |F_d(N - k) - conj F_d(k)| = {symmetry_error:.1e}")

# %% Task 2: spectrum with zero frequency in the centre, np.fft.fftshift
# fftshift swaps the two halves, so the frequencies run from -fa / 2 up to fa / 2 - delta f.
centred_frequencies = np.fft.fftshift(library_frequencies)
centred_amplitude_spectrum = np.fft.fftshift(library_amplitude_spectrum)

print(f"{'f, Hz':>6} {'|F(jw)|':>10}")
for frequency, amplitude in zip(centred_frequencies, centred_amplitude_spectrum):
    print(f"{frequency:6.2f} {amplitude:10.6f}")

# %% Task 3: plots of the samples and of the centred amplitude spectrum
Path("figures").mkdir(exist_ok=True)
figure, (signal_axis, spectrum_axis) = plt.subplots(2, 1, figsize=(11, 9))

signal_axis.plot(sample_times, sample_values, color="green", linewidth=2, label="f(nTₐ)")
signal_axis.scatter(sample_times, sample_values, color="red", s=65, zorder=5, label="Вибірки (nTₐ, f(nTₐ))")
signal_axis.axhline(0, color="black", linewidth=0.8)
signal_axis.set_title("Вибірки сигналу, Tₐ = 0,4 с")
signal_axis.set_xlabel("t, с")
signal_axis.set_ylabel("f(nTₐ)")
signal_axis.set_xticks(sample_times)
signal_axis.grid(linestyle=":", alpha=0.7)
signal_axis.legend()

markers, stems, baseline = spectrum_axis.stem(centred_frequencies, centred_amplitude_spectrum,
                                               linefmt="green", markerfmt="o", basefmt="black")
plt.setp(stems, linewidth=2)
plt.setp(markers, color="red", markersize=8, zorder=5)
for frequency, amplitude in zip(centred_frequencies, centred_amplitude_spectrum):
    spectrum_axis.annotate(f"{amplitude:.3f}".replace(".", ","), (frequency, amplitude),
                           textcoords="offset points", xytext=(0, 8), ha="center")
spectrum_axis.set_title("Амплітудний спектр |F(jω)| = Tₐ·|F_d(jω)|, нульова частота в центрі")
spectrum_axis.set_xlabel("f, Гц")
spectrum_axis.set_ylabel("|F(jω)|")
spectrum_axis.set_xticks(centred_frequencies)
spectrum_axis.set_ylim(0, 1.15 * np.max(centred_amplitude_spectrum))
spectrum_axis.grid(linestyle=":", alpha=0.7)

plt.tight_layout()
plt.savefig("figures/dft_spectrum.png", dpi=180)
plt.show()

# %% Summary
print(f"Variant {variant_number}: N = {sample_count}, Ta = {sampling_period} s,"
      f" fa = {sampling_frequency} Hz, delta f = {frequency_resolution} Hz")
print(f"F_d(0) = {clean(hand_spectrum[0].real):.6f}, maximum |F(jw_k)| = {amplitude_spectrum[maximum_index]:.6f}"
      f" at f = +-{line_frequencies[maximum_index]:.2f} Hz")
print(f"hand DFT vs np.fft.fft: max difference {spectrum_difference:.1e}")
print(f"inverse DFT error {hand_restore_error:.1e}, Parseval {signal_energy:.6f} = {spectrum_energy:.6f}")
