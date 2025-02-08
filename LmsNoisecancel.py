
import numpy as np
import matplotlib.pyplot as plt

# Generate a clean voice signal (desired signal)
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs)
desired_signal = np.sin(2 * np.pi * 5 * t)  # Simulated speech signal

# Generate noise
noise = 0.5 * np.random.randn(len(t))  # Random background noise

# Noisy input signal (speech + noise)
input_signal = desired_signal + noise

# LMS Algorithm Parameters
mu = 0.01  # Learning rate
filter_order = 5  # Number of filter coefficients
weights = np.zeros(filter_order)  # Initialize weights
output_signal = np.zeros(len(t))  # Output signal
error_signal = np.zeros(len(t))  # Error signal

# Adaptive Filtering using LMS
for i in range(filter_order, len(t)):
    x_n = input_signal[i - filter_order:i]  # Take past 'filter_order' samples
    y_n = np.dot(weights, x_n)  # Predicted output
    error_signal[i] = desired_signal[i] - y_n  # Calculate error
    weights += 2 * mu * error_signal[i] * x_n  # Update weights

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(t, desired_signal, label="Clean Voice Signal", linestyle='dashed')
plt.plot(t, input_signal, label="Noisy Input Signal", alpha=0.5)
plt.plot(t, output_signal, label="LMS Filtered Signal", linewidth=2)
plt.legend()
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("LMS Algorithm for Noise Cancellation in Speech")
plt.show()
