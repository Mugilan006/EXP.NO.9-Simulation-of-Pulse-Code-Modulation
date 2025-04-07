# EXP.NO.9-Simulation-of-Pulse-Code-Modulation
9.Simulation of PCM

# AIM
To simulate the Pulse Code Modulation using python

# SOFTWARE REQUIRED
Google Colab

# ALGORITHMS
1. Initialize the sampling rate,frequency,duration and quantization level
2. Create a Time Vector t using linesapce
3. Generate sampling rate duration points
4. Generate the analog message signal using a sine wave
5. Generate a clock signal, used for sampling
6. Calculate quantization_step and quantize message signal
7. Convert quantized_signal to a digital representation
8. Display the plots.

# PROGRAM
import matplotlib.pyplot as plt
import numpy as np
sampling_rate = 5000 # Sampling rate (samples per second)
frequency = 50 # Frequency of the message signal (analog signal)
duration = 0.1 # Duration of the signal in seconds
quantization_levels = 16 # Number of quantization levels (PCM resolution)
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
message_signal = np.sin(2 * np.pi * frequency * t)
clock_signal = np.sign(np.sin(2 * np.pi * 200 * t)) # Increased clock frequency to 200 Hz
quantization_step = (max(message_signal) - min(message_signal)) / quantization_levels
quantized_signal = np.round(message_signal / quantization_step) * quantization_step
pcm_signal = (quantized_signal - min(quantized_signal)) / quantization_step
pcm_signal = pcm_signal.astype(int)
plt.figure(figsize=(12, 10))
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label="Message Signal (Analog)", color='blue')
plt.title("Message Signal (Analog)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(t, clock_signal, label="Clock Signal (Increased Frequency)", color='green')
plt.title("Clock Signal (Increased Frequency)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.subplot(4, 1, 3)
plt.step(t, quantized_signal, label="PCM Modulated Signal", color='red')
plt.title("PCM Modulated Signal (Quantized)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.subplot(4, 1, 4)
plt.plot(t, quantized_signal, label="Signal Demodulation", color='purple', linestyle='--')
plt.title("Signal Without Demodulation")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# OUTPUT
![download](https://github.com/user-attachments/assets/d2945ebc-3d1c-46b6-bb30-bb267fa07102)

 
# RESULT / CONCLUSIONS
Thus the simulation of pulse-code modulation was performed using python
