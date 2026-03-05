import nidaqmx
from nidaqmx.constants import (AcquisitionType, TerminalConfiguration)
import numpy as np
import matplotlib.pyplot as plt

# --- Analog Input User Configuration ---
device1_ai = "MIODAQ/ai0"
device2_ai = "cDAQ9189-2628AACMod1/ai0"
sample_rate = 100_000
samples_per_channel = 2 * sample_rate  # 2 seconds
min_val = -10.0
max_val = 10.0

# --- Digital Square Wave User Configuration ---
square_wave_counter = "MIODAQ/ctr0"
square_wave_terminal = "/MIODAQ/PFI0"
square_wave_frequency = 1.0
square_wave_duty_cycle = 0.5

# --- Skew Test Initialization ---
num_iterations = 5
skews = []
"""
def find_delta_samples_of_rising_edge(wave1, wave2, threshold=0.5):
    # Finds the sample offset between the first rising edge in two waveforms.
    idx1 = np.argmax(wave1 > threshold)
    idx2 = np.argmax(wave2 > threshold)
    return idx2 - idx1
"""
def find_rising_edge_sample_offset(wave1, wave2):
    # Simple edge detection: find first rising edge in each waveform
    thresh = (np.max(wave1) + np.min(wave1)) / 2
    idx1 = np.argmax(wave1 > thresh)
    idx2 = np.argmax(wave2 > thresh)
    return idx2 - idx1

# --- Generate Square Wave, Monitor, And Calculate Skew ---
with (nidaqmx.Task() as pulse_task):
    # --- Create Internal Counter Channel / Square Wave ---
    pulse_task.co_channels.add_co_pulse_chan_freq(
        square_wave_counter, freq=square_wave_frequency, duty_cycle=square_wave_duty_cycle
    )
    # --- Configure task timing to Continuous ---
    pulse_task.timing.cfg_implicit_timing(
        sample_mode=AcquisitionType.CONTINUOUS
    )
    # --- Export Internal Counter Channel to Terminal ---
    pulse_task.export_signals.ctr_out_event_output_term = square_wave_terminal

    # --- Skew Test ---
    for i in range(num_iterations):
        # --- Configure AI Tasks ---
        with nidaqmx.Task() as ai_task1, nidaqmx.Task() as ai_task2:
            # --- Create Analog Input channel for each device ---
            ai_task1.ai_channels.add_ai_voltage_chan(
                device1_ai, min_val=min_val, max_val=max_val, terminal_config= TerminalConfiguration.DIFF
            )
            ai_task2.ai_channels.add_ai_voltage_chan(
                device2_ai, min_val=min_val, max_val=max_val, terminal_config= TerminalConfiguration.DIFF
            )
            # --- Configure task timing to Finite ---
            ai_task1.timing.cfg_samp_clk_timing(
                sample_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=samples_per_channel
            )
            ai_task2.timing.cfg_samp_clk_timing(
                sample_rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=samples_per_channel
            )

            # --- Start Tasks "Together" (software sync) ---
            ai_task1.start()
            ai_task2.start()

            # --- Start Square Wave Generation ---
            pulse_task.start()

            # --- Acquire Data ---
            data1 = ai_task1.read(number_of_samples_per_channel=samples_per_channel)
            data2 = ai_task2.read(number_of_samples_per_channel=samples_per_channel)

            # --- Stop Square Wave Generation ---
            pulse_task.stop()

            # --- Calculate Skew ---
            skew = find_rising_edge_sample_offset(np.array(data1), np.array(data2))
            skews.append(skew)

            # --- Plot (optional) ---
            plt.figure()
            plt.plot(data1, label="Device 1")
            plt.plot(data2, label="Device 2")
            plt.title(f"Iteration {i+1} - Skew: {skew} samples")
            plt.legend()
            plt.show(block=False)
            key_pressed = False
            while key_pressed == False:
                key_pressed = plt.waitforbuttonpress()
            plt.close()

# --- Statistics ---
skews = np.array(skews)
mean_skew = np.mean(skews)
range_skew = np.ptp(skews)
std_skew = np.std(skews)

print(f"Skew (samples) per iteration: {skews}")
print(f"Mean: {mean_skew:.2f}, Range: {range_skew}, Std Dev: {std_skew:.2f}")