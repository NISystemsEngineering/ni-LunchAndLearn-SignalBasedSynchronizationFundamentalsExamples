import nidaqmx
from nidaqmx.constants import (Edge, AcquisitionType, LineGrouping, TerminalConfiguration)
import numpy as np
import keyboard
import time

# --- Analog Input User Configuration ---
device1_ai = "MIODAQ/ai0"
device2_ai = "cDAQ9189-2628AACMod1/ai0"
sample_rate = 100_000
samples_per_channel = sample_rate  # 1 seconds
min_val = -10.0
max_val = 10.0

# --- Digital Start Trigger User Configuration ---
device1_trigger_output_line = "MIODAQ/port0/line2"
device1_trigger_input = "/MIODAQ/PFI4"
device2_trigger_input = "/cDAQ9189-2628AACMod5/PFI4"

# --- Digital Square Wave User Configuration ---
square_wave_counter = "MIODAQ/ctr0"
square_wave_terminal = "/MIODAQ/PFI0"
square_wave_frequency = 1.0
square_wave_duty_cycle = 0.5

def find_rising_edge_sample_offset(wave1, wave2):
    # Simple edge detection: find first rising edge in each waveform
    thresh = (np.max(wave1) + np.min(wave1)) / 2
    idx1 = np.argmax(wave1 > thresh)
    idx2 = np.argmax(wave2 > thresh)
    return idx1 - idx2

# --- Generate Square Wave, Monitor, And Calculate Skew ---
with nidaqmx.Task() as pulse_task, nidaqmx.Task() as start_trigger_do_task, nidaqmx.Task() as ai_task1, nidaqmx.Task() as ai_task2:
    # --- Create Internal Counter Channel / Square Wave ---
    pulse_task.co_channels.add_co_pulse_chan_freq(
        square_wave_counter,
        freq=square_wave_frequency,
        duty_cycle=square_wave_duty_cycle
    )
    # --- Configure task timing to Continuous ---
    pulse_task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)

    # --- Export Internal Counter Channel to Terminal ---
    pulse_task.export_signals.ctr_out_event_output_term = square_wave_terminal

    # --- Configure Start Trigger Digital Output Task ---
    start_trigger_do_task.do_channels.add_do_chan(
        device1_trigger_output_line,
        line_grouping=LineGrouping.CHAN_PER_LINE
    )
    start_trigger_do_task.write(False)  # Ensure trigger is low

    # --- Create Analog Input channel for each device ---
    ai_task1.ai_channels.add_ai_voltage_chan(
        device1_ai,
        min_val=min_val,
        max_val=max_val,
        terminal_config=TerminalConfiguration.DIFF
    )
    ai_task2.ai_channels.add_ai_voltage_chan(
        device2_ai,
        min_val=min_val,
        max_val=max_val,
        terminal_config=TerminalConfiguration.DIFF
    )

    # --- Configure task timing to Continuous ---
    ai_task1.timing.cfg_samp_clk_timing(
        sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=10*samples_per_channel
    )
    ai_task2.timing.cfg_samp_clk_timing(
        sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=10*samples_per_channel
    )

    # --- Configure task start triggers ---
    ai_task1.triggers.start_trigger.cfg_dig_edge_start_trig(
        device1_trigger_input,
        trigger_edge=Edge.RISING
    )
    ai_task2.triggers.start_trigger.cfg_dig_edge_start_trig(
        device2_trigger_input,
        trigger_edge=Edge.RISING
    )

    # --- Start Tasks ---
    ai_task1.start()
    ai_task2.start()

    # --- Fire Trigger ---
    start_trigger_do_task.write(True)

    # --- Start Square Wave Generation ---
    pulse_task.start()

    key_pressed = False
    while not key_pressed:
        # --- Read Data ---
        data1 = ai_task1.read(number_of_samples_per_channel=samples_per_channel, timeout=2)
        data2 = ai_task2.read(number_of_samples_per_channel=samples_per_channel, timeout=2)

        # --- Analyze Skew ---
        skew = find_rising_edge_sample_offset(np.array(data1), np.array(data2))
        print(f"Skew = {skew} samples")
        time.sleep(0.5)

        # Exit condition (e.g., key press or fixed number of iterations)
        key_pressed = keyboard.is_pressed("q")


    # --- Stop Analog Input Acquisition Tasks ---
    ai_task1.stop()
    ai_task2.stop()

    # --- Reset Start Trigger ---
    start_trigger_do_task.write(False)

    # --- Stop Square Wave Generation ---
    pulse_task.stop()