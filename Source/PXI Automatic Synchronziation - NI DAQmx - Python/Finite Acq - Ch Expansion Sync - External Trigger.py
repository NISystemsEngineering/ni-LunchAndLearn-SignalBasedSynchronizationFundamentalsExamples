# Highlights easy NI DAQmx Synchronization on PXI
#   Both devices are managed in a single task
#   Both devices automatically start together even though only one device is acquiring the trigger
#   Both devices automatically synchronize to PXI Clock10 and will not drift over time

# An NI FPGA is generating an output that is wired to both devices
# The same NI FPGA is also generating a single trigger that is routed to one device

import time
import numpy as np
import matplotlib.pyplot as plt
from nifpga import Session
import nidaqmx
from nidaqmx.constants import (
    TerminalConfiguration,
    AcquisitionType,
    Edge,
    READ_ALL_AVAILABLE,
)
from nidaqmx.stream_readers import AnalogMultiChannelReader

# DAQmx constants
RATE = 2_000_000.0
SECONDS_PER_READ = 2
SAMPLES_PER_CHAN = int(RATE * SECONDS_PER_READ)
PHYSICAL_CHANNELS = "DaqDev1/ai1,DaqDev2/ai1"
TRIGGER_SOURCE = "/DaqDev1/PFI0"
THRESHOLD = 2
SAMPLE_PERIOD = 1.0 / RATE

# FPGA constants
BITFILE = "PulseAndTriggerGeneration.lvbitx"
RESOURCE = "RIO3"

def find_threshold_crossing(array, threshold, direction="rising"):
    # --- Determine the first threshold crossing in an array, then interpolate to find the decimal sample ---
    arr = np.asarray(array, dtype=float)
    shifted = arr - threshold
    if direction == "rising":
        mask = (shifted[:-1] < 0) & (shifted[1:] >= 0)
    elif direction == "falling":
        mask = (shifted[:-1] >= 0) & (shifted[1:] < 0)
    else:  # both
        mask = np.diff(np.signbit(shifted))
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    i = indices[0]
    fraction = (threshold - arr[i]) / (arr[i + 1] - arr[i])
    return i + fraction


def main():
    # Waits on a trigger then continuously acquires from two different devices and calculate synchronization

    with nidaqmx.Task() as task:
        data = np.zeros((2, SAMPLES_PER_CHAN), dtype=np.float64)

        # Sets up NI FPGA device that is generating triggers and monitored output signal
        session = Session(bitfile=BITFILE, resource=RESOURCE, no_run=True)
        session.registers["Trigger"].write(False)
        session.registers["Requested Output Frequency"].write(1)
        session.registers["Enable Output"].write(False)
        session.run()

        # Add channels from multiple devices into a single task
        task.ai_channels.add_ai_voltage_chan(
            PHYSICAL_CHANNELS,
            min_val=-5.0,
            max_val=5.0,
            terminal_config=TerminalConfiguration.DIFF,
        )

        task.timing.cfg_samp_clk_timing(
            rate=RATE,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=SAMPLES_PER_CHAN,
        )

        # Sets up single trigger but all devices will automatically start together
        task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source=TRIGGER_SOURCE,
            trigger_edge=Edge.RISING,
        )

        # Sets up single reader but all devices will acquire and read together
        reader = AnalogMultiChannelReader(task.in_stream)

        # Start NI DAQmx Acquisition
        task.start()
        print(f"Task started. Waiting for external trigger on {TRIGGER_SOURCE}...")

        # Start NI FPGA signal generation
        session.registers["Enable Output"].write(True)

        # Send Start Trigger from NI FPGA
        time.sleep(.75)
        session.registers["Trigger"].write(True)

        try:

            reader.read_many_sample(
                data,
                number_of_samples_per_channel=SAMPLES_PER_CHAN,
                timeout=10.0,
            )

            # Calculate synchronization sample offset
            ch1_sample_offset = find_threshold_crossing(data[0, :], THRESHOLD)
            ch2_sample_offset = find_threshold_crossing(data[1, :], THRESHOLD)
            sample_offset = abs(ch1_sample_offset - ch2_sample_offset)

            # Calculate synchronization time offset
            time_offset_sec = abs(sample_offset * SAMPLE_PERIOD)

            # Report results
            print(
                f"Sample Offset: {sample_offset:.6f}, "
                f"Time Offset (secs): {time_offset_sec:.4e}"
            )

            # --- Plot (optional) ---
            plt.figure()
            plt.plot(data[0, :], label="Device 1")
            plt.plot(data[1, :], label="Device 2")
            plt.title(f"Skew: {sample_offset: .6f} samples")
            plt.legend(loc='upper right')
            plt.show(block=False)
            key_pressed = False
            while key_pressed == False:
                key_pressed = plt.waitforbuttonpress()
            plt.close()

        finally:
            # Stop NI DAQmx Acquisition Task
            task.stop()

            # Stop NI FPGA signal generation
            session.registers["Trigger"].write(False)
            session.registers["Enable Output"].write(False)
            session.close(reset_if_last_session=True)


if __name__ == "__main__":
    main()