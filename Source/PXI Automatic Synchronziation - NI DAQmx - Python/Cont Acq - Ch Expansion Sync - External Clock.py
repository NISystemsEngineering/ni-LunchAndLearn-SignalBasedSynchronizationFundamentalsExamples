# Highlights easy NI DAQmx Synchronization on PXI
#   Both devices are managed in a single task
#   Both devices automatically start together even though only one device is acquiring the trigger
#   Both devices automatically synchronize to PXI Clock10 and will not drift over time

# An NI FPGA is generating an output that is wired to both devices
# The same NI FPGA is also generating a single trigger that is routed to one device

import time
import numpy as np
from nifpga import Session
import nidaqmx
from nidaqmx.constants import (
    TerminalConfiguration,
    AcquisitionType,
    Edge,
    READ_ALL_AVAILABLE,
)
from nidaqmx.stream_readers import AnalogMultiChannelReader

import math

# DAQmx constants
RATE = 2_000_000.0
SAMPLES_PER_CHAN = 20_000
PHYSICAL_CHANNELS = "DaqDev1/ai1,DaqDev2/ai1"
CLOCK_SOURCE = "/DaqDev1/PFI0"

# FPGA constants
BITFILE = "PulseAndClockGeneration.lvbitx"
RESOURCE = "RIO3"

def round_by_first_two_digits(n):
    if n == 0:
        return 0
    sign = -1 if n < 0 else 1
    n = abs(n)
    digits = int(math.log10(n)) + 1
    scale = 10 ** (digits - 2)   # keep top 2 digits
    top_two = n / scale
    rounded = round(top_two) * scale
    return sign * rounded

def main():
    # Waits on a trigger then continuously acquires from two different devices and calculate synchronization

    with nidaqmx.Task() as task:
        data = np.zeros((2, SAMPLES_PER_CHAN), dtype=np.float64)

        # Sets up NI FPGA device that is generating triggers and monitored output signal
        session = Session(bitfile=BITFILE, resource=RESOURCE, no_run=True)
        session.registers["Reset"].write(True)
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
            source=CLOCK_SOURCE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=SAMPLES_PER_CHAN,
        )

        # Sets up single reader but all devices will acquire and read together
        reader = AnalogMultiChannelReader(task.in_stream)

        # Start NI DAQmx Acquisition
        task.start()

        # Start NI FPGA signal generation
        session.registers["Reset"].write(False)
        start_time = time.time()
        session.registers["Enable Output"].write(True)

        try:
            for i in range(13):
                # Read from NI DAQmx device; Acq time will change based on variable external sample clock rate
                reader.read_many_sample(
                    data,
                    number_of_samples_per_channel=SAMPLES_PER_CHAN,
                    timeout=25.0,
                )

                # Calculate sample time, sample rate
                stop_time = time.time()
                acquisition_time = stop_time - start_time
                acquisition_frequency = data[0].size / (acquisition_time)
                rounded_acquisition_frequency = round_by_first_two_digits(acquisition_frequency)
                start_time = stop_time

                # Report results
                print(f"Acquisition Time: {acquisition_time} sec")
                print(f"Samples acquired: {data[0].size}")
                print(f"Acquisition Frequency: {rounded_acquisition_frequency} Hz")

        finally:
            # Stop NI DAQmx Acquisition Task
            task.stop()

            # Stop NI FPGA signal generation
            session.registers["Reset"].write(True)
            session.registers["Enable Output"].write(False)
            session.close(reset_if_last_session=True)


if __name__ == "__main__":
    main()