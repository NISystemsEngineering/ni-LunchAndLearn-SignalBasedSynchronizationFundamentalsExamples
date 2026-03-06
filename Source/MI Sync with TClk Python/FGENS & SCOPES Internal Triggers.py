# loopback_fgen_scope.py
import numpy as np
import nifgen
import niscope
import matplotlib.pyplot as plt

# ---- Configure these for your bench ----
FGEN_RESOURCES  = ["FGEN1", "FGEN2"]   # e.g., "PXI1Slot2" or "Dev1" in NI-MAX
SCOPE_RESOURCES = ["SCOPE1", "SCOPE2"]  # e.g., "PXI1Slot3" or "Dev2"
CHANNEL        = "0"       # first output / input channel
FREQ_HZ        = 1_000_000
AMP_VPP        = 1      # V pk-pk for NI-FGEN
SAMPLE_RATE    = 100_000_000  # 100 MS/s
NUM_SAMPLES    = 200_000
V_RANGE        = 1.5       # scope vertical range in V (adjust to fit your signal)

# Initiate session refs
fgens = [nifgen.Session(FGEN_RESOURCE) for FGEN_RESOURCE in FGEN_RESOURCES]
scopes = [niscope.Session(SCOPE_RESOURCE) for SCOPE_RESOURCE in SCOPE_RESOURCES]

rel_initial_x_list = []     # list for calculating phase

# ------------------
# Configure NI-FGEN
# ------------------
for fgen in fgens:
    # Output a simple standard sine for the demo (FUNC mode)
    fgen.output_mode = nifgen.OutputMode.FUNC   # Select "standard function" mode
    fgen.channels[CHANNEL].configure_standard_waveform(
        waveform=nifgen.Waveform.SINE,
        amplitude=AMP_VPP,           # V pk-pk
        frequency=FREQ_HZ,           # Hz
        dc_offset=0.0,               # V
        start_phase=0.0              # degrees
    )  # Basic usage mirrors NI docs. [1](https://nimi-python.readthedocs.io/en/master/nifgen.html)

    # Disable then enable output around initiate for clean start
    fgen.channels[CHANNEL].output_enabled = False

    # Look for start from last configured scope
    fgen.start_trigger_type = nifgen.enums.StartTriggerType.DIGITAL_EDGE
    fgen.digital_edge_start_trigger_edge = nifgen.enums.StartTriggerDigitalEdgeEdge.RISING
    fgen.digital_edge_start_trigger_source = 'PXI_Trig1'

# --------------------
# Configure NI-SCOPE
# --------------------
for scope in scopes:
    # Vertical path on channel 0: DC coupling, requested range V_RANGE
    scope.channels[CHANNEL].configure_vertical(
        range=V_RANGE,
        coupling=niscope.VerticalCoupling.DC,
        offset=0.0,
        probe_attenuation=1.0,
        enabled=True
    )  # Signature & usage per NI SCOPE API examples. [7](https://niscope.readthedocs.io/en/latest/examples.html)[2](https://nimi-python.readthedocs.io/en/master/niscope.html)

    # Horizontal timing (sample rate, record length, reference position)
    scope.configure_horizontal_timing(
        min_sample_rate=SAMPLE_RATE,
        min_num_pts=NUM_SAMPLES,
        ref_position=0,
        num_records=1,
        enforce_realtime=True
    )  # Parameters as in NI-SCOPE docs/examples. [7](https://niscope.readthedocs.io/en/latest/examples.html)[8](https://www.ni.com/docs/en-US/bundle/ni-scope-labview-api-ref/page/instr-lib/niscope/niscope-llb/niscope-configure-horizontal-timing-vi.html)

    # Set up all the triggers
    if scopes.index(scope) >= (len(SCOPE_RESOURCES)-1):
        #Last scope in the list - configure analog edge trigger
        scope.configure_trigger_edge(
            trigger_source=CHANNEL,
            level=0.0,
            trigger_coupling=niscope.TriggerCoupling.DC,
            slope=niscope.enums.TriggerSlope.POSITIVE
        )  # Edge trigger API & enum documented in NI-SCOPE reference. [2](https://nimi-python.readthedocs.io/en/master/niscope.html)

        #Export ready for start (goes to fGEN) & ref trigger (goes to other scopes)
        scope.ready_for_start_event_output_terminal = 'PXI_Trig1'
        scope.exported_ref_trigger_output_terminal = 'PXI_Trig0'
    else:
        #Configure all scopes, but the last one, with a digital trigger.
        scope.configure_trigger_digital(
            trigger_source='PXI_Trig0',
            slope=niscope.enums.TriggerSlope.POSITIVE
        )

# ---------------
# Run & Acquire
# ---------------
# Initiate the fGEN
for fgen in fgens:
    fgen.initiate()  # Arms generation; auto-aborts on exit
    fgen.channels[CHANNEL].output_enabled = True

# Initiate the scope acquisition
# The last scope in the list is configured to generate the main trigger that kicks everything off.
for scope in scopes:
    scope.initiate()  # Arms acquisition; auto-aborts on exit

samples_array = []
for scope in scopes:
    # Fetch
    wfm = scope.channels[CHANNEL].fetch(num_samples=NUM_SAMPLES)[0]  # returns list

    # -----------------
    # Basic post-stats for individual session
    # -----------------
    samples = np.asarray(wfm.samples, dtype=float)
    samples_array.append(samples)
    rel_initial_x_list.append(wfm.relative_initial_x)
    vpp_meas = float(np.max(samples) - np.min(samples))
    vrms_est = float(np.sqrt(np.mean(samples**2)))

    # Very rough frequency estimate using zero-crossings (for demo)
    zero_crossings = np.where(np.diff(np.signbit(samples)))[0]
    est_freq = float('nan')
    if len(zero_crossings) > 1:
        # Average period based on sample intervals
        periods = np.diff(zero_crossings) / SAMPLE_RATE
        if len(periods) > 0:
            est_freq = 1.0 / np.mean(periods) / 2.0  # two zero-crossings per cycle (±)

    #Report out individual session
    print(f"Scope: {SCOPE_RESOURCES[scopes.index(scope)]}")
    print(f"Fetched {len(samples)} samples @ {SAMPLE_RATE/1e6:.1f} MS/s")
    print(f"Measured Vpp ≈ {vpp_meas:.3f} V, Vrms ≈ {vrms_est:.3f} V, f_est ≈ {est_freq/1e6:.6f} MHz")
    print(f"Triggered Time ≈ {wfm.relative_initial_x:.12f} sec")
    print("")

# -----------------
# Calculate phase across all sessions with 10 MHz backplane
# -----------------
max_rel = max(rel_initial_x_list)
min_rel = min(rel_initial_x_list)
delta_ref = (max_rel - min_rel)
phase_ref = (delta_ref / (1/10000000))* 360
print(f"Scope to Scope Phase ≈ {phase_ref:.6f} degrees @ 10 MHz")

# -----------------
# Close out the sessions
# -----------------
for scope in scopes:
    scope.close()
for fgen in fgens:
    fgen.close()

# --- Plot (optional) ---
plt.figure()
for scope in scopes:
    plt.plot(samples_array[scopes.index(scope)], label=f"Scope: {SCOPE_RESOURCES[scopes.index(scope)]}")
plt.title(f"Scope Trace")
plt.legend()
plt.show()