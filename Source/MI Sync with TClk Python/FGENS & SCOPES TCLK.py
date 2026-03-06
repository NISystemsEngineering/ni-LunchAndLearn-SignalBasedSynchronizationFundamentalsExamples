# loopback_fgen_scope.py
import nitclk
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

rel_initial_x_list = []          # list for calculating phase
hardware_session_list = []       # list for TCLK Sync

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

    # Append fgen session reference to TCLK Sync list.
    # Replaces Triggering Setup.
    hardware_session_list.append(fgen)

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
        ref_position=0,       # 50% pre/post trigger balance
        num_records=1,
        enforce_realtime=True
    )  # Parameters as in NI-SCOPE docs/examples. [7](https://niscope.readthedocs.io/en/latest/examples.html)[8](https://www.ni.com/docs/en-US/bundle/ni-scope-labview-api-ref/page/instr-lib/niscope/niscope-llb/niscope-configure-horizontal-timing-vi.html)

    # Set up all the triggers
    if scopes.index(scope) == (len(SCOPE_RESOURCES)-1):
        #Last scope in the list - configure analog edge trigger. No additional triggers are needed because of TCLK.
        scope.configure_trigger_edge(
            trigger_source=CHANNEL,
            level=0.0,
            trigger_coupling=niscope.TriggerCoupling.DC,
            slope=niscope.enums.TriggerSlope.POSITIVE
        )  # Edge trigger API & enum documented in NI-SCOPE reference. [2](https://nimi-python.readthedocs.io/en/master/niscope.html)

    # Append scope session reference to TCLK Sync list.
    # Replaces Triggering Setup.
    # Scopes and fGENs in same list.
    hardware_session_list.append(scope)

# ---------------
# Run & Acquire
# ---------------

# Initiate the TCLK for list
# Trigger routing, delay calculations, and resulting synchronization happen automatically
nitclk.configure_for_homogeneous_triggers(hardware_session_list)
nitclk.synchronize(hardware_session_list, 200e-9)

for fgen in fgens:
    # Allow fGENs to output
    fgen.channels[CHANNEL].output_enabled = True

# Initiate the generation & acquisition together
nitclk.initiate(hardware_session_list)

samples_array = []
for scope in scopes:
    #Fetch
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