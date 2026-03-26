import os
import sys
import warnings

import cothread
from cothread.catools import DBR_STRING, caget, caput
from numpy import array, concatenate, diag, ones, pi, zeros
from PyQt6.QtWidgets import QMessageBox
from scipy.io import loadmat

from . import deal_rm_sr, imc3

# CONFIG_FILE = os.getcwd() + "/src/fastfeedback_opi/fofb.config"
CONFIG_FILE = "/home/ops/diagnostics/config/fofb.config"
PLANES = 2
TEST_MODE = False
NODAMP = False
NOBEAM = False

BPMS_UNREACHABLE = """<html><body>
<p>The following Libera electron BPMs are currently unreachable:</p>
<p>%s</p>
<p>Before restarting fast feedback, please perform one of the following
actions:</p>
<ul>
<li>Power cycle the currently unreachable Liberas.  If this does not restore
normal function, please leave it switched off.  This is the preferred
action.</li>
<li>Switch off the unreachable Liberas and leave them switched off.</li>
</ul>
<p>Press <b>Cancel</b> or return unless the actions above have been performed.
Press <b>Retry</b> to start fast feedback anyway.</p>
</body></html>
"""


def distribute_rm_and_correctors(golden, mu, bpm_weights, cm_weights):
    print("Dealing response matrix")
    bpmen = caget("SR-DI-EBPM-01:ENABLED")
    hstren = caget("SR-PC-HSTR-01:FAST:ENABLED")
    vstren = caget("SR-PC-VSTR-01:FAST:ENABLED")
    hbpmen = caget("SR-PC-HBPM-01:FAST:ENABLED") + bpmen
    vbpmen = caget("SR-PC-VBPM-01:FAST:ENABLED") + bpmen

    # Fudge golden and bpmen (and bpm_weights if set) so that they're all of
    # length suitable for upload -- needs to be multiple of 4!
    extra = -len(bpmen) % 4
    bpmen = concatenate((bpmen, ones(extra)))
    hbpmen = concatenate((hbpmen, ones(extra)))
    vbpmen = concatenate((vbpmen, ones(extra)))
    golden = [concatenate((m, zeros((extra, m.shape[1])))) for m in golden]
    if bpm_weights:
        bpm_weights = [concatenate((m, zeros(extra))) for m in bpm_weights]
    (names, values) = deal_rm_sr.deal(
        hbpmen, vbpmen, hstren, vstren, golden, mu, bpm_weights, cm_weights
    )
    caput(fbmany("BPMEN"), bpmen, repeat_value=True)
    caput(names, values)


class Configurator:
    def __init__(self, config_file):
        self.CONFIG_FILE = config_file
        self.reread()

    def reread(self):
        ringmode = caget("SR-CS-RING-01:MODE", datatype=DBR_STRING)
        print("Ring mode is", ringmode)
        self.read(ringmode)

    def read(self, mode):
        def py2_range(*args):
            return list(range(*args))

        # This is a horrible hack to make the existing config file work with py3 which
        # handles ranges differently. It basically overwrites the default python range()
        # function with py2_range which is the default python2 range().
        context = {
            "pi": pi,
            "array": array,
            "zeros": zeros,
            "mode": mode,
            "range": py2_range,
        }
        params = {}
        exec(
            compile(open(self.CONFIG_FILE, "rb").read(), self.CONFIG_FILE, "exec"),
            context,
            params,
        )
        for k, v in list(params.items()):
            setattr(self, k, v)

        if NODAMP:
            print("turning off damping")
            self.EPSILON = 0

        rm_file = os.path.join(self.DATAROOT, mode, "GoldenBPMResp.mat")
        print(rm_file)
        rm = loadmat(rm_file, appendmat=False, struct_as_record=False)
        self.rm = [rm["Rmat"][0, 0].Data, rm["Rmat"][1, 1].Data]

        try:
            cm_weights_file = os.path.join(
                self.DATAROOT, mode, "CorrectorCalibrations.mat"
            )
            raw_weights = loadmat(cm_weights_file, appendmat=False)
            hcm = 1.0 / raw_weights["HCM"]["physicsvals"][0][0]
            vcm = 1.0 / raw_weights["VCM"]["physicsvals"][0][0]
            self.cm_weights = [diag(hcm.T[0]), diag(vcm.T[0])]
        except OSError:
            print("Warning: Not using corrector magnet weights")
            self.cm_weights = None

        # BPM Weighting is not expected to be used operationally
        try:
            bpm_weights_file = os.path.join(self.DATAROOT, mode, "BpmCalibrations.mat")
            # Hack to work around depreciation warning being shown for missing file
            # This isn't require for newer versions of scipy
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.bpm_weights = loadmat(bpm_weights_file, appendmat=False)["weights"]
            print("Using BPM weights")
        except OSError:
            self.bpm_weights = None

        # get taps for both planes
        delay = self.DELAY
        ts = self.Ts
        eta = array([self.ETAX, self.ETAY]) * delay * ts
        btaps = zeros((self.MAX_TAPS, PLANES))
        ataps = zeros((self.MAX_TAPS, PLANES))
        for n in range(PLANES):
            btaps[:, n], ataps[:, n] = imc3.imc(
                eta[n], epsilon=self.EPSILON, delay=delay, lowpass=self.CUTOFF[n]
            )
        self.btaps = btaps
        self.ataps = ataps


def fbmany(suffix):
    return ["SR%02dA-CS-FOFB-01:%s" % (n + 1, suffix) for n in range(CONFIG.CELLS)]  # noqa: UP031


def cormany(plane, suffix):
    return [
        "SR%02dA-PC-%sSTR-%02d:%s" % (a + 1, plane, i + 1, suffix)  # noqa: UP031
        for a in range(CONFIG.CELLS)
        for i in range(CONFIG.MAGS)
    ]


# I want to remove Qt GUI dependencies altogether, but for the moment we have
# this pesky little popup in the middle of the code.  This function fires up the
# gui just for the sake of this message.  Returns true if retry selected.
def qt_popup(title, body):
    cothread.iqt()
    response = QMessageBox.warning(
        None,
        title,
        body,
        QMessageBox.StandardButton.Cancel
        | QMessageBox.StandardButton.Default
        | QMessageBox.StandardButton.Escape,
        QMessageBox.StandardButton.Retry,
    )
    return response == QMessageBox.StandardButton.Retry


def bpm_name(index):
    """Converts index 0.. to full BPM name."""
    return deal_rm_sr.bpm_names[deal_rm_sr.bpm_concentrator_ids.index(index)]


def stop_feedback():
    print("Stopping FOFB")
    caput(fbmany("FASTART"), 0)
    # make sure the disable flag is cleared
    caput(fbmany("DISAH"), 0)
    caput(fbmany("DISAV"), 0)


def start_feedback():
    print("Starting FOFB")

    # stop first
    caput(fbmany("FASTART"), 0)
    # make sure the disable flag is cleared
    caput(fbmany("DISAH"), 0)
    caput(fbmany("DISAV"), 0)

    # check disabled bpms
    enabled = caget("SR-DI-EBPM-01:ENABLED")
    unreachable = (enabled == 2).nonzero()[0]
    if len(unreachable) > 0:
        # If any BPMs are unreachable, don't proceed without confirmation.
        if not qt_popup(
            "One or more BPMS unreachable",
            BPMS_UNREACHABLE % ", ".join(map(bpm_name, unreachable)),
        ):
            return

    maskcell = zeros(CONFIG.CELLS, dtype=int)
    caput(fbmany("FMASK"), maskcell, repeat_value=True)

    distribute_rm_and_correctors(
        CONFIG.rm, CONFIG.MU, CONFIG.bpm_weights, CONFIG.cm_weights
    )

    # post-mortem arm
    caput(fbmany("PMODE"), 0)
    # safe soft stop
    caput(fbmany("SOFTSTOP"), 1)
    # disable internal test mode!
    if TEST_MODE:
        caput(fbmany("TESTMODE"), 2)
    else:
        caput(fbmany("TESTMODE"), 0)
    # enable corrector limit (uses anti-windup detect)
    caput(fbmany("AWR"), 1)
    # IIR taps

    if NOBEAM:
        CONFIG.ataps = CONFIG.ataps * 0
        CONFIG.btaps = CONFIG.btaps * 0

    caput(fbmany("BTAPSX"), CONFIG.btaps[:, 0], repeat_value=True)
    caput(fbmany("ATAPSX"), CONFIG.ataps[:, 0], repeat_value=True)
    caput(fbmany("BTAPSY"), CONFIG.btaps[:, 1], repeat_value=True)
    caput(fbmany("ATAPSY"), CONFIG.ataps[:, 1], repeat_value=True)
    # maximum abs corrector (A)
    caput(fbmany("MABSC"), CONFIG.CORRECTOR_LIMIT)
    # maximum abs bpm (nm)
    if TEST_MODE or NOBEAM:
        caput(fbmany("MABSB"), 1e20)
    else:
        caput(fbmany("MABSB"), CONFIG.ORBIT_LIMIT, repeat_value=True)
    # maximum number of BPMs out of bounds before tripping
    caput(fbmany("MAOOB"), CONFIG.BPMS_OUTSIDE_ORBIT_LIMIT)

    # turn on output
    if NOBEAM:
        caput(fbmany("APPLY"), 1)
    else:
        caput(fbmany("APPLY"), 1)
    # limit tracks output at 2Hz
    caput(fbmany("TRACK"), CONFIG.TRACK_RATE)
    # reset IIR
    caput(fbmany("ARESET"), 1)

    cothread.Sleep(0.1)
    # reset error counters
    caput(fbmany("EWRITE"), zeros(8, dtype=int), repeat_value=True)
    # enable FOFB flag
    caput(fbmany("FASTART"), 1)


if len(sys.argv) > 1 and sys.argv[1] == "TEST":
    TEST_MODE = True
if len(sys.argv) > 1 and sys.argv[1] == "NODAMP":
    NODAMP = True
if len(sys.argv) > 1 and sys.argv[1] == "NOBEAM":
    NOBEAM = True
if len(sys.argv) >= 3:
    CONFIG_FILE = sys.argv[2]
CONFIG = Configurator(CONFIG_FILE)


def main():
    if sys.argv[1] == "start":
        start_feedback()
    elif sys.argv[1] == "stop":
        stop_feedback()
    else:
        print("Unknown action")
        sys.exit(1)


if __name__ == "__main__":
    main()
