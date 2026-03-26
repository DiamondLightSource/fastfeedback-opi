import os

from cothread.catools import caget
from numpy import nonzero, zeros

from . import invert, table

# response matrix is (bpms, correctors)
# inverse is (correctors, bpms)

MAXCORS = 9
MAXBPMS = 176


def filedir():
    return os.path.dirname(os.path.realpath(__file__))


cname = os.path.join(filedir(), "correctors.txt")
bname = os.path.join(filedir(), "bpms.txt")

bt = table.table(bname)
bpm_concentrator_ids = [int(m) - 1 for m in bt.cols["mml1"]]
bpm_names = bt.cols["epics"]


def deal(
    hbpmen, vbpmen, hstren, vstren, rm, mu=[1, 1], bpm_weights=None, cm_weights=None
):

    ct = table.table(cname)
    bt = table.table(bname)

    channels = [e + ":I" for e in ct.cols["epics"]]
    readback = caget(channels)

    (badhbpm,) = nonzero(hbpmen != 0)
    (badvbpm,) = nonzero(vbpmen != 0)
    (badhstr,) = nonzero(hstren != 0)
    (badvstr,) = nonzero(vstren != 0)

    irm = invert.invert(
        rm,
        mu,
        badbpm=[badhbpm, badvbpm],
        badcm=[badhstr, badvstr],
        bpm_weights=bpm_weights,
        cm_weights=cm_weights,
    )

    slices = {}

    bmml0 = [int(m) - 1 for m in bt.cols["mml1"]]
    bfaid0 = [int(m) - 1 for m in bt.cols["faid"]]

    for i in range(len(ct.cols["epics"])):
        e = ct.cols["epics"][i]
        mml0 = int(ct.cols["mml1"][i]) - 1
        idx = int(ct.cols["farow"][i])
        if "HSTR" in e or "HSCOR" in e:
            target = ct.cols["ioc"][i] + ":IRMX"
            irmorbit = irm[0].irm[mml0, :]
            en_target = ct.cols["ioc"][i] + ":HSTREN"
            st_target = ct.cols["ioc"][i] + ":HSET"
            src = hstren[mml0]
        else:
            target = ct.cols["ioc"][i] + ":IRMY"
            irmorbit = irm[1].irm[mml0, :]
            en_target = ct.cols["ioc"][i] + ":VSTREN"
            st_target = ct.cols["ioc"][i] + ":VSET"
            src = vstren[mml0]
            idx = idx - MAXCORS

        vec = slices.setdefault(st_target, zeros(MAXCORS))
        vec[idx] = readback[i]

        vec = slices.setdefault(en_target, zeros(MAXCORS))
        vec[idx] = src

        vec = slices.setdefault(target, zeros((MAXCORS, MAXBPMS)))
        vec[idx, bfaid0] = irmorbit[bmml0] / 1e6

    values = []

    keys = sorted(slices.keys())
    for k in keys:
        v = slices[k].ravel()
        values.append(v)

    return (keys, values)
