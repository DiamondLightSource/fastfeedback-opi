from numpy import *
from numpy.linalg import *

# These weights seem to be the best so far!
# Weights = [1, 0.2, 0.05, 0.05, 0.05, 0.2, 1]

def struct(**kw): return type("struct", (object,), kw)
    
def invert(resp, mu = [1.0, 1.0],
           badbpm = [[],[]], badcm = [[],[]],
           progress = lambda: None, full = 0, bpm_weights = None, cm_weights = None):

    progress()
    irms = []
    for p in [0, 1]:
        rmq = resp[p]
        (MAXBPM, MAXCM) = rmq.shape
        irm = zeros((MAXCM, MAXBPM))  # Fill in nonzero elements later

        wb = mat(eye(MAXBPM)) if bpm_weights is None else bpm_weights[p]
        wc = mat(eye(MAXCM)) if cm_weights is None else cm_weights[p]

        goodb = arange(MAXBPM)
        goodc = arange(MAXCM)
        bpm = setdiff1d(goodb, badbpm[p])
        cm = setdiff1d(goodc, badcm[p])
        rm = mat(rmq[ix_(bpm, cm)])
        wbm = mat(wb[ix_(bpm, bpm)])
        wcm = mat(wc[ix_(cm, cm)])
        # Matrix might have no elements if all BPMs or correctors are disabled
        if (wbm * rm * wcm).size == 0:
            irms.append(struct(
                rm=rm, irm=irm, u=array([]), s=array([]), vt=array([])))
            continue
        (u, s, vt) = svd(wbm * rm * wcm, full_matrices = full)

        # Scale mu when weights matrices are used
        if cm_weights or bpm_weights:
            s0 = svd(rm)[1][0]
            mu[p] = mu[0] * s[0] / s0

        progress()
        sinv = zeros((vt.shape[0], u.shape[1]))
        for n in range(len(s)):
            sinv[n, n] = s[n] / (s[n]**2 + mu[p]**2)
        partial_irm = wcm * transpose(vt) * sinv * transpose(u) * wbm
        irm[ix_(cm, bpm)] = partial_irm
        res = struct(rm = rm, irm = irm,
                     u = u, s = s, vt = vt,
                     badbpm = badbpm)
        irms.append(res)
    return irms

if __name__ == "__main__":
    from scipy.io import loadmat
    from pylab import imshow, figure
    print("invert test")
    RMFILE = "/home/diamond/common/matlab/middlelayer/2-0" \
         "/machine/diamondopsdata/SR/GoldenBPMResp.mat"
    rm = loadmat(RMFILE)
    sol = invert(rm)
    figure()
    imshow(sol[0].irm)
    figure()
    imshow(sol[1].irm)
    figure()
    # show slow scaling
    mu = [1e-2, 1e-2]
    p = 0
    s = sol[p].s
    w = zeros((sol[p].vt.shape[0], sol[p].u.shape[1]))
    for n in range(len(s)):
        w[n, n] = s[n] / (s[n]**2 + mu[p])
    plot(range(168), diag(w), 'b', range(168), 1/s, 'r')
    raw_input()
    
