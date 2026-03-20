from __future__ import division
from numpy import pi, exp

def imc(eta = 400e-6,
        epsilon = 1e-5,
        lowpass = 100 * 2 * pi,
        delay = 4, NTAPS = 9, Ts = 100e-6):
    
    "symbolic expression for IMC from Maxima"
    
    assert(delay > 0)
    assert(delay < NTAPS)

    lam = exp(-Ts / eta)
    tau = exp(-Ts / (1 / lowpass))
    
    b = [(-(1-epsilon)*lam-epsilon+1)/(1-tau),
         -((epsilon-1)*lam-epsilon+1)*tau/(1-tau)] + [0] * (NTAPS - 2)

    a = [(lam*tau-lam)/(1-tau)] + \
        [0] * (delay - 1) + \
        [(((epsilon-1)*lam-epsilon+1)*tau+(1-epsilon)*lam+epsilon-1)/(1-tau)] + \
        [0] * (NTAPS - 2 - delay + 1)

    return [b, a]

if __name__ == "__main__":
    print(imc(delay = 4))
