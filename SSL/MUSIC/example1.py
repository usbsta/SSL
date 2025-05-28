import numpy as np
from numpy.linalg import eigh

# ---- PARAMETERS -------------------------------------------------
c = 343.0           # speed of sound [m/s]
fs = 8000           # sampling rate [Hz]
M  = 8              # sensors
d  = 0.05           # spacing [m]
Ns = 4096           # samples
DOA = np.deg2rad([20, -10])  # radians

# ---- STEERING MATRIX -------------------------------------------
k = 2*np.pi*1000/c                     # wavenumber at 1 kHz
mics = np.arange(M)*d
A = np.exp(1j*k*np.outer(mics, np.sin(DOA)))  # size M x D

# ---- SYNTHETIC DATA --------------------------------------------
s = np.exp(1j*2*np.pi*1000*np.arange(Ns)/fs)         # baseband tone
S = np.vstack([s, 0.7*s])                            # two sources
X = A @ S + 0.316*np.random.randn(M, Ns)             # + white noise

# ---- SAMPLE COVARIANCE -----------------------------------------
R = (X @ X.conj().T) / Ns

# ---- EVD & NOISE SUBSPACE --------------------------------------
vals, vecs = eigh(R)                 # ascending order
D = 2                                 # number of sources
Un = vecs[:, :M-D]                    # noise subspace
PN = Un @ Un.conj().T

# ---- MUSIC SPECTRUM --------------------------------------------
angles = np.deg2rad(np.arange(-90, 91, 0.1))
spec = []
for u in angles:
    a = np.exp(1j*k*mics*np.sin(u))
    spec.append(1 / np.abs(a.conj() @ PN @ a))
spec = 10*np.log10(np.array(spec)/np.max(spec))
