import inspect

# Green's function parameters
beta   = 45.0  # Inverse temperatura
wM     = 4.0  # Cutoff frequency
N      = 4    # Number of electrons
nk_lin = 8    # Number of k samples per axis

# Hamiltonian parameters
FT     = 0.13 # Jahn-Teller t shell 1st order
gJT    = 0.2 # Electron-distortion coupling constant
theta  = 0 # Angle of distortion space
racahA = 6.40  # Racah parameter
racahB = 0.12  # Racah parameter
racahC = 0.552 # Racah parameter
xiSO   = 0.02 # Spin-orbit strength
hoppampl = 1.5 # Hopping amplitude
pdd    = 0.57 # Slater-Koster
ddd    = 0.17 # Slater-Koster

# Electric field
e0y    = 1 # Ey/Ex
e0z    = 0 # Ez/Ex
phiy   = 90 # Phase on sexagesimal degrees
phiz   = 0 # Phase on sexagesimal degrees


def get_parameters_names():
    current_frame = inspect.currentframe()
    caller_globals = current_frame.f_globals
    
    # Collect variable names (excluding functions and imports)
    return [name for name, value in caller_globals.items() if not callable(value) and not name.startswith("__")]