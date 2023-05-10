import numpy as np
import scipy.constants as sc

c = sc.speed_of_light
G = sc.gravitational_constant
h = sc.Planck

yr = sc.Julian_year
day = sc.day
fyr = 1.0 / yr

AU = sc.astronomical_unit
ly = sc.light_year
pc = sc.parsec
kpc = pc * 1.0e3
Mpc = pc * 1.0e6
Gpc = pc * 1.0e9

GMsun = 1.327124400e20
Msun = GMsun / G
Rsun = GMsun / (c**2)
Tsun = GMsun / (c**3)

erg = sc.erg

DM_K = 2.41e-16  # for DM variation design matrix

# relative angle between the Earth's ecliptic and the galactic equator
e_ecl = 23.43704 * np.pi / 180.0

# unit vector pointing in direction of angle between Earth's ecliptic and the galactic equator
M_ecl = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(e_ecl), -np.sin(e_ecl)], [0.0, np.sin(e_ecl), np.cos(e_ecl)]])
