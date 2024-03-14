import time
import glob
import argparse
import sys

import mpi4py

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import mpi4jax

import discovery as ds

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=int, default=67,     help='number of pulsars')
parser.add_argument('-n', type=int, default=10,     help='number of repetitions')
parser.add_argument('-s', type=int, default=42,     help='random number seed')
parser.add_argument('-m', type=str, required=True,  help='model (m2a, m3a)')
parser.add_argument('-v', type=int, default=0,      help='vmap size')
parser.add_argument('-x', action='store_true',      help='disable MPI')
parser.add_argument('-d', action='store_true',      help='add DMGP (100)')
parser.add_argument('-i', action='store_true',      help='double the number of pulsars')
args = parser.parse_args()

comm = mpi4py.MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class timer:
    def tic(self, s):
        self.s, self.t0 = s, time.time()

    def toc(self, n=1):
        if rank > 0:
            return

        dt = time.time() - self.t0
        if n == 1:
            print(f'{self.s}: {dt:.3f} seconds.')
        else:
            print(f'{self.s}: {dt:.3f} seconds, {dt/n:.3f} seconds/likelihood.')

t = timer()

nmax = args.p
psrfiles = sorted(glob.glob('../data/*-[JB]*.feather'))
allpsrs = [ds.Pulsar.read_feather(psrfile) for psrfile in psrfiles[:nmax]]

# double up the pulsars to get into IPTA territory
if args.i:
    allpsrs2 = [ds.Pulsar.read_feather(psrfile) for psrfile in psrfiles[:nmax]]
    for psr in allpsrs2:
        pos = np.array(psr.pos)
        pos[:] += 0.2
        psr.pos = list(pos / np.sqrt(np.dot(pos, pos)))
    allpsrs = allpsrs + allpsrs2

Tspan = ds.getspan(allpsrs)

t.tic(f"Making {args.m} likelihood on {rank}")
if args.m == 'm2a':
    model = ds.GlobalLikelihood((ds.PulsarLikelihood([psr.residuals,
                                                      ds.makenoise_measurement(psr, psr.noisedict),
                                                      ds.makegp_ecorr(psr, psr.noisedict),
                                                      ds.makegp_timing(psr),
                                                      ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='red_noise'),
                                                      ds.makegp_fourier(psr, ds.powerlaw, 100, T=Tspan, fourierbasis=ds.dmfourierbasis, name='dm') if args.d else None,
                                                      ds.makegp_fourier(psr, ds.powerlaw, 14, T=Tspan, name='gw',
                                                                        common=['gw_log10_A', 'gw_gamma'])])
                                 for psr in allpsrs[rank::size]))
else:
    model = ds.GlobalLikelihood((ds.PulsarLikelihood([psr.residuals,
                                                      ds.makenoise_measurement(psr, psr.noisedict),
                                                      ds.makegp_ecorr(psr, psr.noisedict),
                                                      ds.makegp_timing(psr),
                                                      ds.makegp_fourier(psr, ds.powerlaw, 30, T=Tspan, name='red_noise'),
                                                      ds.makegp_fourier(psr, ds.powerlaw, 100, T=Tspan, fourierbasis=ds.dmfourierbasis, name='dm') if args.d else None])
                                 for psr in allpsrs[rank::size]),
                                ds.makegp_fourier_global(allpsrs, ds.powerlaw, ds.hd_orf, 14, T=Tspan, name='gw'))

if args.x:
    plogl = jax.jit(model.logL)
else:
    plogl = jax.jit(model.plogL)
t.toc()

# seed numpy RNG so all CPUs will get the same random p0s
np.random.seed(args.s)

priordict = {".*_dm_log10_A": [-20, -11], ".*_dm_gamma": [0, 7]}

t.tic(f"Evaluating likelihood once")
p0 = ds.sample_uniform(plogl.params, priordict=priordict)
l = plogl(p0)
print(f'{l} on {rank}')
t.toc()

t.tic(f"Evaluating likelihood {args.n} times")
for i in range(args.n):
    p0 = ds.sample_uniform(plogl.params, priordict=priordict)
    l = plogl(p0)
t.toc(args.n)

if args.v == 0:
    sys.exit(0)

vlogl = jax.jit(jax.vmap(model.logL if args.x else model.plogL))

t.tic(f"Evaluating likelihood once ({args.v}-vector)")
p0 = ds.sample_uniform(plogl.params, n=args.v, priordict=priordict)
l = vlogl(p0)
t.toc()

t.tic(f"Evaluating likelihood {args.n} times ({args.v}-vector)")
for i in range(args.n):
    p0 = ds.sample_uniform(plogl.params, n=args.v, priordict=priordict)
    l = vlogl(p0)
t.toc(args.n * args.v)
