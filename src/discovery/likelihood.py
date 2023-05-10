import functools

import numpy as np

from . import matrix

# Kernel
#   ConstantKernel
#       define solve_1d, perhaps solve_2d (operate on numpy)
#   VariableKernel
#       define make_solve_1d, perhaps make_solve_2d (return functions that operate on jax)
#
#   all define make_kernelproduct (make_kernelterms)
#
# GP
#   ConstantGP
#       consists of a ConstantKernel and a numpy matrix
#   VariableGP
#       consists of a VariableKernel and a numpy matrix
#
# ShermanMorrisonKernel can return a ConstantKernel or a VariableKernel

# npta = je.PulsarLikelihood(je.residuals(psr),
#                            je.makenoise_measurement(psr, noisedict),
#                            je.makegp_ecorr(psr, noisedict),
#                            je.makegp_timing(psr),
#                            je.makegp_fourier('red_noise', psr, je.powerlaw, 10),
#                            je.makegp_fourier('red_noise', psr,
#                                              je.makepowerlaw_crn(5), 10, T=Tspan,
#                                              common=['crn_log10_A', 'crn_gamma']),
#                            concat=True)

class PulsarLikelihood:
    def __init__(self, args, concat=True):
        y     = [arg for arg in args if isinstance(arg, np.ndarray)]
        noise = [arg for arg in args if isinstance(arg, matrix.Kernel)]
        cgps  = [arg for arg in args if isinstance(arg, matrix.ConstantGP)]
        vgps  = [arg for arg in args if isinstance(arg, matrix.VariableGP)]

        if len(y) > 1 or len(noise) > 1:
            raise ValueError("Only one residual vector and one noise Kernel allowed.")
        elif len(noise) == 0:
            raise ValueError("I need exactly one noise Kernel.")

        noise, y0 = noise[0], y[0]

        if cgps:
            if concat:
                cgp = matrix.CompoundGP(cgps)
                csm = matrix.ShermanMorrisonKernel(noise, cgp.F, cgp.Phi)
            else:
                csm = noise
                for cgp in cgps:
                    csm = matrix.ShermanMorrisonKernel(csm, cgp.F, cgp.Phi)
        else:
            csm = noise

        if vgps:
            if concat:
                vgp = matrix.CompoundGP(vgps)
                vsm = matrix.ShermanMorrisonKernel(csm, vgp.F, vgp.Phi)
            else:
                vsm = csm
                for vgp in vgps:
                    vsm = matrix.ShermanMorrisonKernel(vsm, vgp.F, vgp.Phi)
        else:
            vsm = csm

        self.y, self.N = y0, vsm

    @functools.cached_property
    def logL(self):
        return self.N.make_kernelproduct(self.y)
