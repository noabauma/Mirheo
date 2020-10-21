#!/usr/bin/env python

"""Test triplewise forces."""

import mirheo as mir
import sys

def main():
    domain = (12.0, 10.0, 8.0)
    u = mir.Mirheo((1, 1, 1), domain, debug_level=3, log_filename='log', no_splash=True)

    pv = mir.ParticleVectors.ParticleVector('pv', mass=1.0)
    ic = mir.InitialConditions.Uniform(number_density=10.0)
    u.registerParticleVector(pv, ic)

    lj = mir.Interactions.Triplewise('interaction', rc=1.0, kind='Dummy', epsilon=10.0)
    u.registerInteraction(lj)
    u.setInteraction(lj, pv, pv, pv)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

    # There is a temporary printf() in the CUDA kernel that prints the number
    # of particles. This is the output of this test.
    u.run(5, dt=0.0001)


main()

# nTEST: triplewise.dummy
# cd triplewise
# mir.run --runargs "-n 2" ./dummy.py > dummy.out.txt
