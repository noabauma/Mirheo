#!/usr/bin/env python

"""Test triplewise forces."""

import mirheo as mir
import numpy as np
import sys

def main():
    particles = np.loadtxt("particles.csv", delimiter=',')
    velo = np.zeros((particles.shape[0],3))

    domain = (10.0, 10.0, 10.0)
    u = mir.Mirheo((1, 1, 1), domain, debug_level=3, log_filename='log', no_splash=True)

    pv = mir.ParticleVectors.ParticleVector('pv', mass=1.0)
    #ic = mir.InitialConditions.Uniform(number_density=10.0)
    ic = mir.InitialConditions.FromArray(pos=particles, vel=velo)
    u.registerParticleVector(pv, ic)

    dummy = mir.Interactions.Triplewise('interaction', rc=1.0, kind='Dummy', epsilon=10.0)
    u.registerInteraction(dummy)
    u.setInteraction(dummy, pv, pv, pv)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

    #Output should be n(n-1)(n-2)/6 * epsilon  per step (with n = #particles)
    dump_every = 1
    u.registerPlugins(mir.Plugins.createForceSaver('force', pv))
    u.registerPlugins(mir.Plugins.createDumpParticles('force_dump', pv, dump_every, ["forces"], 'h5/pv-'))

    u.run(1, dt=0.0001)


main()

# nTEST: triplewise.dummy
# cd triplewise
# mir.run --runargs "-n 2" ./dummy.py > dummy.out.txt
