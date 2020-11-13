#!/usr/bin/env python

"""Test triplewise forces."""

import mirheo as mir
import numpy as np
import sys
import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def main():
    particles = np.loadtxt("particles.csv", delimiter=',')
    velo = np.zeros((particles.shape[0],3))

    domain = (10.0, 10.0, 10.0)
    u = mir.Mirheo((1, 1, 1), domain, debug_level=3, log_filename='log', no_splash=True)

    pv = mir.ParticleVectors.ParticleVector('pv', mass=1.0)
    ic = mir.InitialConditions.FromArray(pos=particles, vel=velo)
    u.registerParticleVector(pv, ic)

    dummy = mir.Interactions.Triplewise('interaction', rc=1.0, kind='Dummy', epsilon=10.0)
    u.registerInteraction(dummy)
    u.setInteraction(dummy, pv, pv, pv)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

    #Output should be (n-1)(n-2)/2 * epsilon (with n = #particles)
    dump_every = 1
    u.registerPlugins(mir.Plugins.createForceSaver('forces', pv))
    u.registerPlugins(mir.Plugins.createDumpParticles('force_dump', pv, dump_every, ["forces"], 'h5/dummy-'))

    u.run(2, dt=0.0001)

    f = h5py.File('h5/dummy-00001.h5', 'r')
    forces = f['forces']
    if rank == 0:
        print("forces:\n", forces[()])


main()

# nTEST: triplewise.dummy
# cd triplewise
# mir.run --runargs "-n 2" ./dummy.py > dummy.out.txt
