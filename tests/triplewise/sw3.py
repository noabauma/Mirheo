#!/usr/bin/env python

"""Test triplewise forces."""

import mirheo as mir
import numpy as np
import sys
import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

np.random.seed(int(sys.argv[1]))

def main():
    if len(sys.argv) < 3:
        print("needs seed(int) & #particles(int) \n")
        sys.exit(0)
    n = int(sys.argv[2])

    particles = np.random.rand(n, 3) + np.full((n, 3), 5.0)
    if rank == 0: 
        print("particles:\n", particles)
    
    velo = np.zeros((n,3))

    domain = (10.0, 10.0, 10.0)
    u = mir.Mirheo((1, 1, 1), domain, debug_level=3, log_filename='log', no_splash=True)

    pv = mir.ParticleVectors.ParticleVector('pv', mass=1.0)
    ic = mir.InitialConditions.FromArray(pos=particles, vel=velo)
    u.registerParticleVector(pv, ic)

    sw3 = mir.Interactions.Triplewise('interaction', rc=2.0, kind='SW3', lambda_=23.15, epsilon=6.189, theta=1.910633236, gamma=1.2, sigma=2.3925)
    u.registerInteraction(sw3)
    u.setInteraction(sw3, pv, pv, pv)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)


    dump_every = 1
    u.registerPlugins(mir.Plugins.createForceSaver('forces', pv))
    u.registerPlugins(mir.Plugins.createDumpParticles('force_dump', pv, dump_every, ["forces"], 'h5/sw3-'))

    u.run(2, dt=0.0001)

    f = h5py.File('h5/sw3-00001.h5', 'r')
    forces = f['forces']
    if rank == 0:
        print("forces:\n", forces[()])


main()

# nTEST: triplewise.sw3
# cd triplewise
# mir.run --runargs "-n 2" ./sw3.py > sw3.out.txt
