#!/usr/bin/env python

"""Test triplewise forces."""

import mirheo as mir
import numpy as np
import sys
import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def brute_force(positions, domain, rc, epsilon):
    """O(N^3) brute force computation of the dummy force."""
    N = len(positions)

    # Preprocess a 0-1 matrix within_cutoff[i, j] denoting whether or not
    # particles i and j are within the cutoff.
    matrix = np.tile(positions, (N, 1, 1))
    diff = matrix - matrix.transpose((1, 0, 2))
    diff = abs(diff)
    diff = np.minimum(diff, domain - diff)  # Periodic boundary condition.
    within_cutoff = ((diff * diff).sum(axis=2) <= rc * rc).astype(np.int)

    result = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for k in range(j + 1, N):
                if i == k:
                    continue
                if within_cutoff[i, j] + within_cutoff[j, k] + within_cutoff[k, i] >= 2:
                    result[i, 0] += epsilon
    return result


def main():
    # Use a small domain to have a relatively large density with few particles.
    # domain = (10.0, 10.0, 10.0)
    domain = (4.0, 4.0, 4.0)
    rc = 1.0
    epsilon = 10.0

    particles = np.loadtxt("particles.csv", delimiter=',')
    particles = np.array([
    #     [0.01, 0.01, 0.01],
    #     [9.99, 0.01, 0.01],
    #     [0.01, 9.99, 0.01],
    #     [0.01, 0.01, 9.99],
    #     [0.02, 9.99, 0.01]
	[0.48, 0.48, 1.36],
	[2.28, 1.04, 1.32],
	[2.56, 1.8 , 1.  ],
	[2.12, 1.64, 0.44],
    ])
    # particles = np.random.rand(int(1.5 * domain[0] * domain[1] * domain[2]), 3) * domain
    # np.savetxt('particles.csv', particles, fmt='%f', delimiter=',')

    vel = np.zeros(particles.shape)
    u = mir.Mirheo((1, 1, 1), domain, debug_level=3, log_filename='log', no_splash=True)

    pv = mir.ParticleVectors.ParticleVector('pv', mass=1.0)
    ic = mir.InitialConditions.FromArray(pos=particles, vel=vel)
    u.registerParticleVector(pv, ic)

    dummy = mir.Interactions.Triplewise('interaction', rc=rc, kind='Dummy', epsilon=epsilon)
    u.registerInteraction(dummy)
    u.setInteraction(dummy, pv, pv, pv)

    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

    #Output should be (n-1)(n-2)/2 * epsilon (with n = #particles)
    dump_every = 1
    u.registerPlugins(mir.Plugins.createForceSaver('forces', pv))
    u.registerPlugins(mir.Plugins.createDumpParticles('force_dump', pv, dump_every, ["forces"], 'h5/dummy-'))

    u.run(2, dt=0)

    if rank == 0:
        f = h5py.File('h5/dummy-00001.h5', 'r')
        mirheo = f['forces'][()]
        brute = brute_force(particles, domain, rc, epsilon)
        # Note: Mirheo changes the order of particles, so we have to sort the
        #       result somehow before comparing.
        print("mirheo positions:\n", f['position'][()])
        print("mirheo:\n", mirheo)
        print("brute force:\n", brute)
        np.testing.assert_array_equal(np.sort(mirheo[:, 0]), np.sort(brute[:, 0]))
        print("OK")


main()

# TEST: triplewise.dummy
# cd triplewise
# mir.run --runargs "-n 2" ./dummy.py > dummy.out.txt
