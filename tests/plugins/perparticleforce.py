#!/usr/bin/env python

import mirheo as mir
import numpy as np
import h5py
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()



def main():
    #this is for generating new particles for hdf5 file
    """
    if rank == 0:
        #np.random.seed(42)
        n = 1000
        position = np.random.rand(n,3)
        extraforces = np.random.rand(n,3)
        id = np.reshape(np.arange(n), (n,1))
        velocity = np.zeros((n,3))

        os.remove("h5_c/pv.PV-00000.h5")
        f = h5py.File("h5_c/pv.PV-00000.h5", "a")
        dset = f.create_dataset("position", (n,3), dtype=np.float64)
        dset[...] = position
        dset = f.create_dataset("velocity", (n,3), dtype=np.float64)
        dset[...] = velocity
        dset = f.create_dataset("id", (n,1), dtype=np.int)
        dset[...] = id
        dset = f.create_dataset("extraforces", (n,3), dtype=np.float64)
        dset[...] = extraforces
        #print(f.keys())
        f.close()
    """
    

    ranks = (1, 1, 1)
    domain = (10.0, 10.0, 10.0)

    u = mir.Mirheo(ranks, domain, debug_level=3, log_filename='log', no_splash=True, checkpoint_every=10)

    pv = mir.ParticleVectors.ParticleVector('pv', mass=1.0)
    ic = mir.InitialConditions.Restart('h5_c')
    u.registerParticleVector(pv, ic)

    u.registerPlugins(mir.Plugins.createAddPerParticleForce("extra_force", pv, "extraforces"))    #my plugin

    sw2 = mir.Interactions.Pairwise('sw', rc=1.0, kind="SW", epsilon=0.0, sigma=0.0, A=0.0, B=0.0)  #has no effect (here because else bug)
    u.registerInteraction(sw2)
    u.setInteraction(sw2, pv, pv)
    
    vv = mir.Integrators.VelocityVerlet('vv')
    u.registerIntegrator(vv)
    u.setIntegrator(vv, pv)

    dump_every = 1
    u.registerPlugins(mir.Plugins.createForceSaver('forces', pv))
    u.registerPlugins(mir.Plugins.createDumpParticles('force_dump', pv, dump_every, ["forces", "extraforces"], 'h5_c/sw3-'))

    u.run(5, dt=0.0)

    if rank == 0:
        f_ = h5py.File('h5_c/sw3-00004.h5', 'r')
        forces = f_['forces'][()]
        extraforces = f_['extraforces'][()]
        try: 
            np.testing.assert_allclose(np.sort(extraforces, axis=0), np.sort(forces, axis=0), rtol=1e-15)
        except:
            print("positions:\n", f_['position'][()])
            print("forces:\n", np.sort(np.sort(forces), axis=0))
            print("extraforces:\n", np.sort(np.sort(extraforces), axis=0))
            print("differences:\n", np.sum(np.sort(np.sort(forces), axis=0) - np.sort(np.sort(extraforces), axis=0)))
            raise
        
        print("OK")
        
        

main()

# TEST: plugins.perparticleforce
# cd plugins
# mir.run --runargs "-n 2" ./perparticleforce.py > perparticleforce.out.txt