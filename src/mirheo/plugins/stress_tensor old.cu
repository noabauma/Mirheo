// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "stress_tensor.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

//#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/path.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/mpi_types.h>

namespace mirheo
{


namespace stress_tensor_kernels
{
__global__ void totalStress(PVview view, const Stress *stress, const real mass, stress_tensor_plugin::ReductionType *StressTensor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    stress_tensor_plugin::ReductionType P;

    const int N = view.size;

    if (tid < N){
        const real4 v  = view.velocities[tid];
        const Stress s = stress[tid];

        P.xx = mass*v.x*v.x + s.xx;
        P.xy = mass*v.x*v.y + s.xy;
        P.xz = mass*v.x*v.z + s.xz;
        P.yy = mass*v.y*v.y + s.yy;
        P.yz = mass*v.y*v.z + s.yz;
        P.zz = mass*v.z*v.z + s.zz;
    }

    auto sum = [](real a, real b) { return a+b; };

    P.xx = warpReduce(P.xx, sum);
    P.xy = warpReduce(P.xy, sum);
    P.xz = warpReduce(P.xz, sum);
    P.yy = warpReduce(P.yy, sum);
    P.yz = warpReduce(P.yz, sum);
    P.zz = warpReduce(P.zz, sum);

    if (laneId() == 0){
        atomicAdd(&StressTensor->xx, P.xx);
        atomicAdd(&StressTensor->xy, P.xy);
        atomicAdd(&StressTensor->xz, P.xz);
        atomicAdd(&StressTensor->yy, P.yy);
        atomicAdd(&StressTensor->yz, P.yz);
        atomicAdd(&StressTensor->zz, P.zz);
    }
}
} // namespace stress_tensor_kernels

StressTensorPlugin::StressTensorPlugin(const MirState *state, std::string name, std::string pvName, int dumpEvery) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    dumpEvery_(dumpEvery)
{}

StressTensorPlugin::~StressTensorPlugin() = default;

void StressTensorPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);

    info("Plugin %s initialized for the following particle vector: %s", getCName(), pvName_.c_str());
}

void StressTensorPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer_, pvName_);
    _send(sendBuffer_);
}

void StressTensorPlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    PVview view(pv_, pv_->local());
    const Stress *stress = pv_->local()->dataPerParticle.getData<Stress>(channel_names::stresses)->devPtr();

    localStressTensor_.clear(stream);

    constexpr int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);

    SAFE_KERNEL_LAUNCH(
        stress_tensor_kernels::totalStress,
        nblocks, nthreads, 0, stream,
        view, stress, pv_->getMassPerParticle(), localStressTensor_.devPtr() );

        localStressTensor_.downloadFromDevice(stream, ContainersSynch::Synch);

    savedTime_ = getState()->currentTime;
    needToSend_ = true;
}

void StressTensorPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!needToSend_) return;

    debug2("Plugin %s is sending now data", getCName());

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, savedTime_, localStressTensor_[0]);
    _send(sendBuffer_);

    needToSend_ = false;
}

//=================================================================================

StressTensorDumper::StressTensorDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path_(makePath(path)),
    fname_(name)
{}

void StressTensorDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated_ = createFoldersCollective(comm, path_);
}

void StressTensorDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string pvName;
    SimpleSerializer::deserialize(data_, pvName);

    if (activated_ && fdump_.get() == nullptr)
    {
        //auto fname = joinPaths(path_, setExtensionOrDie(pvName, "csv"));
        auto fname = joinPaths(path_, setExtensionOrDie(fname_, "csv"));
        auto status = fdump_.open(fname, "w");
        if (status != FileWrapper::Status::Success)
            die("Could not open file '%s'", fname.c_str());
        fprintf(fdump_.get(), "time,Pxx,Pxy,Pxz,Pyy,Pyz,Pzz\n");
    }
}

void StressTensorDumper::deserialize()
{
    MirState::TimeType curTime;
    stress_tensor_plugin::ReductionType localStress, totalStress;

    SimpleSerializer::deserialize(data_, curTime, localStress);

    if (!activated_) return;

    const auto dataType = getMPIFloatType<real>();

    static_assert(sizeof(totalStress) == 6 * sizeof(real), "unexpected sizeof(Stress)");

    MPI_Check( MPI_Reduce(&localStress, &totalStress, 6, dataType, MPI_SUM, 0, comm_) );
    
    /*
    MPI_Check( MPI_Reduce(&localStress.xx, &totalStress.xx, 1, dataType, MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce(&localStress.xy, &totalStress.xy, 1, dataType, MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce(&localStress.xz, &totalStress.xz, 1, dataType, MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce(&localStress.yy, &totalStress.yy, 1, dataType, MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce(&localStress.yz, &totalStress.yz, 1, dataType, MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce(&localStress.zz, &totalStress.zz, 1, dataType, MPI_SUM, 0, comm_) );
    */

    fprintf(fdump_.get(), "%g,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n", curTime, totalStress.xx, totalStress.xy, totalStress.xz, totalStress.yy, totalStress.yz, totalStress.zz);
}

} // namespace mirheo
