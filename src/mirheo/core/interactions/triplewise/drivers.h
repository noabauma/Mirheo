// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

// FIXME: Move type_traits to interactions/type_traits.h?
#include <mirheo/core/interactions/pairwise/kernels/type_traits.h>

#include <array>

#include <mirheo/core/datatypes.h>  //real3
#include <mirheo/core/celllist.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/pvs/views/pv.h>

#include <cassert>
#include <type_traits>

namespace mirheo
{

///Template parameter for wether we need to add force to dst(Self) or src(Other)
enum class InteractionWith
{
    Self, Other
};

/** \brief Compute triplewise interactions within a single ParticleVector.
    \tparam Interaction The triplewise interaction kernel

    \param [in] cinfo cell-list data
    \param [in,out] view The view that contains the particle data
    \param [in] interaction The triplewise interaction kernel

    Mapping is one thread per particle.
    TODO: Explain the algorithm.
  */
template<InteractionWith InteractWith, typename Handler>
__launch_bounds__(128, 16)
__global__ void computeTriplewiseSelfInteractions(
        CellListInfo cinfo, typename Handler::ViewType dstView, typename Handler::ViewType srcView, Handler handler)
{

    const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
    if (dstId >= dstView.size) return;

    const auto dstP = handler.read(dstView, dstId);

    real3 frc_ = make_real3(0.0_r); //only if dst gets force
    
    typename Handler::ParticleType srcP1, srcP2;

    const int3 cell0 = cinfo.getCellIdAlongAxes(handler.getPosition(dstP));
    
    const int cellZMin = math::max(cell0.y-1, 0);
    const int cellZMax = math::min(cell0.z+1, cinfo.ncells.z-1);
    const int cellYMin = math::max(cell0.y-1, 0);
    const int cellYMax = math::min(cell0.y+1, cinfo.ncells.y-1);
    const int cellXMin = math::max(cell0.x-1, 0);
    const int cellXMax = math::min(cell0.x+2, cinfo.ncells.x);

    for (int cellZ1 = cellZMin; cellZ1 <= cellZMax; ++cellZ1)
    {
        for (int cellY1 = cellYMin; cellY1 <= cellYMax; ++cellY1)
        {
            const int rowStart1 = cinfo.encode(cellXMin, cellY1, cellZ1);
            const int rowEnd1 = cinfo.encode(cellXMax, cellY1, cellZ1);

            const int pstart1 = cinfo.cellStarts[rowStart1];
            const int pend1   = cinfo.cellStarts[rowEnd1];

            for (int cellZ2 = cellZ1; cellZ2 <= cellZMax; ++cellZ2)
            {
                for (int cellY2 = (cellZ1 == cellZ2)? cellY1 : cellYMin; cellY2 <= cellYMax; ++cellY2)
                {
                    const int rowStart2 = cinfo.encode(cellXMin, cellY2, cellZ2);
                    const int rowEnd2 = cinfo.encode(cellXMax, cellY2, cellZ2);

                    const int pstart2 = cinfo.cellStarts[rowStart2];
                    const int pend2   = cinfo.cellStarts[rowEnd2];
                    
                    for (int srcId1 = pstart1; srcId1 < pend1; ++srcId1)
                    {
                        handler.readCoordinates(srcP1, srcView, srcId1);
                        bool interacting_01 = handler.withinCutoff(dstP, srcP1);
                        for (int srcId2 = (cellZ2 == cellZ1) && (cellY2 == cellY1)? srcId1 + 1 : pstart2; srcId2 < pend2; ++srcId2)
                        {
                            if((dstId == srcId1) || (dstId == srcId2)) continue;

                            handler.readCoordinates(srcP2, srcView, srcId2);

                            bool interacting_20 = handler.withinCutoff(dstP , srcP2);
                            bool interacting_12 = handler.withinCutoff(srcP1, srcP2);

                            if ((interacting_01 && interacting_12) || (interacting_12 && interacting_20) || (interacting_20 && interacting_01)) //atleast 2 vectors should be close
                            {
                                handler.readExtraData(srcP1, srcView, srcId1);    //SW3 & Dummy doesn't need this
                                handler.readExtraData(srcP2, srcView, srcId2);
                                
                                const std::array<real3, 3> val = handler(dstP, srcP1, srcP2, dstId, srcId1, srcId2);

                                if(InteractWith == InteractionWith::Self)
                                {
                                    frc_ += val[0];
                                }
                                else
                                {
                                    atomicAdd(srcView.forces + srcId1, val[1]);
                                    atomicAdd(srcView.forces + srcId2, val[2]);
                                }
                            }
                        }
                    }
                } //cellY2
            } //cellZ2
        } //cellY1
    } //cellZ1
    if(InteractWith == InteractionWith::Self)
        atomicAdd(dstView.forces + dstId, frc_); 
}

} // namespace mirheo
