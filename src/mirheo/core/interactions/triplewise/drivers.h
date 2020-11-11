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

/** \brief Compute triplewise interactions within a single ParticleVector.
    \tparam Interaction The triplewise interaction kernel

    \param [in] cinfo cell-list data
    \param [in,out] view The view that contains the particle data
    \param [in] interaction The triplewise interaction kernel

    Mapping is one thread per particle.
    TODO: Explain the algorithm.
  */
template<typename Handler>
__launch_bounds__(128, 16)
__global__ void computeTriplewiseSelfInteractions(
        CellListInfo cinfo, typename Handler::ViewType view, Handler handler)
{

    const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
    if (dstId >= view.size) return;

    const auto dstP = handler.read(view, dstId);

    real3 frc_ = make_real3(0.0_r);
    
    typename Handler::ParticleType srcP1, srcP2;

    //auto accumulator = interaction.getZeroedAccumulator();    //SW3 doesn't have an accumulator

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
                for (int cellY2 = cellY1; cellY2 <= cellYMax; ++cellY2)
                {
                    if((cellZ1 == cellZ2) && (cellY1 == cellY2))    //if they are in the same cell, do upper-matrix loop
                    {
                        //const int rowStart2 = cinfo.encode(cellXMin, cellY2, cellZ2);
                        //const int rowEnd2 = cinfo.encode(cellXMax, cellY2, cellZ2);

                        //const int pstart2 = cinfo.cellStarts[rowStart2];
                        //const int pend2   = cinfo.cellStarts[rowEnd2];
                        
                        for (int srcId1 = pstart1; srcId1 < pend1; ++srcId1)
                        {
                            handler.readCoordinates(srcP1, view, srcId1);
                            bool interacting_01 = handler.withinCutoff(dstP, srcP1);
                            for (int srcId2 = srcId1 + 1; srcId2 < pend1; ++srcId2)
                            {
                                if((dstId == srcId1) || (dstId == srcId2)) continue;
                                if(dstId == 0){
                                    printf("(%i,%i,%i) ", dstId, srcId1, srcId2);
                                }
                                handler.readCoordinates(srcP2, view, srcId2);

                                bool interacting_20 = handler.withinCutoff(dstP , srcP2);
                                bool interacting_12 = handler.withinCutoff(srcP1, srcP2);

                                if ((interacting_01 && interacting_12) || (interacting_12 && interacting_20) || (interacting_20 && interacting_01)) //atleast 2 vectors should be close
                                {
                                    //handler.readExtraData(srcP, srcView, srcId);    //SW3 doesn't need this

                                    const std::array<real3, 3> val = handler(dstP, srcP1, srcP2, dstId, srcId1, srcId2);

                                    frc_ += val[0];
                                    /*
                                    if (NeedDstOutput == InteractionOutMode::NeedOutput)
                                        accumulator.add(val[0]);

                                    if (NeedSrcOutput == InteractionOutMode::NeedOutput)
                                        accumulator.atomicAddToSrc(val, srcView, srcId);
                                    */
                                }
                            }
                        }
                    }
                    else    //O(N^2) go throw all 
                    {
                        const int rowStart2 = cinfo.encode(cellXMin, cellY2, cellZ2);
                        const int rowEnd2 = cinfo.encode(cellXMax, cellY2, cellZ2);

                        const int pstart2 = cinfo.cellStarts[rowStart2];
                        const int pend2   = cinfo.cellStarts[rowEnd2];
                        
                        for (int srcId1 = pstart1; srcId1 < pend1; ++srcId1)
                        {
                            handler.readCoordinates(srcP1, view, srcId1);
                            bool interacting_01 = handler.withinCutoff(dstP, srcP1);
                            for (int srcId2 = pstart2; srcId2 < pend2; ++srcId2)
                            {
                                if((dstId == srcId1) || (dstId == srcId2)) continue;

                                handler.readCoordinates(srcP2, view, srcId2);

                                bool interacting_20 = handler.withinCutoff(dstP , srcP2);
                                bool interacting_12 = handler.withinCutoff(srcP1, srcP2);

                                if ((interacting_01 && interacting_12) || (interacting_12 && interacting_20) || (interacting_20 && interacting_01)) //atleast 2 vectors should be close
                                {
                                    //handler.readExtraData(srcP, srcView, srcId);    //SW3 doesn't need this

                                    const std::array<real3, 3> val = handler(dstP, srcP1, srcP2, dstId, srcId1, srcId2);

                                    frc_ += val[0];
                                    /*
                                    if (NeedDstOutput == InteractionOutMode::NeedOutput)
                                        accumulator.add(val[0]);

                                    if (NeedSrcOutput == InteractionOutMode::NeedOutput)
                                        accumulator.atomicAddToSrc(val, srcView, srcId);
                                    */
                                }
                            }
                        }
                    } //else
                } //cellY2
            } //cellZ2
        } //cellY1
    } //cellZ1
    atomicAdd(view.forces + dstId, frc_);
}

} // namespace mirheo
