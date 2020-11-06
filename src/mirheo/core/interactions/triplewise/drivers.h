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

    //auto accumulator = interaction.getZeroedAccumulator();    //SW3 doesn't have accumulator

    const int3 cell0 = cinfo.getCellIdAlongAxes(handler.getPosition(dstP));

    for (int cellZ = cell0.z-1; cellZ <= cell0.z+1; cellZ++)
    {
        for (int cellY = cell0.y-1; cellY <= cell0.y+1; cellY++)
        {
            if ( !(cellY >= 0 && cellY < cinfo.ncells.y && cellZ >= 0 && cellZ < cinfo.ncells.z) ) continue;

            const int midCellId = cinfo.encode(cell0.x, cellY, cellZ);
            const int rowStart  = math::max(midCellId-1, 0);
            const int rowEnd    = math::min(midCellId+2, cinfo.totcells);

            const int pstart = cinfo.cellStarts[rowStart];
            const int pend   = cinfo.cellStarts[rowEnd];

            typename Handler::ParticleType srcP1, srcP2;
            for (int srcId1 = pstart; srcId1 < pend; srcId1++)
            {
                handler.readCoordinates(srcP1, view, srcId1);
                bool interacting_01 = handler.withinCutoff(dstP, srcP1);
                for (int srcId2 = srcId1 + 1; srcId2 < pend; srcId2++)
                {
                    
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
    }
    atomicAdd(view.forces + dstId, frc_);
}

} // namespace mirheo
