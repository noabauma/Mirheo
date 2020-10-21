// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

// FIXME: Move type_traits to interactions/type_traits.h?
#include <mirheo/core/interactions/pairwise/kernels/type_traits.h>

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
    (void)cinfo;
    (void)view;
    (void)handler;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Hello from computeTriplewiseSelfInteraction view.size=%d\n",
               view.size);
    }

    // TODO
}

} // namespace mirheo
