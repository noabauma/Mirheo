// Code: Noah Baumann Bachelor Thesis
// 2Body SW potential
// SW: Stillinger-Weber potiential
// this code is inspired by "lj.h"
#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

#include <mirheo/core/utils/cuda_common.h>      //math::exp()

namespace mirheo
{

/// Compute Stillinger-Weber forces on the device.
class PairwiseSW : public PairwiseKernel, public ParticleFetcher
{
public:
    using ViewType     = PVview;     ///< Compatible view type
    using ParticleType = Particle;   ///< Compatible particle type
    using HandlerType  = PairwiseSW; ///< Corresponding handler
    using ParamsType   = SWParams;   ///< Corresponding parameters type

    /// Constructor
    PairwiseSW(real rc, real epsilon, real sigma, real A, real B) :
        ParticleFetcher(rc),
        epsilon_(epsilon),
        sigma_(sigma),
        A_(A),
        B_(B)
    {}

    /// Generic constructor
    PairwiseSW(real rc, const ParamsType& p, __UNUSED real dt, __UNUSED long seed=42424242) :
        PairwiseSW{rc, p.epsilon, p.sigma, p.A, p.B}
    {}

    /// Evaluate the force
    __D__ inline real3 operator()(ParticleType dst, int /*dstId*/,
                                  ParticleType src, int /*srcId*/) const
    {
        const real3 dr = dst.r - src.r;
        const real dr2 = dot(dr, dr);
        if (dr2 > rc2_)
            return make_real3(0.0_r);

        const real rs2 = (sigma_*sigma_) / dr2;
        const real rs4 = rs2 * rs2;
        const real phi = A_*epsilon*(B_*rs4 - 1.0_r)*math::exp(sigma_ / (math::sqrt(dr2) - rc_));

        return phi * dr/dr2;
    }

    /// initialize accumulator
    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }

    /// \return type name string
    static std::string getTypeName()
    {
        return "PairwiseSW";
    }

private:
    real epsilon_;
    real sigma_;
    real A_;        //given from the paper 7.049556277
    real B_;        //0.6022245584
};

} // namespace mirheo
