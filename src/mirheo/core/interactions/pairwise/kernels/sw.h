// Code: Noah Baumann Bachelor Thesis
// 2Body SW potential
// SW: Stillinger-Weber potiential
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
    using HandlerType  = PairwiseLJ; ///< Corresponding handler
    using ParamsType   = LJParams;   ///< Corresponding parameters type

    /// Constructor
    PairwiseSW(real rc, real epsilon, real sigma) :
        ParticleFetcher(rc),
        epsilon_(epsilon),
        sigma_(sigma)
    {}

    /// Generic constructor
    PairwiseSW(real rc, const ParamsType& p, __UNUSED real dt, __UNUSED long seed=42424242) :
        PairwiseSW{rc, p.epsilon, p.sigma}
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
        const real phi = A*epsilon*(B*rs4 - 1.0_r)*math::exp(sigma_ / (dr - a*sigma_));

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
    const real A = 7.049556277;     //given from the paper
    const real B = 0.6022245584;
    const real a = 1.8;             //reduced cutoff
};

} // namespace mirheo
