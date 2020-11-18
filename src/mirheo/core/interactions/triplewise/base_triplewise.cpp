// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "base_triplewise.h"

#include <mirheo/core/utils/config.h>
#include <mirheo/core/celllist.h>

namespace mirheo
{

BaseTriplewiseInteraction::BaseTriplewiseInteraction(const MirState *state, const std::string& name, real rc) :
    Interaction(state, name),
    rc_(rc)
{}

BaseTriplewiseInteraction::BaseTriplewiseInteraction(const MirState *state, __UNUSED Loader& loader, const ConfigObject& config) :
    BaseTriplewiseInteraction{state, config["name"], config["rc"]}
{}

BaseTriplewiseInteraction::~BaseTriplewiseInteraction() = default;

real BaseTriplewiseInteraction::getCutoffRadius() const
{
    // local-halo-halo particles have a reach of 2*rc, see base_triplewise.h
    return 2 * rc_;
}

BaseTriplewiseInteraction::CellListPair::CellListPair(
        ParticleVector *pv, real rc, const CellList *ref) :
    refinedLocal(pv, rc, ref->localDomainSize, ParticleVectorLocality::Local),
    halo(pv, rc, ref->localDomainSize + make_real3(4 * rc),  // 2*rc on each side
         ParticleVectorLocality::Halo)
{}

BaseTriplewiseInteraction::CellListPair *BaseTriplewiseInteraction::_getOrCreateCellLists(
        ParticleVector *pv, const CellList *refCL)
{
    const auto it = cellLists_.find(pv);
    if (it != cellLists_.end())
        return &it->second;
    const auto newIt = cellLists_.emplace(
            std::piecewise_construct,
            std::make_tuple(pv),                    // ParticleVector *
            std::make_tuple(pv, rc_, refCL)).first; // CellListPair
    return &newIt->second;
}

ConfigObject BaseTriplewiseInteraction::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = Interaction::_saveSnapshot(saver, typeName);
    config.emplace("rc", saver(rc_));
    return config;
}

} // namespace mirheo
