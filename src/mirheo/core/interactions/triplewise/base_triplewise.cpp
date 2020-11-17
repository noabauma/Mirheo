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
    return 2.0_r*rc_;   //due to the property of 3body interaction, increasing cell sizes
}

CellList* BaseTriplewiseInteraction::_getOrCreateHaloCellList(
        ParticleVector *pv, const CellList *refCL) {
    auto it = haloCLs_.find(pv);
    if (it != haloCLs_.end())
        return &it->second;
    // std::map<ParticleVector *, CellList>
    auto pair = haloCLs_.emplace(
            std::piecewise_construct,
            std::make_tuple(pv),
            std::make_tuple(pv, getCutoffRadius(), refCL->localDomainSize,
                            ParticleVectorLocality::Halo));
    return &pair.first->second;
}

ConfigObject BaseTriplewiseInteraction::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = Interaction::_saveSnapshot(saver, typeName);
    config.emplace("rc", saver(rc_));
    return config;
}

} // namespace mirheo
