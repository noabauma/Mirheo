// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "base_triplewise.h"

#include <mirheo/core/utils/config.h>

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
    return rc_;
}

ConfigObject BaseTriplewiseInteraction::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = Interaction::_saveSnapshot(saver, typeName);
    config.emplace("rc", saver(rc_));
    return config;
}

} // namespace mirheo
