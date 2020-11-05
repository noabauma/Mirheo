// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "factory.h"

#include "triplewise.h"

#include "kernels/sw.h"
#include "kernels/dummy.h"

#include <mirheo/core/utils/variant_foreach.h>

namespace mirheo
{

std::shared_ptr<BaseTriplewiseInteraction>
createInteractionTriplewise(const MirState *state, const std::string& name, real rc, const VarTriplewiseParams& varParams)
{
    // NOTE: This is a simplified version of the force. We assume thar stresses are not needed.
    return mpark::visit([&](const auto& params) -> std::shared_ptr<BaseTriplewiseInteraction>
    {
        using Kernel = typename std::remove_reference_t<decltype(params)>::KernelType;
        return std::make_shared<TriplewiseInteraction<Kernel>>(state, name, rc, params);
    }, varParams);
}


namespace {
/// Helper class for implementing variant foreach, see pairwise factory.
struct TriplewiseFactoryVisitor {
    const MirState *state;
    Loader& loader;
    const ConfigObject& config;
    const std::string& typeName;
    std::shared_ptr<BaseTriplewiseInteraction> impl;
};
} // anonymous namespace

/// Creates the given triplewise interaction if the type name matches.
template <class KernelType>
static void tryLoadTriplewise(TriplewiseFactoryVisitor &visitor)
{
    using T = TriplewiseInteraction<KernelType>;
    if (T::getTypeName() == visitor.typeName) {
        visitor.impl = std::make_shared<TriplewiseInteraction<KernelType>>(
                visitor.state, visitor.loader, visitor.config);
    }
}

std::shared_ptr<BaseTriplewiseInteraction>
loadInteractionTriplewise(const MirState *state, Loader& loader, const ConfigObject& config)
{
    static_assert(std::is_same<
            VarTriplewiseParams,
            mpark::variant<SW3Params, DummyParams>>::value,
            "Load interactions must be updated if VarTriplewiseParams is changed.");

    const std::string& typeName = config["__type"].getString();
    TriplewiseFactoryVisitor visitor{state, loader, config, typeName, nullptr};
    
    // SW 3Body
    tryLoadTriplewise<SW3Params::KernelType>(visitor);
    // Dummy.
    tryLoadTriplewise<DummyParams::KernelType>(visitor);

    if (!visitor.impl)
        die("Unrecognized impl type \"%s\".", typeName.c_str());

    return std::move(visitor.impl);
}

} // namespace mirheo
