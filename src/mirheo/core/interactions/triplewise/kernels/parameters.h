// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/reflection.h>
#include <mirheo/core/utils/variant.h>

namespace mirheo
{

// forward declaration of triplewise kernels
class TriplewiseDummy;

class SW3;


/// parameters of the dummy interaction
struct DummyParams
{
    using KernelType = TriplewiseDummy; ///< the corresponding kernel
    real epsilon;   ///< force coefficient
};
MIRHEO_MEMBER_VARS(DummyParams, epsilon);

struct SW3Params
{
    using KernelType = SW3;

};
MIRHEO_MEMBER_VARS(SW3Params, )

/// variant of all possible triplewise interactions
using VarTriplewiseParams = mpark::variant<DummyParams>;

} // namespace mirheo
