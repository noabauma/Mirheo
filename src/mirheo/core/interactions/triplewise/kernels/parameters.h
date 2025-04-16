// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/reflection.h>
#include <mirheo/core/utils/variant.h>

namespace mirheo
{

// forward declaration of triplewise kernels
class SW3;
class TriplewiseDummy;


struct SW3Params
{
    using KernelType = SW3;
    real lambda;
    real epsilon;
    real theta;
    real gamma;
    real sigma;

};
MIRHEO_MEMBER_VARS(SW3Params, lambda, epsilon, theta, gamma, sigma);

/// parameters of the dummy interaction
struct DummyParams
{
    using KernelType = TriplewiseDummy; ///< the corresponding kernel
    real epsilon;   ///< force coefficient
};
MIRHEO_MEMBER_VARS(DummyParams, epsilon);


/// variant of all possible triplewise interactions
using VarTriplewiseParams = mpark::variant<SW3Params, DummyParams>;

} // namespace mirheo
