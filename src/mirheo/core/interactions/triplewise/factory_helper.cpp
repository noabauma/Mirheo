// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "factory_helper.h"

namespace mirheo
{

namespace factory_helper
{

DummyParams readDummyParams(ParametersWrap& desc)
{
    DummyParams p;
    p.epsilon = desc.read<real>("epsilon");
    return p;
}

} // namespace factory_helper

} // namespace mirheo
