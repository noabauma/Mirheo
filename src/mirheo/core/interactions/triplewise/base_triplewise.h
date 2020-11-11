// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/interactions/interface.h>
#include <mirheo/core/interactions/utils/parameters_wrap.h>

namespace mirheo
{

/** \brief Base class for short-range triplewise interactions
 */
class BaseTriplewiseInteraction : public Interaction
{
public:
    /** \brief Construct a base triplewise interaction from parameters.
        \param [in] state The global state of the system.
        \param [in] name The name of the interaction.
        \param [in] rc The cutoff radius of the interaction.
                       Must be positive and smaller than the sub-domain size.
    */
    BaseTriplewiseInteraction(const MirState *state, const std::string& name, real rc);

    /** \brief Construct the interaction from a snapshot.
        \param [in] state The global state of the system.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the interaction.
     */
    BaseTriplewiseInteraction(const MirState *state, Loader& loader, const ConfigObject& config);
    ~BaseTriplewiseInteraction();

    /// \return the cut-off radius of the triplewise interaction.
    real getCutoffRadius() const override;

protected:
    /** \brief Snapshot saving for base triplewise interactions. Stores the cutoff value.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
    */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

protected:
    real rc_; ///< cut-off radius of the interaction
};

} // namespace mirheo