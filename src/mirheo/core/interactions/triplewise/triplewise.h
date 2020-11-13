// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "base_triplewise.h"
#include "drivers.h"
#include "factory_helper.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/snapshot.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <fstream>
#include <map>

namespace mirheo
{

template <typename T> class Debug;
/** \brief Short-range symmetric triplewise interactions
    \tparam TriplewiseKernel The functor that describes the interaction between two particles (interaction kernel).

    See the triplewise interaction entry of the developer documentation for the interface requirements of the kernel.
 */
template <class TriplewiseKernel>
class TriplewiseInteraction : public BaseTriplewiseInteraction
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // bug in breathe
    /// The parameters corresponding to the interaction kernel.
    using KernelParams = typename TriplewiseKernel::ParamsType;
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /** \brief Construct a TriplewiseInteraction object
        \param [in] state The global state of the system
        \param [in] name The name of the interaction
        \param [in] rc The cut-off radius of the interaction
        \param [in] params The parameters used to construct the interaction kernel
        \param [in] seed used to initialize random number generator (needed to construct some interaction kernels).
     */
    TriplewiseInteraction(const MirState *state, const std::string& name, real rc,
                        KernelParams params, long seed = 42424242) :
        BaseTriplewiseInteraction(state, name, rc),
        kernel_{rc, params, seed},
        params_{params}
    {}

    /** \brief Constructs a TriplewiseInteraction object from a snapshot.
        \param [in] state The global state of the system
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The parameters of the interaction.
     */
    TriplewiseInteraction(const MirState *state, Loader& loader, const ConfigObject& config) :
        TriplewiseInteraction(state, config["name"], config["rc"],
                              loader.load<KernelParams>(config["params"]), 42424242)
    {
        long seed = 42424242;
        warn("NOTE: Seed not serialized, resetting it to %ld!", seed);
    }

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, ParticleVector *pv3,
                          CellList *cl1, CellList *cl2, CellList *cl3) override
    {
        if (outputsDensity <TriplewiseKernel>::value || requiresDensity<TriplewiseKernel>::value)
        {
            pv1->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);
            pv2->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);
            pv3->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);

            cl1->requireExtraDataPerParticle<real>(channel_names::densities);
            cl2->requireExtraDataPerParticle<real>(channel_names::densities);
            cl3->requireExtraDataPerParticle<real>(channel_names::densities);
        }
    }

    void local(ParticleVector *pv1, ParticleVector *pv2, ParticleVector *pv3,
               CellList *cl1, CellList *cl2, CellList *cl3, cudaStream_t stream) override
    {
        _computeLocal(pv1, pv2, pv3, cl1, cl2, cl3, stream);
    }

    void halo(ParticleVector *pv1, ParticleVector *pv2, ParticleVector *pv3,
              CellList *cl1, CellList *cl2, CellList *cl3, cudaStream_t stream) override
    {
        const bool isov1 = dynamic_cast<ObjectVector *>(pv1) != nullptr;
        const bool isov2 = dynamic_cast<ObjectVector *>(pv2) != nullptr;
        const bool isov3 = dynamic_cast<ObjectVector *>(pv2) != nullptr;

        if (isov1 || isov2 || isov3)
            die("3-body force with ObjectVectors not implemented.");
        if (pv1 != pv2 || pv2 != pv3 || pv3 != pv1)
            die("3-body forces with two or three different ParticleVectors not implemented.");

        (void)pv1;
        (void)pv2;
        (void)pv3;
        (void)cl1;
        (void)cl2;
        (void)cl3;
        (void)stream;
        
        //_computeHalo(pv1, pv2, pv3, cl1, cl2, cl3, stream);
    }

    Stage getStage() const override
    {
        if (isFinal<TriplewiseKernel>::value)
            return Stage::Final;
        else
            return Stage::Intermediate;
    }

    std::vector<InteractionChannel> getInputChannels() const override
    {
        std::vector<InteractionChannel> channels;

        if (requiresDensity<TriplewiseKernel>::value)
            channels.push_back({channel_names::densities, Interaction::alwaysActive});

        return channels;
    }

    std::vector<InteractionChannel> getOutputChannels() const override
    {
        std::vector<InteractionChannel> channels;

        if (outputsDensity<TriplewiseKernel>::value)
            channels.push_back({channel_names::densities, Interaction::alwaysActive});

        if (outputsForce<TriplewiseKernel>::value)
            channels.push_back({channel_names::forces, Interaction::alwaysActive});

        return channels;
    }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override
    {
        auto fname = createCheckpointNameWithId(path, "TriplewiseInt", "txt", checkpointId);
        {
            std::ofstream fout(fname);
            kernel_.writeState(fout);
        }
        createCheckpointSymlink(comm, path, "TriplewiseInt", "txt", checkpointId);
    }

    void restart(__UNUSED MPI_Comm comm, const std::string& path) override
    {
        auto fname = createCheckpointName(path, "TriplewiseInt", "txt");
        std::ifstream fin(fname);

        auto check = [&](bool good) {
            if (!good) die("failed to read '%s'\n", fname.c_str());
        };

        check(fin.good());
        check( kernel_.readState(fin) );
    }

    /// \return A string that describes the type of this object
    static std::string getTypeName()
    {
        return constructTypeName("TriplewiseInteraction", 1, TriplewiseKernel::getTypeName().c_str());
    }

    void saveSnapshotAndRegister(Saver& saver) override
    {
        saver.registerObject<TriplewiseInteraction>(
                this, _saveSnapshot(saver, getTypeName()));
    }

protected:
    /** \brief Serialize raw parameters of the kernel.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
    */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName)
    {
        ConfigObject config = BaseTriplewiseInteraction::_saveSnapshot(saver, typeName);
        config.emplace("params", saver(params_));
        return config;
    }

private:
    /** \brief Compute forces between all the triples of particles that are closer
        than rc to each other.
    */
    void _computeLocal(ParticleVector* pv1, ParticleVector* pv2, ParticleVector* pv3,
                       CellList* cl1, CellList* cl2, CellList* cl3, cudaStream_t stream)
    {
        using ViewType = typename TriplewiseKernel::ViewType;
        kernel_.setup(pv1->local(), pv2->local(), pv3->local(), cl1, cl2, cl3, getState());

        /*  Self interaction */
        if (pv1 == pv2 && pv2 == pv3 && pv3 == pv1)
        {
            auto view = cl1->getView<ViewType>();
            const int np = view.size;
            if(np >= 3){
                debug("Computing internal forces for %s (%d particles)", pv1->getCName(), np);

                const int nth = 128;

                auto cinfo = cl1->cellInfo();
                
                SAFE_KERNEL_LAUNCH(
                    computeTriplewiseSelfInteractions,
                    getNblocks(np, nth), nth, 0, stream,
                    cinfo, view, kernel_.handler());
            }
        }
        else /*  External interaction */
        {
            die("3-body interactions with two or three different PVs not implemented.");
        }
    }

    /** \brief Compute halo forces */
    void _computeHalo(ParticleVector *pv1, ParticleVector *pv2, ParticleVector *pv3, CellList *cl1, CellList *cl2, CellList *cl3, cudaStream_t stream)
    {
        using ViewType = typename TriplewiseKernel::ViewType;

        /*  Self interaction */
        if (pv1 == pv2 && pv2 == pv3 && pv3 == pv1)
        {
            //local-local-halo
            kernel_.setup(pv1->local(), pv2->local(), pv3->halo(), cl1, cl2, cl3, getState());
            auto view = cl1->getView<ViewType>();
            const int np_local = pv1->local()->size();
            const int np_halo = pv1->halo()->size();
            debug("Computing internal forces for %s (%d particles)", pv1->getCName(), np_local);

            const int nth = 128;

            auto cinfo = cl1->cellInfo();

            if(np_local > 0 && np_halo > 0){
                SAFE_KERNEL_LAUNCH(
                    computeTriplewiseSelfInteractions,      //ATM I use SelfInteractions, because no update for srcView implemented
                    getNblocks(np_halo, nth), nth, 0, stream,
                    cinfo, view, kernel_.handler());
                
                //local-halo-halo
                kernel_.setup(pv1->local(), pv2->halo(), pv3->halo(), cl1, cl2, cl3, getState());
                SAFE_KERNEL_LAUNCH(
                    computeTriplewiseSelfInteractions,
                    getNblocks(np_halo, nth), nth, 0, stream,
                    cinfo, view, kernel_.handler());
            }
        }
        else /*  External interaction */
        {
            die("3-body interactions with two or three different PVs not implemented.");
        }
    }

private:
    TriplewiseKernel kernel_;
    KernelParams params_;
};

} // namespace mirheo
