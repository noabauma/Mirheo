// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"
#include "parameters.h"

#include <mirheo/core/utils/cuda_common.h>
#include <array>

namespace mirheo
{

struct SW3Handler
{
    using ViewType = PVview;
    using ParticleType = Particle;

    SW3Handler(real rc, real epsilon, real lambda, real theta, real gamma, real sigma) : 
        rc_(rc),
        rc2_(rc * rc), 
        lambda_epsilon_(lambda*epsilon),
        cos_theta_(math::cos(theta)),   //theta = 1.910633236 -> cos(theta) ~ -1/3
        gamma_sigma_(gamma*sigma)
        {}

    __D__ inline std::array<real3, 3> operator()(ParticleType p_i, ParticleType p_j, ParticleType p_k, int id_i, int id_j, int id_k) const
    {
        const real3 r_ij = p_i.r - p_j.r;
        const real3 r_jk = p_j.r - p_k.r;
        const real3 r_ki = p_k.r - p_i.r;

        const real dr_ij2 = dot(r_ij, r_ij);
        const real dr_jk2 = dot(r_jk, r_jk);
        const real dr_ki2 = dot(r_ki, r_ki);

        if(dr_ij2 >= rc2_){
            if(dr_jk2 < rc2_ && dr_ki2 < rc2_){     //(ij, jk, ki): (n,y,y)
                //return -h_ikj
                const real dr_jk = math::sqrt(dr_jk2);
                const real dr_ki = math::sqrt(dr_ki2);

                const real3 r_jk_hat = r_jk/dr_jk;
                const real3 r_ki_hat = r_ki/dr_ki;

                const real cos_theta_ikj = -dot(r_ki_hat, r_jk_hat);

                const real cos_cos = (cos_theta_ikj - cos_theta_);

                const real exp = math::exp(gamma_sigma_/(dr_ki-rc_) + gamma_sigma_/(dr_jk-rc_));

                const real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

                const real3 grad_i = exp_lambda_epsilon_cos_cos*(2.0_r*( r_jk_hat/dr_ki + cos_theta_ikj*r_ki_hat/dr_ki) + cos_cos*r_ki_hat*gamma_sigma_/((dr_ki-rc_)*(dr_ki-rc_)));
                const real3 grad_j = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ki_hat/dr_jk - cos_theta_ikj*r_jk_hat/dr_jk) - cos_cos*r_jk_hat*gamma_sigma_/((dr_jk-rc_)*(dr_jk-rc_)));
                const real3 grad_k = -grad_i - grad_j;

                return {-grad_i, -grad_j, -grad_k};
            }else{                                  //(ij, jk, ki): (n,y,n),(n,n,y),(n,n,n)
                const real3 zero = make_real3(0.0_r);
                return {zero, zero, zero};
            }
        }else if(dr_jk2 >= rc2_ && dr_ki2 >= rc2_){ //(ij, jk, ki): (y,n,n)
            const real3 zero = make_real3(0.0_r);
            return {zero, zero, zero};
        }else{                                      //(ij, jk, ki): (y,y,y),(y,y,n),(y,n,y)
            if(dr_jk2 < rc2_ && dr_ki2 < rc2_){     //(ij, jk, ki): (y,y,y)
                //return -(h_jik+h_ijk+h_ikj)
                const real dr_ij = math::sqrt(dr_ij2);
                const real dr_jk = math::sqrt(dr_jk2);
                const real dr_ki = math::sqrt(dr_ki2);

                const real3 r_ij_hat = r_ij/dr_ij;
                const real3 r_jk_hat = r_jk/dr_jk;
                const real3 r_ki_hat = r_ki/dr_ki;

                const real cos_theta_jik = -dot(r_ij_hat, r_ki_hat);
                const real cos_theta_ijk = -dot(r_ij_hat, r_jk_hat);
                const real cos_theta_ikj = -dot(r_ki_hat, r_jk_hat);

                //h_jik
                real cos_cos = (cos_theta_jik - cos_theta_);

                real exp = math::exp(gamma_sigma_/(dr_ij-rc_) + gamma_sigma_/(dr_ki-rc_));

                real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

                const real3 h_jik_j = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ki_hat/dr_ij + cos_theta_jik*r_ij_hat/dr_ij) + cos_cos*r_ij_hat*gamma_sigma_/((dr_ij-rc_)*(dr_ij-rc_)));
                const real3 h_jik_k = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ij_hat/dr_ki - cos_theta_jik*r_ki_hat/dr_ki) - cos_cos*r_ki_hat*gamma_sigma_/((dr_ki-rc_)*(dr_ki-rc_)));
                const real3 h_jik_i = -h_jik_j - h_jik_k;

                //h_ijk
                cos_cos = (cos_theta_ijk - cos_theta_);

                exp = math::exp(gamma_sigma_/(dr_ij-rc_) + gamma_sigma_/(dr_jk-rc_));

                exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

                const real3 h_ijk_i = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_jk_hat/dr_ij - cos_theta_ijk*r_ij_hat/dr_ij) - cos_cos*r_ij_hat*gamma_sigma_/((dr_ij-rc_)*(dr_ij-rc_)));
                const real3 h_ijk_k = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ij_hat/dr_jk + cos_theta_ijk*r_jk_hat/dr_jk) + cos_cos*r_jk_hat*gamma_sigma_/((dr_jk-rc_)*(dr_jk-rc_)));
                const real3 h_ijk_j = -h_ijk_i - h_ijk_k;

                //h_ikj
                cos_cos = (cos_theta_ikj - cos_theta_);

                exp = math::exp(gamma_sigma_/(dr_ki-rc_) + gamma_sigma_/(dr_jk-rc_));

                exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

                const real3 h_ikj_i = exp_lambda_epsilon_cos_cos*(2.0_r*( r_jk_hat/dr_ki + cos_theta_ikj*r_ki_hat/dr_ki) + cos_cos*r_ki_hat*gamma_sigma_/((dr_ki-rc_)*(dr_ki-rc_)));
                const real3 h_ikj_j = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ki_hat/dr_jk - cos_theta_ikj*r_jk_hat/dr_jk) - cos_cos*r_jk_hat*gamma_sigma_/((dr_jk-rc_)*(dr_jk-rc_)));
                const real3 h_ikj_k = -h_ikj_i - h_ikj_j;

                return {-(h_jik_i + h_ijk_i + h_ikj_i), -(h_jik_j + h_ijk_j + h_ikj_j), -(h_jik_k + h_ijk_k + h_ikj_k)};

            }else if(dr_jk2 < rc2_){                //(ij, jk, ki): (y,y,n)
                //return -h_ijk
                const real dr_ij = math::sqrt(dr_ij2);
                const real dr_jk = math::sqrt(dr_jk2);

                const real3 r_ij_hat = r_ij/dr_ij;
                const real3 r_jk_hat = r_jk/dr_jk;

                const real cos_theta_ijk = -dot(r_ij_hat, r_jk_hat);

                const real cos_cos = (cos_theta_ijk - cos_theta_);

                const real exp = math::exp(gamma_sigma_/(dr_ij-rc_) + gamma_sigma_/(dr_jk-rc_));

                const real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

                const real3 grad_i = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_jk_hat/dr_ij - cos_theta_ijk*r_ij_hat/dr_ij) - cos_cos*r_ij_hat*gamma_sigma_/((dr_ij-rc_)*(dr_ij-rc_)));
                const real3 grad_k = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ij_hat/dr_jk + cos_theta_ijk*r_jk_hat/dr_jk) + cos_cos*r_jk_hat*gamma_sigma_/((dr_jk-rc_)*(dr_jk-rc_)));
                const real3 grad_j = -grad_i - grad_k;

                return {-grad_i, -grad_j, -grad_k};
            }else{                                  //(ij, jk, ki): (y,n,y)
                //return -h_jik
                const real dr_ij = math::sqrt(dr_ij2);
                const real dr_ki = math::sqrt(dr_ki2);

                const real3 r_ij_hat = r_ij/dr_ij;
                const real3 r_ki_hat = r_ki/dr_ki;

                const real cos_theta_jik = -dot(r_ij_hat, r_ki_hat);

                const real cos_cos = (cos_theta_jik - cos_theta_);

                const real exp = math::exp(gamma_sigma_/(dr_ij-rc_) + gamma_sigma_/(dr_ki-rc_));

                const real exp_lambda_epsilon_cos_cos = exp*lambda_epsilon_*cos_cos;

                const real3 grad_j = exp_lambda_epsilon_cos_cos*(2.0_r*( r_ki_hat/dr_ij + cos_theta_jik*r_ij_hat/dr_ij) + cos_cos*r_ij_hat*gamma_sigma_/((dr_ij-rc_)*(dr_ij-rc_)));
                const real3 grad_k = exp_lambda_epsilon_cos_cos*(2.0_r*(-r_ij_hat/dr_ki - cos_theta_jik*r_ki_hat/dr_ki) - cos_cos*r_ki_hat*gamma_sigma_/((dr_ki-rc_)*(dr_ki-rc_)));
                const real3 grad_i = -grad_j - grad_k;

                return {-grad_i, -grad_j, -grad_k};
            }
        }
    }

    real rc_;
    real rc2_;
    real lambda_epsilon_;
    real cos_theta_;
    real gamma_sigma_;
};

class TriplewiseDummy : public TriplewiseKernelBase
{
public:
    using HandlerType = TriplewiseDummyHandler; ///< handler type corresponding to this object
    using ParamsType  = DummyParams; ///< parameters that are used to create this object
    using ViewType    = PVview;     ///< compatible view type

    /// Constructor
    TriplewiseDummy(real rc, real epsilon) : handler_(rc, epsilon) {}

    /// Generic constructor
    TriplewiseDummy(real rc, const ParamsType& p, __UNUSED long seed=42424242) :
        TriplewiseDummy(rc, p.epsilon)
    {}

    /// get the handler that can be used on device
    const HandlerType& handler() const { return handler_; }

    /// \return type name string
    static std::string getTypeName() { return "TriplewiseDummy"; }

private:
    TriplewiseDummyHandler handler_;
};

} // namespace mirheo
