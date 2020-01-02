module GEEBRA

using NLsolve, ForwardDiff, LinearAlgebra
export objective_function, estimating_function, get_estimating_function, solve_estimating_equation, geebra_ef_template, geebra_obj_template

struct geebra_ef_template
    nobs::Function
    ef_contribution::Function
end

struct geebra_obj_template
    nobs::Function
    obj_contribution::Function
end

## Objective
function objective_function(theta::Vector,
                            data::Any,
                            template::geebra_obj_template,
                            br::Bool = false)
    p = length(theta)
    n_obs = template.nobs(data)
    contributions = Vector(undef, n_obs)
    for i in 1:n_obs
        contributions[i] = template.obj_contribution(theta, data, i)
    end
    if (br)
        quants = obj_quantities(theta, data, template, br)
        sum(contributions) + quants[2]
    else
        sum(contributions)
    end
end


## Estimating function 
function estimating_function(theta::Vector,
                             data::Any,
                             template::geebra_ef_template,
                             br::Bool = false)
    p = length(theta)
    n_obs = template.nobs(data)
    contributions = Matrix(undef, p, n_obs)
    for i in 1:n_obs
        contributions[:, i] = template.ef_contribution(theta, data, i)
    end
    if (br)
        quants = ef_quantities(theta, data, template, br)
        sum(contributions, dims = 2) + quants[2]
    else
        sum(contributions, dims = 2)
    end
end

## Esimating function 
function get_estimating_function(data::Any,
                                 template::geebra_ef_template,
                                 br::Bool = false)
    function g(F, theta::Vector) 
        out = estimating_function(theta, data, template, br)
        for i in 1:length(out)
            F[i] = out[i]
        end
    end
end

## nlsolve_argumentsa are further arguments to be passed to nlsolve
function solve_estimating_equation(theta::Vector,
                                   data::Any,
                                   template::geebra_ef_template,
                                   br::Bool = false;
                                   nlsolve_arguments...)
    ef = get_estimating_function(data, template, br)
    nlsolve(ef, theta; nlsolve_arguments...)
end

function ef_quantities(theta::Vector,
                       data::Any,
                       template::geebra_ef_template,
                       adjustment::Bool = false)
    nj(eta::Vector, i::Int) = ForwardDiff.jacobian(beta -> template.ef_contribution(beta, data, i), eta)
    p = length(theta)
    n_obs = template.nobs(data)
    psi = Matrix(undef, n_obs, p)
    njmats = Vector(undef, n_obs)
    for i in 1:n_obs
        psi[i, :] =  template.ef_contribution(theta, data, i) 
        njmats[i] = nj(theta, i)
    end
    jmat_inv = inv(-sum(njmats))
    vcov = jmat_inv * psi'
    vcov = vcov * vcov'
    if (adjustment)
        u(eta::Vector, i::Int) = ForwardDiff.jacobian(beta -> nj(beta, i), eta)
        psi_tilde = Matrix(undef, n_obs, p)
        umats = Vector(undef, n_obs)
        for i in 1:n_obs
            umats[i] = u(theta, i)
        end       
        umat = sum(umats)
        A = Vector(undef, p)
        for j in 1:p
            for i in 1:n_obs
                psi_tilde[i, :] = njmats[i][j, :]
            end
            A[j] = -tr(jmat_inv * psi_tilde' * psi +
                       vcov * umat[j:p:(p*p - p + j), :] / 2)
        end
        [vcov, A]
    else
        [vcov]
    end
end

function obj_quantities(theta::Vector,
                        data::Any,
                        template::geebra_obj_template,
                        penalty::Bool = false)
    npsi(eta::Vector, i::Int) = ForwardDiff.gradient(beta -> template.obj_contribution(beta, data, i), eta)
    nj(eta::Vector, i::Int) = ForwardDiff.hessian(beta -> template.obj_contribution(beta, data, i), eta)
    p = length(theta)
    n_obs = template.nobs(data)
    psi = Matrix(undef, n_obs, p)
    njmats = Vector(undef, n_obs)
    for i in 1:n_obs
        psi[i, :] =  npsi(theta, i)
        njmats[i] = nj(theta, i)
    end
    jmat_inv = inv(-sum(njmats))
    emat = psi' * psi
    emat = convert(Array{Float64, 2}, emat)
    vcov = jmat_inv * (emat * jmat_inv)
    if (penalty)        
        penalty = - tr(jmat_inv * emat) / 2
        [vcov, penalty]
    else
        [vcov]
    end
end


end # module

