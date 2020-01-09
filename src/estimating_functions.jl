"""
    estimating_function_template(nobs::Function, ef_contribution::Function)

Define an `estimating_function_template` by supplying:
+ `nobs`: a function of `data` that computes the number of observations of the particular data type,
+ `ef_contribution`: a function of the parameters `theta`, the `data` and the observation index `i` that returns a vector of length `length(theta)`.
"""
struct estimating_function_template
    nobs::Function
    ef_contribution::Function
end

function ef_quantities(theta::Vector,
                       data::Any,
                       template::estimating_function_template,
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

""" 
    estimating_function(theta::Vector, data::Any, template::estimating_function_template, br::Bool = false)

Construct the estimating function by adding up all contributions in the `data` according to [`estimating_function_template`](@ref), and evaluate it at `theta`. If `br = true` then automatic differentiation is used to compute the empirical bias-reducing adjustments and add them to the estimating function.
"""
function estimating_function(theta::Vector,
                             data::Any,
                             template::estimating_function_template,
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

function get_estimating_function(data::Any,
                                 template::estimating_function_template,
                                 br::Bool = false)
    function g(F, theta::Vector) 
        out = estimating_function(theta, data, template, br)
        for i in 1:length(out)
            F[i] = out[i]
        end
    end
end

"""   
    fit(template::estimating_function_template, data::Any, theta::Vector, br::Bool = false; nlsolve_arguments...)

"""
function fit(template::estimating_function_template,
             data::Any,
             theta::Vector,
             br::Bool = false;
             nlsolve_arguments...)
    ef = get_estimating_function(data, template, br)
    out = nlsolve(ef, theta; nlsolve_arguments...)
    GEEBRA_results(out, out.zero, data, template, br, false)
end
