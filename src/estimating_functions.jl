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

""" 
    estimating_function(theta::Vector, data::Any, template::estimating_function_template, br::Bool = false)

Construct the estimating function by adding up all contributions in the `data` according to [`estimating_function_template`](@ref), and evaluate it at `theta`. If `br = true` then automatic differentiation is used to compute the empirical bias-reducing adjustments and add them to the estimating function. Bias-reducing adjustments can be computed for only a subset of estimating functions by setting the ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `concentrate` to the vector of the indices for those estimating functions.
"""
function estimating_function(theta::Vector,
                             data::Any,
                             template::estimating_function_template,
                             br::Bool = false,
                             concentrate::Vector{Int64} = Vector{Int64}())
    p = length(theta)
    n_obs = template.nobs(data)
    contributions = Matrix(undef, p, n_obs)
    for i in 1:n_obs
        contributions[:, i] = template.ef_contribution(theta, data, i)
    end
    if (br)
        quants = ef_quantities(theta, data, template, br, concentrate)
        ef = sum(contributions, dims = 2) 
        ef + quants[1]
    else
        sum(contributions, dims = 2)
    end
end

""" 
    get_estimating_function(data::Any, template::estimating_function_template, br::Bool = false)

Construct the estimating function by adding up all contributions in the `data` according to [`estimating_function_template`](@ref). If `br = true` then automatic differentiation is used to compute the empirical bias-reducing adjustments and add them to the estimating function. Bias-reducing adjustments can be computed for only a subset of estimating functions by setting the ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `concentrate` to the vector of the indices for those estimating functions. The result is a function that stores the estimating functions values at its second argument, in a preallocated vector passed as its first argument, ready to be used withing `NLsolve.nlsolve`. 
"""
function get_estimating_function(data::Any,
                                 template::estimating_function_template,
                                 br::Bool = false,
                                 concentrate::Vector{Int64} = Vector{Int64}())
    function g!(F, theta::Vector) 
        out = estimating_function(theta, data, template, br, concentrate)
        for i in 1:length(out)
            F[i] = out[i]
        end
    end
end

function ef_quantities(theta::Vector,
                       data::Any,
                       template::estimating_function_template,
                       adjustment::Bool = false,
                       concentrate::Vector{Int64} = Vector{Int64}())
    nj(eta::Vector, i::Int) = ForwardDiff.jacobian(beta -> template.ef_contribution(beta, data, i), eta)
    p = length(theta)
    n_obs = template.nobs(data)
    psi = Matrix{Float64}(undef, n_obs, p)
    njmats = Vector(undef, n_obs)
    for i in 1:n_obs
        psi[i, :] =  template.ef_contribution(theta, data, i) 
        njmats[i] = nj(theta, i)
    end
    jmat_inv = inv(-sum(njmats))
    emat = psi' * psi
    vcov = jmat_inv * (emat * jmat_inv')
    # vcov = jmat_inv * psi'
    # vcov = vcov * vcov'
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
                       vcov * umat[j:p:(p * p - p + j), :] / 2)
        end
        ## if concentrate then redefine A
        if length(concentrate) > 0
            if any((concentrate .> p) .| (concentrate .< 1))
                error(concentrate, " should be a vector of integers between ", 1, " and ", p)
            else
                psi = concentrate
                lam = deleteat!(collect(1:p), concentrate)
                A_psi = A[psi]
                A_lam = A[lam]
                A = vcat(A_psi + inv(jmat_inv[psi, psi]) * jmat_inv[psi, lam] * A_lam,
                         zeros(length(lam)))
            end
        end        
        [A, jmat_inv, emat]
    else
        [vcov, jmat_inv, emat]
    end
end

