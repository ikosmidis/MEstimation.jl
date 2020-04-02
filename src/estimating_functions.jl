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
    estimating_function(theta::Vector, data::Any, template::estimating_function_template, br::Bool = false, concentrate::Vector{Int64} = Vector{Int64}())

Construct the estimating function by adding up all contributions in the `data` according to [`estimating_function_template`](@ref), and evaluate it at `theta`. If `br = true` then automatic differentiation is used to compute the empirical bias-reducing adjustments and add them to the estimating function. Bias-reducing adjustments can be computed for only a subset of estimating functions by setting the ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `concentrate` to the vector of the indices for those estimating functions.
"""
function estimating_function(theta::Vector,
                             data::Any,
                             template::estimating_function_template,
                             br::Bool = false,
                             concentrate::Vector{Int64} = Vector{Int64}())
    p = length(theta)
    n_obs = template.nobs(data)
    ef = zeros(p)
    for i in 1:n_obs
        ef += template.ef_contribution(theta, data, i)
    end
    if (br)
        quants = ef_quantities(theta, data, template, br, concentrate)
        ef + quants[1]
    else
        ef
    end
end

""" 
    get_estimating_function(data::Any, template::estimating_function_template, br::Bool = false, concentrate::Vector{Int64} = Vector{Int64}(), regularizer::Function = function regularizer(theta::Vector{Float64}, data::Any) Vector{Float64}() end)

Construct the estimating function by adding up all contributions in the `data` according to [`estimating_function_template`](@ref). If `br = true` then automatic differentiation is used to compute the empirical bias-reducing adjustments and add them to the estimating function. Bias-reducing adjustments can be computed for only a subset of estimating functions by setting the ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `concentrate` to the vector of the indices for those estimating functions. The result is a function that stores the estimating functions evaluated at its second argument, in a preallocated vector passed as its first argument, ready to be used withing `NLsolve.nlsolve`. 

An extra additive regularizer to either the estimating functions or the adjusted estimating functions can be suplied via the [keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `regularizer`, which must be a function of the parameters and the data returning a real vector of dimension equal to the number of the estimating functions; the default value will result in no regularization.
"""
function get_estimating_function(data::Any,
                                 template::estimating_function_template,
                                 br::Bool = false,
                                 concentrate::Vector{Int64} = Vector{Int64}(),
                                 regularizer::Any = Vector{Int64}())
    ## regulizer here has different type and default than fit, because
    ## the dimension of theta cannot be inferred
    has_regularizer = typeof(regularizer) <: Function
    function (F, theta::Vector)
        if has_regularizer
            out = estimating_function(theta, data, template, br, concentrate) + regularizer(theta, data)
        else
             out = estimating_function(theta, data, template, br, concentrate)
        end
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
    function nj(eta::Vector, i::Int)
        out = similar(eta, p, p)
        ForwardDiff.jacobian!(out, beta -> template.ef_contribution(beta, data, i), eta)
    end
    p = length(theta)
    n_obs = template.nobs(data)
    psi = zeros(p)
    emat = zeros(p, p)
    jmat = zeros(p, p)

    if adjustment
        function u(eta::Vector, i::Int)
            out = similar(eta, p * p, p)
            ForwardDiff.jacobian!(out, beta -> nj(beta, i), eta)
        end
        psi2 = Vector(undef, p)
        for j in 1:p
            psi2[j] = zeros(p, p)
        end
        # psi2 = zeros(p, p, p)
        umat = zeros(p * p, p)
        for i in 1:n_obs
            cpsi = template.ef_contribution(theta, data, i)
            psi += cpsi
            emat += cpsi * cpsi'
            hess = nj(theta, i)
            jmat += -hess
            umat += u(theta, i)
            for j in 1:p
                psi2[j] += hess[j, :] * cpsi'
            end
        end
    else
        for i in 1:n_obs
            cpsi = template.ef_contribution(theta, data, i)
            psi += cpsi
            emat += cpsi * cpsi'
            jmat += -nj(theta, i)
        end
    end
      
    jmat_inv = try
        inv(jmat)
    catch
        fill(NaN, p, p)
    end
    vcov = jmat_inv * (emat * jmat_inv') 

    if adjustment
        Afun(j::Int64) = -tr(jmat_inv * psi2[j] + vcov * umat[j:p:(p * p - p + j), :] / 2)
        A = map(Afun, 1:p)
        ## if concentrate then redefine A
        if length(concentrate) > 0
            if any((concentrate .> p) .| (concentrate .< 1))
                error(concentrate, " should be a vector of integers between ", 1, " and ", p)
            else
                ist = concentrate
                nce = deleteat!(collect(1:p), concentrate)
                A_ist = A[ist]
                A_nce = A[nce]
                A = vcat(A_ist + inv(jmat_inv[ist, ist]) * jmat_inv[ist, nce] * A_nce,
                         zeros(length(nce)))
            end
        end
        [A, jmat_inv, emat, psi]
    else
        [vcov, jmat_inv, emat, psi]
    end
end
