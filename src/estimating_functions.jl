"""
    estimating_function_template(nobs::Function, 
                                 ef_contribution::Function)

[Composite type](https://docs.julialang.org/en/v1/manual/types/#Composite-Types-1) for defining an `estimating_function_template`.

Arguments
===
+ `nobs`: a function of `data` that computes the number of observations of the particular data type,
+ `ef_contribution`: a function of the parameters `theta`, the `data` and the observation index `i` that returns a vector of length `length(theta)`.

Result
===
An `estimating_function_template` object with fields `nobs` and `obj_contributions`.
"""
struct estimating_function_template
    nobs::Function
    ef_contribution::Function
end

""" 
    estimating_function(theta::Vector,
                        data::Any,
                        template::estimating_function_template,
                        br::Bool = false,
                        concentrate::Vector{Int64} = Vector{Int64}())

Evaluate a vector of estimating functions at `theta` by adding up all contributions in `data`, according to an [`estimating_function_template`](@ref).

Arguments
===
+ `theta`: a `Vector` of parameter values at which to evaluate the estimating functions
+ `data`: typically an object of [composite type](https://docs.julialang.org/en/v1/manual/types/#Composite-Types-1) with all the data required to compute the `estimating_function`.
+ `template`: an [`estimating_function_template`](@ref) object.
+ `br`: a `Bool`. If `false` (default), the estimating functions is constructed by adding up all contributions in 
`data`, according to [`estimating_function_template`](@ref), before it is evaluated at `theta`. If `true` then the empirical bias-reducing adjustments in [Kosmidis & Lunardon, 2020](http://arxiv.org/abs/2001.03786) are computed and added to the estimating functions.
+ `concentrate`: a `Vector{Int64}`; if specified, empirical bias-reducing adjustments are added only to the subset of estimating functions indexed by `concentrate`. The default is to add empirical bias-reducing adjustments to all estimating functions.

Result
===
A `Vector`.

Details
===
`data` can be used to pass additional constants other than the actual data to the objective.
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
    get_estimating_function(data::Any,
                            template::estimating_function_template,
                            br::Bool = false,
                            concentrate::Vector{Int64} = Vector{Int64}(),
                            regularizer::Any = Vector{Int64}())

Construct the estimating functions by adding up all contributions in the `data` according to [`estimating_function_template`](@ref).

Arguments
===
+ `data`: typically an object of [composite type](https://docs.julialang.org/en/v1/manual/types/#Composite-Types-1) with all the data required to compute the `estimating_function`.
+ `template`: an [`estimating_function_template`](@ref) object.
+ `br`: a `Bool`. If `false` (default), the estimating functions is constructed by adding up all contributions in 
`data`, according to [`estimating_function_template`](@ref), before it is evaluated at `theta`. If `true` then the empirical bias-reducing adjustments in [Kosmidis & Lunardon, 2020](http://arxiv.org/abs/2001.03786) are computed and added to the estimating functions.
+ `concentrate`: a `Vector{Int64}`; if specified, empirical bias-reducing adjustments are added only to the subset of estimating functions indexed by `concentrate`. The default is to add empirical bias-reducing adjustments to all estimating functions.
+ `regularizer`: a function of `theta` and `data` returning a `Vector` of dimension equal to the number of the estimating functions, which is added to the (bias-reducing) estimating function; the default value will result in no regularization.

Result
===
An in-place function that stores the value of the estimating functions inferred from `template`, in a preallocated vector passed as its first argument, ready to be used withing `NLsolve.nlsolve`. This is the in-place version of [`estimating_function`](@ref) with the extra `regularizer` argument.
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
    estfun_i = (pars, i) -> template.ef_contribution(pars, data, i)
    function ja_i(eta::Vector, i::Int)
        out = similar(eta, p, p)
        ForwardDiff.jacobian!(out, pars -> estfun_i(pars, i), eta)
    end
    p = length(theta)
    n_obs = template.nobs(data)
    psi = zeros(p)
    emat = zeros(p, p)
    jmat = zeros(p, p)

    if adjustment
        function u(eta::Vector, i::Int)
            out = similar(eta, p * p, p)
            ForwardDiff.jacobian!(out, beta -> ja_i(beta, i), eta)
        end
        psi2 = Vector(undef, p)
        for j in 1:p
            psi2[j] = zeros(p, p)
        end
        # psi2 = zeros(p, p, p)
        umat = zeros(p * p, p)
        for i in 1:n_obs
            cpsi = estfun_i(theta, i)
            psi += cpsi
            emat += cpsi * cpsi'
            jaco = ja_i(theta, i)
            jmat += -jaco
            umat += u(theta, i)
            for j in 1:p
                psi2[j] += jaco[j, :] * cpsi'
            end
        end
    else
        for i in 1:n_obs
            cpsi = estfun_i(theta, i)
            psi += cpsi
            emat += cpsi * cpsi'
            jmat += - ja_i(theta, i)
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
                A = zeros(p)
                A[ist] = A_ist + inv(jmat_inv[ist, ist]) * jmat_inv[ist, nce] * A_nce
            end
        end
        [A, jmat_inv, emat, psi]
    else
        [vcov, jmat_inv, emat, psi]
    end
end
