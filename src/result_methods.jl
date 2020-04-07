"""
    MEstimation_results(results::Union{NLsolve.SolverResults, Optim.MultivariateOptimizationResults, Optim.UnivariateOptimizationResults},
                        theta::Vector,
                        data::Any,
                        template::Union{objective_function_template, estimating_function_template},
                        regularizer::Function,
                        br::Bool,
                        has_objective::Bool,
                        has_regularizer::Bool,
                        br_method::String)

[Composite type](https://docs.julialang.org/en/v1/manual/types/#Composite-Types-1) for the output of [`fit`](@ref) for an [`objective_function_template`](@ref) or an [`estimating_function_template`](@ref).

Arguments
===
+ `results`: a `Union{NLsolve.SolverResults, Optim.MultivariateOptimizationResults, Optim.UnivariateOptimizationResults}` object holding the optimization results as returned from `Optim.optimize` or `NLsolve.nlsolve`.
+ `theta`: a `Vector{Float64}` with the M-estimates or their reduced-bias version.
+ `data`: an object of [composite type](https://docs.julialang.org/en/v1/manual/types/#Composite-Types-1) with all the data required to compute the objective function or the estimating functions; see [`objective_function`](@ref)` and [`estimating_function`](@ref).
+ `template`: an [`objective_function_template`](@ref) or [`estimating_function_template`](@ref) object.
+ `regularizer`: a function of `theta` and `data` returning either a `Vector{Float64}` of dimension equal to the number of the estimating functions (for [`estimating_function_template`](@ref)) or a `Float64` (for [`objective_function_template`](@ref)). See [`fit`](@ref) for details.
+ `br`: a `Bool`; if `false` then `results` are from the computation of M-estimates, otherwise from the computation of the RBM-estimates.
+ `has_objective`: a `Bool`; if `true` then `template` is an [`objective_function_template`](@ref). 
+ `has_regularizer`: a `Bool`; if `true` then a regularizer function has been used during optimization; see [`fit`](@ref).
+ `br_method`: either "implicit_trace" (default) or "explicit_trace"; see [`fit`](@ref).
"""
struct MEstimation_results
    results::Union{NLsolve.SolverResults, Optim.MultivariateOptimizationResults, Optim.UnivariateOptimizationResults}
    theta::Vector
    data::Any
    template::Union{objective_function_template, estimating_function_template}
    regularizer::Function
    br::Bool
    has_objective::Bool
    has_regularizer::Bool
    br_method::String
end

"""
    vcov(results::MEstimation_results)

Compute an estimate of the variance-covariance matrix of the `M`-estimator or its reduced-bias version at `results.theta`, from a [`MEstimation_results`](@ref) object.

Arguments
===
+ `results`: a [`MEstimation_results`](@ref) object.

Result
===
The `length(coef(results))` times `length(coef(results))` estimated variance covariance matrix for the parameters. This matrix is the empirical sandwich variance covariance matrix for M- and RBM-estimators. See, for example, [Stefanski and Boos (2002, expression 10)](https://www.jstor.org/stable/3087324).

"""
function vcov(results::MEstimation_results)
    if (results.has_objective)
        obj_quantities(results.theta, results.data, results.template, false)[1]
    else
        ef_quantities(results.theta, results.data, results.template, false)[1]
    end
end

"""
    tic(results::MEstimation_results)

Compute the Takeuchi Information Criterion at `results.theta`, from a [`MEstimation_results`](@ref) object.

Arguments
===
+ `results`: a [`MEstimation_results`](@ref) object.

Details
===
`nothing` is returned if `results.template` is an [`estimating_function_template`](@ref).
"""
function tic(results::MEstimation_results)
    if (results.has_objective)
        obj = objective_function(results.theta, results.data, results.template, false)
        quants = obj_quantities(results.theta, results.data, results.template, true)
        -2 * (obj + 2 * quants[1])
    end
end

"""
    aic(results::MEstimation_results)

Compute the Akaike Information Criterion at `results.theta`, from a [`MEstimation_results`](@ref) object with an [`objective_function_template`](@ref). 

Arguments
===
+ `results`: a [`MEstimation_results`](@ref) object.

Details
===
`nothing` is returned if `results.template` is an [`estimating_function_template`](@ref).
"""
function aic(results::MEstimation_results)
    if (results.has_objective)
        obj = objective_function(results.theta, results.data, results.template, false)
        p = length(results.theta)
        -2 * (obj - p)
    end
end

"""
    coef(results::MEstimation_results)

Extract the parameter estimates from a [`MEstimation_results`](@ref) object.

Arguments
===
+ `results`: a [`MEstimation_results`](@ref) object.

Details
===
`coef(results)` returns `results.theta`
"""
function coef(results::MEstimation_results)
    results.theta
end

"""
    show(io::IO, 
         results::MEstimation_results; 
         digits::Int64 = 4)

`show` method for `MEstimation_results` objects. 

Arguments
===
+ `io`: an `IO` object; see [`show`](@ref) for details.
+ `results`: a [`MEstimation_results`](@ref) object.

[Keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1)
===
+ `digits`: an `Int64` indicating the number of digits to display for the various summaries. Default is `4`.

Details
===
If `MEstimation_results.has_objective == true`, then the result of `aic(results)` and `tic(results)` are also printed.
"""
function Base.show(io::IO, results::MEstimation_results;
                   digits::Real = 4)
    theta = results.theta
    p = length(theta)
    v = vcov(results)
    if results.has_objective
        if results.has_regularizer
            println(io,
                    (results.br ? "RBM" : "M") * "-estimation with objective contributions `",
                    results.template.obj_contribution, "` and user-supplied regularizer")
        else
            println(io,
                    (results.br ? "RBM" : "M") * "-estimation with objective contributions `",
                    results.template.obj_contribution, "`")
        end
    else
        if results.has_regularizer
            println(io,
                    (results.br ? "RBM" : "M") * "-estimation with estimating function contributions `",
                    results.template.ef_contribution, "` and user-supplied regularizer")
        else
            println(io,
                    (results.br ? "RBM" : "M") * "-estimation with estimating function contributions `",
                    results.template.ef_contribution, "`")
        end
    end
    if (results.br) 
        println(io, "Bias reduction method: ", results.br_method)
    end
    println(io)
    # println("Parameter\tEstimate\tS.E")
    # for i in 1:p
    #     est = theta[i]
    #     std = sqrt(v[i, i])
    #     println("theta[$(i)]", "\t", round(est, digits = digits), "\t\t", round(std, digits = digits))
    # end
    show(io, coeftable(results))
    if results.has_objective
        objfun = objective_function(results.theta, results.data, results.template, results.br)
        if results.br
            print(io, "\nPenalized objetive:\t\t", round(objfun, digits = digits))
        else
            print(io, "\nObjective:\t\t\t", round(objfun, digits = digits))
        end
        print(io, "\nTakeuchi information criterion:\t", round(tic(results), digits = digits))
        print(io, "\nAkaike information criterion:\t", round(aic(results), digits = digits))
        print(io, "\nConverged: ", Optim.converged(results.results))
    else
        estfun = estimating_function(results.theta, results.data, results.template, results.br)
        if results.br
            print(io, "\nAdjusted estimating functions:\t", estfun)
        else
            print(io, "\nEstimating functions:\t", estfun)
        end
        print(io, "\nConverged: ", NLsolve.converged(results.results))
    end     
end

"""
    stderror(results::MEstimation_results)

Compute estimated standard errors from a from a [`MEstimation_results`](@ref) object.

Arguments
===
+ `results`: a [`MEstimation_results`](@ref) object.

Details
===
The estimated standard errors are computed as `sqrt.(diag(vcov(results)))`.
"""
function stderror(results::MEstimation_results)
    sqrt.(diag(vcov(results)))
end

"""
    coeftable(results::MEstimation_results; 
              level::Real=0.95)

Return a `StatsBase.CoefTable` from a [`MEstimation_results`](@ref) object. 

Arguments
===
+ `results`: a [`MEstimation_results`](@ref) object.

[Keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1)
===
+ `level`: a `Real` that determines the level of the reported confidence intervals; default is `0.95`; see Details.

Details
===
The reported confidence intervals are Wald-type of nominal level `level`, using quantiles of the standard normal distribution.
"""
function coeftable(results::MEstimation_results; level::Real=0.95)
    cc = coef(results)
    se = stderror(results)
    zz = cc ./ se
    p = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se * quantile(Normal(), (1-level)/2)
    levstr = isinteger(level*100) ? string(Integer(level*100)) : string(level*100)
    CoefTable(hcat(cc, se, zz, p, cc + ci, cc - ci),
              ["Estimate","Std. Error","z value","Pr(>|z|)","Lower $levstr%","Upper $levstr%"],
              ["theta[$i]" for i = 1:length(cc)], 4)
end
