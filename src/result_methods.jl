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

Composite type for the output of [`fit`](@ref) for an [`objective_function_template`](@ref) or an [`estimating_function_template`](@ref).
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

Compute an esitmate of the variance-covariance matrix of the `M`-estimator or its reduced-bias version from the output of [`fit`](@ref) for an [`objective_function_template`](@ref) or an [`estimating_function_template`](@ref).
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

Compute the Takeuchi Information Criterion at the `M`-estimator or its reduced-bias version from the output of [`fit`](@ref) for an [`objective_function_template`](@ref). `nothing` is returned if `results` is the output of [`fit`](@ref) for an [`estimating_function_template`](@ref).
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

Compute the Akaike Information Criterion at the `M`-estimator or its reduced-bias version from the output of [`fit`](@ref) for an [`objective_function_template`](@ref). `nothing` is returned if `results` is the output of [`fit`](@ref) for an [`estimating_function_template`](@ref).
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

Extract the `M`-estimates or their reduced-bias versions from the output of [`fit`](@ref) for an [`objective_function_template`](@ref) or an [`estimating_function_template`](@ref).
"""
function coef(results::MEstimation_results)
    results.theta
end

"""
    show(io::IO, results::MEstimation_results; digits::Real = 4)

`show` method for `MEstimation_results` objects. If `MEstimation_results.has_object == true`, then the result of `aic(results)` and `tic(results)` are also printed.
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

Compute esitmated standard errors for the `M`-estimator or its reduced-bias version from the output of [`fit`](@ref) for an [`objective_function_template`](@ref) or an [`estimating_function_template`](@ref).
"""
function stderror(results::MEstimation_results)
    sqrt.(diag(vcov(results)))
end

"""
    coeftable(results::MEstimation_results; 
              level::Real=0.95)

Return a `StatsBase.CoefTable` for the `M`-estimator or its reduced-bias version from the output of [`fit`](@ref) for an [`objective_function_template`](@ref) or an [`estimating_function_template`](@ref). `level` can be used to set the level of the reported Wald-type confidence intervals (using quantiles of the standard normal distribution). 
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
