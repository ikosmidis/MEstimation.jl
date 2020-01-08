struct GEEBRA_results
    results::Union{NLsolve.SolverResults, Optim.MultivariateOptimizationResults, Optim.UnivariateOptimizationResults}
    theta::Vector
    data::Any
    template::Union{objective_function_template, estimating_function_template}
    br::Bool
    has_objective::Bool
end

function vcov(results::GEEBRA_results)
    if (results.has_objective)
        obj_quantities(results.theta, results.data, results.template, false)[1]
    else
        ef_quantities(results.theta, results.data, results.template, false)[1]
    end
end

function tic(results::GEEBRA_results)
    if (results.has_objective)
        obj = objective_function(results.theta, results.data, results.template, false)
        quants = obj_quantities(results.theta, results.data, results.template, true)
        -2 * (obj + 2 * quants[2])
    end
end

function aic(results::GEEBRA_results)
    if (results.has_objective)
        obj = objective_function(results.theta, results.data, results.template, false)
        p = length(results.theta)
        -2 * (obj - p)
    end
end


function coef(results::GEEBRA_results)
    results.theta
end

function Base.show(io::IO, ::MIME"text/plain", results::GEEBRA_results;
                   digits::Real = 4)
    theta = results.theta
    p = length(theta)
    v = vcov(results)
    if results.has_objective
        println("M-estimation with objective contributions ",
                results.template.obj_contribution)
    else
        println("M-estimation with estimating function contributions ",
                results.template.ef_contribution)
    end
    println("Bias reduction: ", results.br)
    println()
    println("Parameter\tEstimate\tS.E")
    for i in 1:p
        est = theta[i]
        std = sqrt(v[i, i])
        println("theta[$(i)]", "\t", round(est, digits = digits), "\t\t", round(std, digits = digits))
    end
    if results.has_objective
        if results.br
            Base.print("\nMaximum penalized objetive:\t", round(-results.results.minimum, digits = digits))
        else
            Base.print("\nMaximum objetive:\t\t", round(-results.results.minimum, digits = digits))
        end
        Base.print("\nTakeuchi information criterion:\t", round(tic(results), digits = digits))
        Base.print("\nAkaike information criterion:\t", round(aic(results), digits = digits))
    else
        estfun = estimating_function(results.theta, results.data, results.template, results.br)
        if results.br
            Base.print("\nAdjusted estimating functions:\t", estfun)
        else
            Base.print("\nEstimating functions:\t", estfun)
        end
    end     
end

function stderror(results::GEEBRA_results)
    sqrt.(diag(vcov(results)))
end

function coeftable(results::GEEBRA_results; level::Real=0.95)
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
