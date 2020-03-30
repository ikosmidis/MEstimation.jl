function slice(results::GEEBRA_results,
               what::Int64,
               range::Vector{Float64},
               at::Vector{Float64} = Vector{Float64}())
    data = results.data
    theta = length(at) > 0 ? at : coef(results)
    n_par = length(theta)
    model_template = results.template
    br = results.br
    regularizer = results.regularizer    
    if results.has_regularizer > 0
        if results.has_objective
            slice_function = beta -> objective_function(beta, data, model_template, br) + regularizer(beta, data)
        else
            slice_function = beta -> estimating_function(beta, data, model_template, br)[what] + regularizer(beta, data)[what]
        end
    else
        if results.has_objective
            slice_function = beta -> objective_function(beta, data, model_template, br)
        else
            slice_function = beta -> estimating_function(beta, data, model_template, br)[what]
        end
    end
    slice_m = map(function(b) th = theta[1:n_par]; th[what] = b; slice_function(th) end, range)
    Dict("value" => range, "slice" => slice_m)
end
