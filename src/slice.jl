"""
    slice(results::MEstimation_results,
          what::Int64;
          range::Vector{Float64} = Vector{Float64}(),
          at::Vector{Float64} = Vector{Float64}())

Compute 1-dimensional slices of objective functions and estimating function surfaces, from a [`MEstimation_results`](@ref) object.

Arguments
===
+ `results`
+ `what`

[Keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1)
===
+ `range`
+ `at`

Details
===
The default value of range 

"""
function slice(results::MEstimation_results,
               what::Int64;
               range::Vector{Float64} = Vector{Float64}(),
               at::Vector{Float64} = Vector{Float64}())
    data = results.data
    theta = length(at) > 0 ? at : coef(results)
    if length(range) == 0
        
    end
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

