"""
    slice(results::MEstimation_results,
          what::Int64;
          grid::Vector{Float64} = Vector{Float64}(),
          at::Vector{Float64} = Vector{Float64}(),
          n_points::Int64 = 50,
          n_sd::Real = 2)

Compute 1-dimensional slices of objective functions and estimating function surfaces for the parameter `what` over a grid of points `grid`, from a [`MEstimation_results`](@ref) object.

Arguments
===
+ `results`
+ `what`

[Keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1)
===
+ `grid`
+ `at`
+ `n_points`
+ `n_sd`

Details
===
The default value of `grid` will result in the automatic calculation of a grid of `n_points` points, between `coef(results)[what] - n_sd * stderror(results)[what]` and `coef(results)[what] + n_sd * stderror(results)[what]`.

Unless `at` is specified, slices will be computed at `coef(results)`.

"""
function slice(results::MEstimation_results,
               what::Int64;
               grid::Vector{Float64} = Vector{Float64}(),
               at::Vector{Float64} = Vector{Float64}(),
               n_points::Int64 = 50,
               n_sd::Real = 2.0)
    data = results.data
    theta = length(at) > 0 ? at : coef(results)
    n_par = length(theta)
    model_template = results.template
    br = results.br
    if length(grid) == 0
        sd = stderror(results)[what]
        co = coef(results)[what]
        step = 2 * n_sd * sd / n_points
        grid = range(co - n_sd * sd, co + n_sd * sd, length = n_points)
    end
    if results.has_regularizer
        regularizer = results.regularizer    
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
    slice_m = map(function(b) th = theta[1:n_par]; th[what] = b; slice_function(th) end, grid)
    Dict("value" => grid, "slice" => slice_m)
end
