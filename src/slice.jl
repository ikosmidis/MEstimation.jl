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
+ `results`: An [`MEstimation_results`](@ref) object.
+ `what`: the index of the parameter for which to compute a slice for

[Keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1)
===
+ `grid`: a `Vector{Float64}`; if supplied, the slice is computed at each element of `grid`. The default will result in the automatic calculation of the grid; see Details.
+ `at`: a `Vector{Float64}` of the sample length as `coef(results)`, specifying the parameter values at which to compute the slice. The default results in computing the slice at `coef(results)`.
+ `n_points`: an `Int64` specifying the number of grid points to generate. Applicable only if `grid` is not supplied; see Details.
+ `n_sd`: an `Int64` specifying the number of standard errors to be used for the grid generation. Applicable only if `grid` is not supplied; see Details.

Result
===
A [`Dict`](https://docs.julialang.org/en/v1/base/collections/#Dictionaries-1) with keys "grid" and "slice", holding `grid` and the values of the slice at `grid`, respectively.

Details
===
The default value of `grid` will result in the automatic calculation of a grid of `n_points` points, between `coef(results)[what] - n_sd * stderror(results)[what]` and `coef(results)[what] + n_sd * stderror(results)[what]`.
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
    Dict("grid" => grid, "slice" => slice_m)
end
