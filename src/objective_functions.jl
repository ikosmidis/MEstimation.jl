"""
    objective_function_template(nobs::Function, 
                                obj_contribution::Function)

A constructor of objects of [composite type](https://docs.julialang.org/en/v1/manual/types/#Composite-Types-1) for defining an `objective_function_template`.

Arguments
===
+ `nobs`: a function of `data` that computes the number of observations of the particular data type,
+ `obj_contribution`: a function of the parameters `theta`, the `data` and the observation index `i` that returns a `Float64`.

Result
===
An `objective_function_template` object with fields `nobs` and `obj_contributions`.

"""
struct objective_function_template
    nobs::Function
    obj_contribution::Function
end


"""   
    objective_function(theta::Vector{Float64}, 
                       data::Any, 
                       template::objective_function_template, 
                       br::Bool = false)

Evaluates the objective function at `theta` by adding up all contributions in 
`data`, according to [`objective_function_template`](@ref).

Arguments
===
+ `theta`: a `Vector{Float64}` of parameter values at which to evaluate the objective function
+ `data`: typically an object of [composite type]((https://docs.julialang.org/en/v1/manual/types/#Composite-Types-1)) with all the data required to compute the `objective_function`.
+ `template`: an [`objective_function_template`](@ref) object
+ `br`: a `Bool`. If `false` (default), the objective function is constructed by adding up all contributions in 
`data`, according to [`objective_function_template`](@ref), before it is evaluated at `theta`. If `true` then the bias-reducing penalty in [Kosmidis & Lunardon, 2020](http://arxiv.org/abs/2001.03786) is computed and added to the objective function.

Result
===
A `Float64`.

Details
===
`data` can be used to pass additional constants other than the actual data to the objective. 
"""
function objective_function(theta::Vector{Float64},
                            data::Any,
                            template::objective_function_template,
                            br::Bool = false)
    p = length(theta)
    n_obs = template.nobs(data)
    objective = 0
    for i in 1:n_obs
        objective += template.obj_contribution(theta, data, i)
    end
    if (br)
        quants = obj_quantities(theta, data, template, true)
        objective + quants[1]
    else
        objective
    end
end


function obj_quantities(theta::Vector,
                        data::Any,
                        template::objective_function_template,
                        penalty::Bool = false)
    objective_i = (pars, i) -> template.obj_contribution(pars, data, i)
    results_i = (x, i) -> begin
        out = DiffResults.HessianResult(x)
        ForwardDiff.hessian!(out, pars -> objective_i(pars, i), x)
    end

    p = length(theta)
    n_obs = template.nobs(data)
    psi = zeros(p)
    emat = zeros(p, p)
    jmat = zeros(p, p)
    for i in 1:n_obs
        cdiffres = results_i(theta, i)
        cpsi = DiffResults.gradient(cdiffres)
        psi += cpsi
        emat += cpsi * cpsi'
        jmat += - DiffResults.hessian(cdiffres)
    end
    jmat_inv = try
        inv(jmat)
    catch
        fill(NaN, p, p)
    end
    vcov = jmat_inv * (emat * jmat_inv)
    if (penalty)        
        br_penalty = - tr(jmat_inv * emat) / 2
        # br_penalty = n_obs * log(det(Matrix{Float64}(I * n_obs, p, p) -
        #                              jmat_inv * emat)) / 2
        # br_penalty = + log(det(sum(njmats))) / 2 - log(det(emat)) / 2
        [br_penalty, jmat_inv, emat, psi]
    else
        [vcov, jmat_inv, emat, psi]
    end
end


"""
    estimating_function_template(object::objective_function_template)

Construct an [`estimating_function_template`](@ref) from an [`objective_function_template`](@ref).

Arguments
===
+ `object`: an [`objective_function_template`](@ref)

Details
===
The field `ef_contribution` of the result is computed by differentiating (using the [**ForwardDiff**](https://github.com/JuliaDiff/ForwardDiff.jl) package) `object.obj_contribution` with respect to `theta`.

Result
===
A [`estimating_function_template`](@ref) object.

"""
function estimating_function_template(object::objective_function_template)
    function ef_contribution(theta::Vector, data::Any, i::Int64)
        out = similar(theta)
        ForwardDiff.gradient!(out, b -> object.obj_contribution(b, data, i), theta)
    end
    estimating_function_template(object.nobs, ef_contribution)
end
