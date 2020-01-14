"""
    objective_function_template(nobs::Function, obj_contribution::Function)

Define an `objective_function_template` by supplying:
+ `nobs`: a function of `data` that computes the number of observations of the particular data type,
+ `obj_contribution`: a function of the parameters `theta`, the `data` and the observation index `i` that returns a real.
"""
struct objective_function_template
    nobs::Function
    obj_contribution::Function
end


"""   

    objective_function(theta::Vector, data::Any, template::objective_function_template, br::Bool = false)

Construct the objective function by adding up all contributions in the
`data` according to [`objective_function_template`](@ref), and
evaluate it at `theta`. If `br = true` then automatic differentiation
is used to compute the empirical bias-reducing penalty and add it to
the objective function.  
"""
function objective_function(theta::Vector,
                            data::Any,
                            template::objective_function_template,
                            br::Bool = false)
    p = length(theta)
    n_obs = template.nobs(data)
    contributions = Vector(undef, n_obs)
    for i in 1:n_obs
        contributions[i] = template.obj_contribution(theta, data, i)
    end
    if (br)
        quants = obj_quantities(theta, data, template, br)
        sum(contributions) + quants[1]
    else
        sum(contributions)
    end
end

function obj_quantities(theta::Vector,
                        data::Any,
                        template::objective_function_template,
                        penalty::Bool = false)
    npsi(eta::Vector, i::Int) = ForwardDiff.gradient(beta -> template.obj_contribution(beta, data, i), eta)
    nj(eta::Vector, i::Int) = ForwardDiff.hessian(beta -> template.obj_contribution(beta, data, i), eta)
    p = length(theta)
    n_obs = template.nobs(data)
    psi = Matrix{Float64}(undef, n_obs, p)
    njmats = Vector(undef, n_obs)
    for i in 1:n_obs
        psi[i, :] =  npsi(theta, i)
        njmats[i] = nj(theta, i)
    end
    jmat_inv = inv(-sum(njmats))
    emat = psi' * psi
    vcov = jmat_inv * (emat * jmat_inv)
    if (penalty)        
        penalty = - tr(jmat_inv * emat) / 2
        [penalty, jmat_inv, emat]
    else
        [vcov, jmat_inv, emat]
    end
end
