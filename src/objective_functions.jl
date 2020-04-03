"""
    objective_function_template(nobs::Function, 
                                obj_contribution::Function)

Define an `objective_function_template` by supplying:
+ `nobs`: a function of `data` that computes the number of observations of the particular data type,
+ `obj_contribution`: a function of the parameters `theta`, the `data` and the observation index `i` that returns a real.
"""
struct objective_function_template
    nobs::Function
    obj_contribution::Function
end


"""   
    objective_function(theta::Vector, 
                       data::Any, 
                       template::objective_function_template, 
                       br::Bool = false)

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
    function npsi(eta::Vector, i::Int)
        out = similar(eta)
        ForwardDiff.gradient!(out, beta -> template.obj_contribution(beta, data, i), eta)
    end
    function nj(eta::Vector, i::Int)
        out = similar(eta, p, p)
        ForwardDiff.hessian!(out, beta -> template.obj_contribution(beta, data, i), eta)
    end
    p = length(theta)
    n_obs = template.nobs(data)
    psi = zeros(p)
    emat = zeros(p, p)
    jmat = zeros(p, p)
    for i in 1:n_obs
        cpsi = npsi(theta, i)
        psi += cpsi
        emat += cpsi * cpsi'
        jmat += -nj(theta, i)
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



function estimating_function_template(object::objective_function_template)
    function ef_contribution(theta::Vector, data::Any, i::Int64)
        out = similar(theta)
        ForwardDiff.gradient!(out, b -> object.obj_contribution(b, data, i), theta)
    end
    estimating_function_template(object.nobs, ef_contribution)
end
