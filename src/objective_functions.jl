"""
    objective_function_template(nobs::Function, ef_contribution::Function)

Define an `objective_function_template` by supplying:
+ `nobs`: a function of `data` that computes the number of observations of the particular data type,
+ `ef_contribution`: a function of the parameters `theta`, the `data` and the observation index `i` that returns a real.
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


"""   
    fit(template::objective_function_template, data::Any, theta::Vector, br::Bool = false; method = LBFGS(), optim_Options = Optim.Options())

Fit an [`objective_function_template`](@ref) on `data` with (`br = true`) or without (`br = false`) bias reduction. Bias reduction is through the maximization of the bias-reducing penalized objective in Kosmidis & Lunardon (2020). The bias-reducing penalty is constructed internally using automatic differentiation (using the [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) package).

The maximization of the objective or the penalized objective is done using the [**Optim**](https://github.com/JuliaNLSolvers/Optim.jl) package. Optimization methods and options can be supplied directly through the [keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `method` and `optim.Options`, respectively.
"""
function fit(template::objective_function_template,
             data::Any,
             theta::Vector;
             estimation_method::String = "M",
             br_method::String = "implicit_trace",
             nlsolve_arguments...)

    if (estimation_method == "M")
        br = false
    elseif (estimation_method == "RBM")
        if (br_method == "implicit_trace")
            br = true
        elseif (br_method == "explicit_trace")
            br = false
        else
            error(br_method, " is not a recognized bias-reduction method")
        end
    else
        error(estimation_method, " is not a recognized estimation method")
    end

    ## down the line when det is implemented we need to be passing the
    ## bias reduction method to estimating_function

    obj = beta -> -objective_function(beta, data, template, br)
    out = optimize(obj, theta, method, optim_options)
    
    out = nlsolve(ef, theta; nlsolve_arguments...)

    if (estimation_method == "M")
        theta = out.zero
    elseif (estimation_method == "RBM") 
        if (br_method == "implicit_trace") 
            theta = out.zero
        elseif (br_method == "explicit_trace")
            quants = ef_quantities(out.zero, data, template, true)
            adjustment = quants[1]
            jmat_inv = quants[2]
            theta = out.zero + jmat_inv * adjustment
            ## Reset br
            br = true
        end
    end
    GEEBRA_results(out, theta, data, template, br, false, br_method)
end
# function fit(template::objective_function_template,
#              data::Any,
#              theta::Vector,
#              br::Bool = false;
#              method = LBFGS(),
#              optim_options = Optim.Options())
#     obj = beta -> -objective_function(beta, data, template, br)
#     out = optimize(obj, theta, method, optim_options)
#     GEEBRA_results(out, out.minimizer, data, template, br, true)
# end

function obj_quantities(theta::Vector,
                        data::Any,
                        template::objective_function_template,
                        penalty::Bool = false)
    npsi(eta::Vector, i::Int) = ForwardDiff.gradient(beta -> template.obj_contribution(beta, data, i), eta)
    nj(eta::Vector, i::Int) = ForwardDiff.hessian(beta -> template.obj_contribution(beta, data, i), eta)
    p = length(theta)
    n_obs = template.nobs(data)
    psi = Matrix(undef, n_obs, p)
    njmats = Vector(undef, n_obs)
    for i in 1:n_obs
        psi[i, :] =  npsi(theta, i)
        njmats[i] = nj(theta, i)
    end
    jmat_inv = inv(-sum(njmats))
    emat = psi' * psi
    emat = convert(Array{Float64, 2}, emat)
    vcov = jmat_inv * (emat * jmat_inv)
    if (penalty)        
        penalty = - tr(jmat_inv * emat) / 2
        [penalty, jmat_inv, emat]
    else
        [vcov, jmat_inv, emat]
    end
end
