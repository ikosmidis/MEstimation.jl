"""
    estimating_function_template(nobs::Function, ef_contribution::Function)

Define an `estimating_function_template` by supplying:
+ `nobs`: a function of `data` that computes the number of observations of the particular data type,
+ `ef_contribution`: a function of the parameters `theta`, the `data` and the observation index `i` that returns a vector of length `length(theta)`.
"""
struct estimating_function_template
    nobs::Function
    ef_contribution::Function
end

""" 
    estimating_function(theta::Vector, data::Any, template::estimating_function_template, br::Bool = false)

Construct the estimating function by adding up all contributions in the `data` according to [`estimating_function_template`](@ref), and evaluate it at `theta`. If `br = true` then automatic differentiation is used to compute the empirical bias-reducing adjustments and add them to the estimating function.
"""
function estimating_function(theta::Vector,
                             data::Any,
                             template::estimating_function_template,
                             br::Bool = false)
    p = length(theta)
    n_obs = template.nobs(data)
    contributions = Matrix(undef, p, n_obs)
    for i in 1:n_obs
        contributions[:, i] = template.ef_contribution(theta, data, i)
    end
    if (br)
        quants = ef_quantities(theta, data, template, br)
        sum(contributions, dims = 2) + quants[1]
    else
        sum(contributions, dims = 2)
    end
end

""" 
    get_estimating_function(data::Any, template::estimating_function_template, br::Bool = false)

Construct the estimating function by adding up all contributions in the `data` according to [`estimating_function_template`](@ref). If `br = true` then automatic differentiation is used to compute the empirical bias-reducing adjustments and add them to the estimating function. The result is a function that stores the estimating functions values at its second argument, in a preallocated vector passed as its first argument, ready to be used withing `NLsolve.nlsolve`.
"""
function get_estimating_function(data::Any,
                                 template::estimating_function_template,
                                 br::Bool = false)
    function g!(F, theta::Vector) 
        out = estimating_function(theta, data, template, br)
        for i in 1:length(out)
            F[i] = out[i]
        end
    end
end

"""   
    fit(template::estimating_function_template, data::Any, theta::Vector; estimation_method::String = "M", br_method::String = "implicit_trace", nlsolve_arguments...)

Fit an [`estimating_function_template`](@ref) on `data` using M-estimation ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `estimation_method = "M"`; default) or RBM-estimation (reduced-bias M estimation; [keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `estimation_method = "RBM"`). Bias reduction is either through the solution of the empirically adjusted estimating functions in Kosmidis & Lunardon (2020) ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `br_method = "implicit_trace"`; default) or by subtracting an estimate of the bias from the M-estimates ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `br_method = "explicit_trace"`). The bias-reducing adjustments and the bias estimate are constructed internally using automatic differentiation (using the [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) package).

The solution of the estimating equations or the adjusted estimating equations is done using the [**NLsolve**](https://github.com/JuliaNLSolvers/NLsolve.jl) package. Arguments can be passed directly to `NLsolve.nlsolve` through [keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1). See the [NLsolve README](https://github.com/JuliaNLSolvers/NLsolve.jl) for more information on available options.
"""
function fit(template::estimating_function_template,
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
    ef = get_estimating_function(data, template, br)   
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

function ef_quantities(theta::Vector,
                       data::Any,
                       template::estimating_function_template,
                       adjustment::Bool = false)
    nj(eta::Vector, i::Int) = ForwardDiff.jacobian(beta -> template.ef_contribution(beta, data, i), eta)
    p = length(theta)
    n_obs = template.nobs(data)
    psi = Matrix{Float64}(undef, n_obs, p)
    njmats = Vector(undef, n_obs)
    for i in 1:n_obs
        psi[i, :] =  template.ef_contribution(theta, data, i) 
        njmats[i] = nj(theta, i)
    end
    jmat_inv = inv(-sum(njmats))
    emat = psi' * psi
    vcov = jmat_inv * (emat * jmat_inv')
    # vcov = jmat_inv * psi'
    # vcov = vcov * vcov'
    if (adjustment)
        u(eta::Vector, i::Int) = ForwardDiff.jacobian(beta -> nj(beta, i), eta)
        psi_tilde = Matrix(undef, n_obs, p)
        umats = Vector(undef, n_obs)
        for i in 1:n_obs
            umats[i] = u(theta, i)
        end       
        umat = sum(umats)
        A = Vector(undef, p)
        for j in 1:p
            for i in 1:n_obs
                psi_tilde[i, :] = njmats[i][j, :]
            end
            A[j] = -tr(jmat_inv * psi_tilde' * psi +
                       vcov * umat[j:p:(p * p - p + j), :] / 2)
        end
        [A, jmat_inv, emat]
    else
        [vcov, jmat_inv, emat]
    end
end

