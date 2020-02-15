"""   
    fit(template::objective_function_template, data::Any, theta::Vector; estimation_method::String = "M", br_method::String = "implicit_trace", optim_method = LBFGS(), optim_options = Optim.Options())

Fit an [`objective_function_template`](@ref) on `data` using M-estimation ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `estimation_method = "M"`; default) or RBM-estimation (reduced-bias M estimation; [Kosmidis & Lunardon, 2020](http://arxiv.org/abs/2001.03786); [keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `estimation_method = "RBM"`). Bias reduction is either through the maximization of the bias-reducing penalized objective in Kosmidis & Lunardon (2020) ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `br_method = "implicit_trace"`; default) or by subtracting an estimate of the bias from the M-estimates ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `br_method = "explicit_trace"`). The bias-reducing penalty is constructed internally using automatic differentiation (using the [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) package), and the bias estimate using a combination of automatic differentiation (using the [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) package) and numerical differentiation (using the [FiniteDiff](https://github.com/JuliaDiff/FiniteDiff.jl) package).

The maximization of the objective or the penalized objective is done using the [**Optim**](https://github.com/JuliaNLSolvers/Optim.jl) package. Optimization methods and options can be supplied directly through the [keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `optim_method` and `optim_options`, respectively. `optim_options` expects an object of class `Optim.Options`. See the [Optim documentation](https://julianlsolvers.github.io/Optim.jl/stable/#user/config/#general-options) for more details on the available options.
"""
function fit(template::objective_function_template,
             data::Any,
             theta::Vector;
             estimation_method::String = "M",
             br_method::String = "implicit_trace",
             optim_method = LBFGS(),
             optim_options = Optim.Options())
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
    ## bias reduction method to objetive_function
    obj = beta -> -objective_function(beta, data, template, br)
    out = optimize(obj, theta, optim_method, optim_options)
    if (estimation_method == "M")
        theta = out.minimizer
    elseif (estimation_method == "RBM") 
        if (br_method == "implicit_trace") 
            theta = out.minimizer
        elseif (br_method == "explicit_trace")
            quants = obj_quantities(out.minimizer, data, template, true)
            jmat_inv = quants[2]
            ## We use finite differences to get the adjustment
            adjustment = FiniteDiff.finite_difference_gradient(beta -> obj_quantities(beta, data, template, true)[1], out.minimizer)
            theta = out.minimizer + jmat_inv * adjustment
            ## Reset br
            br = true
        end
    end
    GEEBRA_results(out, theta, data, template, br, true, br_method)
end

"""   
    fit(template::estimating_function_template, data::Any, theta::Vector; estimation_method::String = "M", br_method::String = "implicit_trace", nlsolve_arguments...)

Fit an [`estimating_function_template`](@ref) on `data` using M-estimation ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `estimation_method = "M"`; default) or RBM-estimation (reduced-bias M estimation; [Kosmidis & Lunardon, 2020](http://arxiv.org/abs/2001.03786); [keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `estimation_method = "RBM"`). Bias reduction is either through the solution of the empirically adjusted estimating functions in Kosmidis & Lunardon (2020) ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `br_method = "implicit_trace"`; default) or by subtracting an estimate of the bias from the M-estimates ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `br_method = "explicit_trace"`). The bias-reducing adjustments and the bias estimate are constructed internally using automatic differentiation (using the [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) package). Bias reduction for only a subset of parameters can be performed by setting the ([keyword argument](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1) `concentrate` to the vector of the indices for those parameters.

The solution of the estimating equations or the adjusted estimating equations is done using the [**NLsolve**](https://github.com/JuliaNLSolvers/NLsolve.jl) package. Arguments can be passed directly to `NLsolve.nlsolve` through [keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments-1). See the [NLsolve README](https://github.com/JuliaNLSolvers/NLsolve.jl) for more information on available options.
"""
function fit(template::estimating_function_template,
             data::Any,
             theta::Vector;
             estimation_method::String = "M",
             br_method::String = "implicit_trace",
             concentrate::Vector{Int64} = Vector{Int64}(),
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
    ef = get_estimating_function(data, template, br, concentrate)   
    out = nlsolve(ef, theta; nlsolve_arguments...)
    if (estimation_method == "M")
        theta = out.zero
    elseif (estimation_method == "RBM") 
        if (br_method == "implicit_trace") 
            theta = out.zero
        elseif (br_method == "explicit_trace")
            quants = ef_quantities(out.zero, data, template, true, concentrate)
            adjustment = quants[1]
            jmat_inv = quants[2]
            theta = out.zero + jmat_inv * adjustment
            ## Reset br
            br = true
        end
    end
    GEEBRA_results(out, theta, data, template, br, false, br_method)
end

