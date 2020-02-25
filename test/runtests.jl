using Test

## Ratios
@testset "ef implementation for a single parameter" begin
    using GEEBRA
    using Random
    
    ## Ratio data
    struct ratio_data
        y::Vector
        x::Vector
    end

    ## Ratio nobs
    function ratio_nobs(data::ratio_data)
        nx = length(data.x)
        ny = length(data.y)
        if (nx != ny) 
            error("length of x is not equal to the length of y")
        end
        nx
    end

    ## Ratio contribution to estimating function
    function ratio_ef(theta::Vector,
                      data::ratio_data,
                      i::Int64)
        p = length(theta)
        (data.y[i] .- theta * data.x[i])[1:p]
    end

    ## Set the ratio template
    ratio_template = estimating_function_template(ratio_nobs, ratio_ef)
    @inferred estimating_function_template(ratio_nobs, ratio_ef)
    
    ## Generate some data
    Random.seed!(123);
    my_data = ratio_data(randn(10), rand(10));

    ## Get M-estimator for the ratio
    result_m = fit(ratio_template, my_data, [0.1], estimation_method = "M")
    @inferred fit(ratio_template, my_data, [0.1], estimation_method = "M")
    ## Get reduced-bias estimator for the ratio
    result_br = fit(ratio_template, my_data, [0.1], estimation_method = "RBM")
    @inferred fit(ratio_template, my_data, [0.1], estimation_method = "RBM")
    ## Gere reduced-bias estimator for the ration using explicit RBM-estimation
    result_br1 = fit(ratio_template, my_data, [0.1], estimation_method = "RBM", br_method = "explicit_trace")

    ## Quantities for estimators
    sx = sum(my_data.x)
    sxx = sum(my_data.x .* my_data.x)
    sy = sum(my_data.y)
    sxy = sum(my_data.x .* my_data.y)

    @test isapprox(sy/sx, coef(result_m)[1])
    @test isapprox((sy + sxy/sx)/(sx + sxx/sx), coef(result_br)[1])
    @test isapprox(sy/sx * (1 - sxx / sx^2) + sxy/sx^2, coef(result_br1)[1])
    @test_throws ErrorException  result_br1 = fit(ratio_template, my_data, [0.1], estimation_method = "RBM", br_method = "magic_br_method")
end


## Instrumental variables
@testset "ef implementation for multiple parameters" begin
    using GEEBRA
    using Random
    using Distributions
    using NLsolve
    
    ## IV data
    struct iv_data
        y::Vector
        t::Vector
        w::Vector
    end

    ## IV nobs   
    function iv_nobs(data::iv_data) 
        nw = length(data.w)
        ny = length(data.y)
        nt = length(data.t)
        if (nw != ny) 
            error("length of w is not equal to the length of y")
        elseif (nw != nt)
            error("length of w is not equal to the length of t")
        elseif (nw != nt)
            error("length of w is not equal to the length of t")
        end
        nw
    end

    ## Contributions to the estimating functions
    function iv_ef(theta::Vector,
                   data::iv_data,
                   i::Int64)
        [theta[1] - data.t[i], (data.y[i] - theta[2] * data.w[i]) * (theta[1] - data.t[i])]
    end

    
    ## Simulating IV data
    function simulate_iv(nobs::Int,
                         theta::Vector) 
        alpha = theta[1]
        beta = theta[2]
        gamma = theta[3]
        delta = theta[4]
        mux = theta[5]
        sigmax = theta[6]
        sigmae = theta[7]
        sigmau = theta[8]
        sigmat = theta[9]
        e1 = rand(Normal(0, sigmae), nobs)
        e2 = rand(Normal(0, sigmau), nobs)
        e3 = rand(Normal(0, sigmat), nobs)
        x = rand(Normal(mux, sigmax), nobs)
        w = x + e2
        y = alpha .+ beta * x + e1
        t = gamma .+ delta * x + e3
        iv_data(y, t, w)
    end

    ## Set up IV GEEBRA template
    iv_template = estimating_function_template(iv_nobs, iv_ef)
    @inferred estimating_function_template(iv_nobs, iv_ef)

    ## Simulate data
    true_theta = [2.0, 2.0, 1.0, 3.0, 0.0, 1.0, 2.0, 1.0, 1.0]
    true_parameter = true_theta[[3, 2]]
    
    Random.seed!(123)
    my_data = simulate_iv(100, true_theta)
    
    o1_ml = fit(iv_template, my_data, true_parameter, estimation_method = "M")
    @inferred fit(iv_template, my_data, true_parameter, estimation_method = "M")
    o1_br = fit(iv_template, my_data, true_parameter, estimation_method = "RBM")
    @inferred fit(iv_template, my_data, true_parameter, estimation_method = "RBM")
    o1_br1 = fit(iv_template, my_data, true_parameter, estimation_method = "RBM", br_method = "explicit_trace")
    @inferred fit(iv_template, my_data, true_parameter, estimation_method = "RBM", br_method = "explicit_trace")
    
    ef_br = get_estimating_function(my_data, iv_template, true)
    @inferred get_estimating_function(my_data, iv_template, true)
    ef_ml = get_estimating_function(my_data, iv_template, false)
    @inferred get_estimating_function(my_data, iv_template, false)
    o2_ml = nlsolve(ef_ml, [0.1, 0.2])
    o2_br = nlsolve(ef_br, [0.1, 0.2])

    qs = GEEBRA.ef_quantities(coef(o1_ml), my_data, iv_template, true)
    @test isapprox(coef(o1_ml) + qs[2] * qs[1], coef(o1_br1))
    
    @test isapprox(coef(o1_ml), o2_ml.zero)
    @test isapprox(coef(o1_br), o2_br.zero)

    ## Estimating function at the estimates
    @test isapprox(estimating_function(o2_br.zero, my_data, iv_template, true),
                   zeros(Float64, 2, 1),
                   atol = 1e-10)

    
    function iv_ef(theta::Vector,
                   data::iv_data,
                   i::Int64)
        [theta[1] - data.t[i], (data.y[i] - theta[2] * data.w[i]) * (theta[1] - data.t[i])]
    end


    function iv_ef2(theta::Vector,
                   data::iv_data,
                   i::Int64)
        [(data.y[i] - theta[2] * data.w[i]) * (theta[1] - data.t[i]), theta[1] - data.t[i]]
    end

    function iv_ef3(theta::Vector,
                   data::iv_data,
                   i::Int64)
        [(data.y[i] - theta[1] * data.w[i]) * (theta[2] - data.t[i]), theta[2] - data.t[i]]
    end

    
    ## Swapping the order of the estimating functions gives the same results
    iv_template2 = estimating_function_template(iv_nobs, iv_ef2)
    iv_template3 = estimating_function_template(iv_nobs, iv_ef3)

    coef(fit(iv_template, my_data, true_parameter, estimation_method = "M"))
    coef(fit(iv_template2, my_data, true_parameter, estimation_method = "M"))
    coef(fit(iv_template3, my_data, true_parameter, estimation_method = "M"))

    coef(fit(iv_template, my_data, true_parameter, estimation_method = "RBM"))
    coef(fit(iv_template2, my_data, true_parameter, estimation_method = "RBM"))
    coef(fit(iv_template3, my_data, true_parameter, estimation_method = "RBM"))

    GEEBRA.ef_quantities(true_parameter, my_data, iv_template, true)[1]
    GEEBRA.ef_quantities(true_parameter, my_data, iv_template2, true)[1]
    GEEBRA.ef_quantities(true_parameter[[2,1]], my_data, iv_template3, true)[1]
    
end


@testset "obj implementation for multiple parameters" begin
    using GEEBRA
    using Random
    using Distributions
    using Optim
    using LinearAlgebra
    
    ## Logistic regression data
    struct logistic_data
        y::Vector
        x::Array{Float64}
        m::Vector
    end

    ## Logistic regression nobs
    function logistic_nobs(data::logistic_data)
        nx = size(data.x)[1]
        ny = length(data.y)
        nm = length(data.m)
        if (nx != ny) 
            error("number of rows in of x is not equal to the length of y")
        elseif (nx != nm)
            error("number of rows in of x is not equal to the length of m")
        elseif (ny != nm)
            error("length of y is not equal to the length of m")
        end
        nx
    end

    function logistic_loglik(theta::Vector,
                             data::logistic_data,
                             i::Int64)
        eta = sum(data.x[i, :] .* theta)
        mu = exp.(eta)./(1 .+ exp.(eta))
        ll = data.y[i] .* log.(mu) + (data.m[i] - data.y[i]) .* log.(1 .- mu)
        ll
    end

    Random.seed!(123);
    n = 100;
    m = 1;
    x = Array{Float64}(undef, n, 2);
    x[:, 1] .= 1.0;
    x[:, 2] .= rand(n);
    true_betas = [0.5, -1];
    y = rand.(Binomial.(m, cdf.(Logistic(), x * true_betas)));
    my_data = logistic_data(y, x, fill(m, n));

    logistic_template = objective_function_template(logistic_nobs, logistic_loglik)
    @inferred objective_function_template(logistic_nobs, logistic_loglik)

    o1_ml = optimize(b -> -objective_function(b, my_data, logistic_template, false),
                     true_betas, LBFGS())
    o1_br = optimize(b -> -objective_function(b, my_data, logistic_template, true),
                     true_betas, LBFGS())

    o2_ml = fit(logistic_template, my_data, true_betas, estimation_method = "M")
    o2_br = fit(logistic_template, my_data, true_betas, estimation_method = "RBM")  

    o3_ml = optimize(b -> -objective_function(b, my_data, logistic_template, false),
                     true_betas, Optim.Options(iterations = 2))
    o3_br = optimize(b -> -objective_function(b, my_data, logistic_template, true),
                     true_betas, Optim.Options(iterations = 2))

    o4_ml = fit(logistic_template, my_data, true_betas, estimation_method = "M",
                               optim_method = NelderMead(),
                               optim_options = Optim.Options(iterations = 2))
    o4_br = fit(logistic_template, my_data, true_betas, estimation_method = "RBM",
                               optim_method = NelderMead(),
                               optim_options = Optim.Options(iterations = 2))
             
    @test isapprox(Optim.minimizer(o1_ml), Optim.minimizer(o2_ml.results))
    @test isapprox(Optim.minimizer(o1_br), Optim.minimizer(o2_br.results))
    @test isapprox(Optim.minimizer(o3_ml), Optim.minimizer(o4_ml.results))
    @test isapprox(Optim.minimizer(o3_br), Optim.minimizer(o4_br.results))

    @test isapprox(sqrt.(diag(vcov(o2_br))), stderror(o2_br))
    
end

@testset "agreement between obj and ef implementations" begin
    using GEEBRA
    using Random
    using Distributions
    using Optim
    using NLsolve
    
    ## Logisti regression data
    struct logistic_data
        y::Vector
        x::Array{Float64}
        m::Vector
    end

    ## Logistic regression nobs
    function logistic_nobs(data::logistic_data)
        nx = size(data.x)[1]
        ny = length(data.y)
        nm = length(data.m)
        if (nx != ny) 
            error("number of rows in of x is not equal to the length of y")
        elseif (nx != nm)
            error("number of rows in of x is not equal to the length of m")
        elseif (ny != nm)
            error("length of y is not equal to the length of m")
        end
        nx
    end

    function logistic_loglik(theta::Vector,
                             data::logistic_data,
                             i::Int64)
        eta = sum(data.x[i, :] .* theta)
        mu = exp.(eta)./(1 .+ exp.(eta))
        ll = data.y[i] .* log.(mu) + (data.m[i] - data.y[i]) .* log.(1 .- mu)
        ll
    end

    function logistic_ef(theta::Vector,
                         data::logistic_data,
                         i::Int64)
        eta = sum(data.x[i, :] .* theta)
        mu = exp.(eta)./(1 .+ exp.(eta))
        data.x[i, :] * (data.y[i] - data.m[i] * mu)
    end
   
    Random.seed!(123);
    n = 100;
    m = 1;
    p = 5;
    x = Array{Float64}(undef, n, p);
    x[:, 1] .= 1.0;
    for j in 2:p
        x[:, j] .= rand(n);
    end
    true_betas = randn(p) * sqrt(p);
    y = rand.(Binomial.(m, cdf.(Logistic(), x * true_betas)));
    my_data = logistic_data(y, x, fill(m, n));

    logistic_obj_template = objective_function_template(logistic_nobs, logistic_loglik)
    logistic_ef_template = estimating_function_template(logistic_nobs, logistic_ef)
    
    o1_ml = fit(logistic_obj_template, my_data, true_betas, estimation_method = "M")
    e1_ml = fit(logistic_ef_template, my_data, true_betas, estimation_method = "M")
    o1_br = fit(logistic_obj_template, my_data, true_betas, estimation_method = "RBM")
    e1_br = fit(logistic_ef_template, my_data, true_betas, estimation_method = "RBM")
    o1_br1 = fit(logistic_obj_template, my_data, true_betas, estimation_method = "RBM", br_method = "explicit_trace")
    o1_br2 = fit(logistic_obj_template, my_data, coef(o1_ml), estimation_method = "RBM", br_method = "explicit_trace")
    e1_br1 = fit(logistic_ef_template, my_data, true_betas, estimation_method = "RBM", br_method = "explicit_trace")
    e1_br2 = fit(logistic_ef_template, my_data, coef(o1_ml), estimation_method = "RBM", br_method = "explicit_trace")

    
    @test isapprox(coef(o1_ml), coef(e1_ml), atol = 1e-05)
    @test isapprox(coef(o1_br), coef(e1_br), atol = 1e-05)
    @test isapprox(coef(o1_br1), coef(e1_br1), atol = 1e-05)
    @test isapprox(coef(o1_br2), coef(e1_br2), atol = 1e-05)
    @test isapprox(coef(o1_br1), coef(e1_br2), atol = 1e-05)   

    @test isapprox(aic(o1_ml),
                   -2 * (objective_function(coef(o1_ml), my_data, logistic_obj_template, false) - p))

    @test isapprox(aic(o1_br),
                   -2 * (objective_function(coef(o1_br), my_data, logistic_obj_template) - p))

    quants_ml = GEEBRA.obj_quantities(coef(o1_ml), my_data, logistic_obj_template, true)
    quants_br = GEEBRA.obj_quantities(coef(o1_br), my_data, logistic_obj_template, true)

    @test isapprox(tic(o1_ml),
                   -2 * (objective_function(coef(o1_ml), my_data, logistic_obj_template) + 2 * quants_ml[1]))
    
    @test isapprox(tic(o1_br),
                   -2 * (objective_function(coef(o1_br), my_data, logistic_obj_template) + 2 * quants_br[1]))
    

    @test isapprox(vcov(o1_ml), vcov(e1_ml))
    @test isapprox(vcov(o1_br), vcov(e1_br))
    @test isapprox(vcov(o1_br1), vcov(e1_br1))
    @test isapprox(vcov(o1_br2), vcov(e1_br2))

    @test isapprox(coeftable(o1_ml).cols, coeftable(e1_ml).cols)
    @test isapprox(coeftable(o1_br).cols, coeftable(e1_br).cols)
    @test isapprox(coeftable(o1_br1).cols, coeftable(e1_br1).cols)
    @test isapprox(coeftable(o1_br2).cols, coeftable(e1_br2).cols)
   
    
end

# using Revise
# using Pkg
# Pkg.activate("/Users/yiannis/Repositories/GEEBRA.jl")
