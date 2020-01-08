using Test

## Ratios
@testset "ef implementation for a single parameter" begin
    using geebra
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
    result_m = geebra.fit([0.1], my_data, ratio_template, false)
    @inferred geebra.fit([0.1], my_data, ratio_template, false)
    ## Get reduced-bias estimator for the ratio
    result_br = geebra.fit([0.1], my_data, ratio_template, true)
    @inferred geebra.fit([0.1], my_data, ratio_template, true)

    ## Quantities for estimators
    sx = sum(my_data.x)
    sxx = sum(my_data.x .* my_data.x)
    sy = sum(my_data.y)
    sxy = sum(my_data.x .* my_data.y)

    @test isapprox(sy/sx, result_m.theta[1])
    @test isapprox((sy + sxy/sx)/(sx + sxx/sx), result_br.theta[1])
end


## Instrumental variables
@testset "ef implementation for multiple parameters" begin
    using geebra
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

    ## Set up IV geebra template
    iv_template = estimating_function_template(iv_nobs, iv_ef)
    @inferred estimating_function_template(iv_nobs, iv_ef)

    ## Simulate data
    true_theta = [2.0, 2.0, 1.0, 3.0, 0.0, 1.0, 2.0, 1.0, 1.0]
    true_parameter = true_theta[[3, 2]]
    
    Random.seed!(123)
    my_data = simulate_iv(100, true_theta)
    
    o1_ml = geebra.fit(true_parameter, my_data, iv_template, false)
    @inferred geebra.fit(true_parameter, my_data, iv_template, false)
    o1_br = geebra.fit(true_parameter, my_data, iv_template, true)
    @inferred geebra.fit(true_parameter, my_data, iv_template, true)
    
    ef_br = get_estimating_function(my_data, iv_template, true)
    @inferred get_estimating_function(my_data, iv_template, true)
    ef_ml = get_estimating_function(my_data, iv_template, false)
    @inferred get_estimating_function(my_data, iv_template, false)
    o2_ml = nlsolve(ef_ml, [0.1, 0.2])
    o2_br = nlsolve(ef_br, [0.1, 0.2])

    @test isapprox(o1_ml.theta, o2_ml.zero)
    @test isapprox(o1_br.theta, o2_br.zero)

    ## Estimating function at the estimates
    @test isapprox(estimating_function(o2_br.zero, my_data, iv_template, true),
                   zeros(Float64, 2, 1),
                   atol = 1e-10)
end


@testset "obj implementation for multiple parameters" begin
    using geebra
    using Random
    using Distributions
    using Optim
    
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

    o2_ml = geebra.fit(true_betas, my_data, logistic_template, false)
    o2_br = geebra.fit(true_betas, my_data, logistic_template, true)  

    o3_ml = optimize(b -> -objective_function(b, my_data, logistic_template, false),
                     true_betas, Optim.Options(iterations = 2))
    o3_br = optimize(b -> -objective_function(b, my_data, logistic_template, true),
                     true_betas, Optim.Options(iterations = 2))

    o4_ml = geebra.fit(true_betas, my_data, logistic_template, false,
                               method = NelderMead(),
                               optim_options = Optim.Options(iterations = 2))
    o4_br = geebra.fit(true_betas, my_data, logistic_template, true,
                               method = NelderMead(),
                               optim_options = Optim.Options(iterations = 2))
    
    @test isapprox(Optim.minimizer(o1_ml), Optim.minimizer(o2_ml.results))
    @test isapprox(Optim.minimizer(o1_br), Optim.minimizer(o2_br.results))
    @test isapprox(Optim.minimizer(o3_ml), Optim.minimizer(o4_ml.results))
    @test isapprox(Optim.minimizer(o3_br), Optim.minimizer(o4_br.results))
   
end

@testset "agreement between obj and ef implementations" begin
    using geebra
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
    x = Array{Float64}(undef, n, 2);
    x[:, 1] .= 1.0;
    x[:, 2] .= rand(n);
    true_betas = [0.5, -1];
    y = rand.(Binomial.(m, cdf.(Logistic(), x * true_betas)));
    my_data = logistic_data(y, x, fill(m, n));

    logistic_obj_template = objective_function_template(logistic_nobs, logistic_loglik)
    logistic_ef_template = estimating_function_template(logistic_nobs, logistic_ef)
    
    o1_ml = geebra.fit(true_betas, my_data, logistic_obj_template, false)
    e1_ml = geebra.fit(true_betas, my_data, logistic_ef_template, false)
    o1_br = geebra.fit(true_betas, my_data, logistic_obj_template, true)
    e1_br = geebra.fit(true_betas, my_data, logistic_ef_template, true)

    @test isapprox(o1_ml.theta, e1_ml.theta)
    @test isapprox(o1_br.theta, e1_br.theta)   
    
end
