using Test

## Ratios
@testset "implementation for a single parameter" begin
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
    ratio_template = geebra_ef_template(ratio_nobs, ratio_ef)
    @inferred geebra_ef_template(ratio_nobs, ratio_ef)
    
    ## Generate some data
    Random.seed!(123);
    my_data = ratio_data(randn(10), rand(10));

    ## Get M-estimator for the ratio
    result_m = solve_estimating_equation([0.1], my_data, ratio_template, false)
    @inferred solve_estimating_equation([0.1], my_data, ratio_template, false)
    ## Get reduced-bias estimator for the ratio
    result_br = solve_estimating_equation([0.1], my_data, ratio_template, true)
    @inferred solve_estimating_equation([0.1], my_data, ratio_template, true)

    ## Quantities for estimators
    sx = sum(my_data.x)
    sxx = sum(my_data.x .* my_data.x)
    sy = sum(my_data.y)
    sxy = sum(my_data.x .* my_data.y)

    @test isapprox(sy/sx, result_m.zero[1])
    @test isapprox((sy + sxy/sx)/(sx + sxx/sx), result_br.zero[1])
end


## Instrumental variables
@testset "implementation for multiple parameters" begin
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

    ## Set up IV geebra template
    iv_template = geebra_ef_template(iv_nobs, iv_ef)
    @inferred geebra_ef_template(iv_nobs, iv_ef)

    ## Simulate data
    true_theta = [2.0, 2.0, 1.0, 3.0, 0.0, 1.0, 2.0, 1.0, 1.0]
    true_parameter = true_theta[[3, 2]]
    
    Random.seed!(123)
    my_data = simulate_iv(100, true_theta)
    
    o1_ml = solve_estimating_equation(true_parameter, my_data, iv_template, false)
    @inferred solve_estimating_equation(true_parameter, my_data, iv_template, false)
    o1_br = solve_estimating_equation(true_parameter, my_data, iv_template, true)
    @inferred solve_estimating_equation(true_parameter, my_data, iv_template, true)
    
    ef_br = get_estimating_function(my_data, iv_template, true)
    @inferred get_estimating_function(my_data, iv_template, true)
    ef_ml = get_estimating_function(my_data, iv_template, false)
    @inferred get_estimating_function(my_data, iv_template, false)
    o2_ml = nlsolve(ef_ml, [0.1, 0.2])
    o2_br = nlsolve(ef_br, [0.1, 0.2])

    @test isapprox(o1_ml.zero, o2_ml.zero)
    @test isapprox(o1_br.zero, o2_br.zero)

    ## Estimating function at the estimates
    @test isapprox(estimating_function(o2_br.zero, my_data, iv_template, true),
                   zeros(Float64, 2, 1),
                   atol = 1e-10)
    
end
