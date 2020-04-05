## Logistic regression

module logistic_regression

using Distributions

struct data
    y::Vector
    x::Array{Float64}
    m::Vector
end

## Logistic regression nobs
function nobs(data::data)
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

function loglik(theta::Vector,
                data::data,
                i::Int64)
    eta = sum(data.x[i, :] .* theta)
    mu = exp.(eta)./(1 .+ exp.(eta))
    ll = data.y[i] .* log.(mu) + (data.m[i] - data.y[i]) .* log.(1 .- mu)
    ll
end

function ef(theta::Vector,
            data::data,
            i::Int64)
    eta = sum(data.x[i, :] .* theta)
    mu = exp.(eta)./(1 .+ exp.(eta))
    data.x[i, :] * (data.y[i] - data.m[i] * mu)
end

function simulate(theta::Vector,
                  x::Matrix{Float64},
                  m::Vector{Int64})
    n = size(x)[2]
    y = rand.(Binomial.(m, cdf.(Logistic(), x * theta)))
    data(y, x, m);
end

end
