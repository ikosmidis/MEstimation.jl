module MEstimation # general estimating equations with or without bias-reducting adjustments

using NLsolve
using Optim
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using Distributions
# using InvertedIndices

import Base: show, print
import StatsBase: fit, aic, vcov, coef, coeftable, stderror, CoefTable

export objective_function
export objective_function_template

export estimating_function
export get_estimating_function
export estimating_function_template

export aic
export tic
export vcov
export coef
export coeftable
export fit
export stderror

export slice
# export profile

include("estimating_functions.jl")
include("objective_functions.jl")
include("fit.jl")
include("result_methods.jl")
include("slice.jl")
# include("profile.jl")

end # module
