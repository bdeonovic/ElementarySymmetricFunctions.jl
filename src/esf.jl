function esf_sum!(S::AbstractArray{T,1}, x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  fill!(S,zero(T))
  S[1] = one(T)
  @inbounds for col in 1:n
    for r in 1:col
        row = col - r + 1 
        S[row+1] = x[col] * S[row] + S[row+1]
    end
  end
end

"""
    esf_sum(x)

Compute the elementary symmetric functions of order k = 1, ..., n
where n = length(x)

# Examples
```julia-repl
julia> esf_sum([3.5118, .6219, .2905, .8450, 1.8648])
6-element Array{Float64,1}:
  1.0
  7.134
 16.9493
 16.7781
  7.05289
 0.999736
```
"""
function esf_sum(x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  S = Vector{T}(undef,n+1)
  esf_sum!(S,x)
  return S
end

function esf_sum_reg!(S::AbstractArray{T,1}, x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  fill!(S,zero(T))
  S[1] = one(T)
  @inbounds for col in 1:n
    for r in 1:col
        row = col - r + 1 
        S[row+1] = ((col-row)/col) * S[row+1] + (row/col) * x[col] * S[row]
    end
  end
end
"""
    esf_sum_reg(x)

Compute the elementary symmetric functions of order k = 1, ..., n
where n = length(x). Values are computed regularized by the binomial
coefficient binomial(n, k) to prevent over/under-flow.

# Examples
```julia-repl
julia> esf_sum_reg([3.5118, .6219, .2905, .8450, 1.8648])
6-element Array{Float64,1}:
  1.0
  1.4268
  1.69493
  1.67781
  1.41058
  0.999736
```
"""
function esf_sum_reg(x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  S = Vector{T}(undef,n+1)
  esf_sum_reg!(S,x)
  return S
end

#Regularized summation algorithm where one input is zeroed out (for computing derivatives)
function esf_sum_reg2!(S::AbstractArray{T,1}, x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  fill!(S,zero(T))
  S[1] = one(T)
  adj = 0
  @inbounds for col in 1:n
    if x[col] == 0.0
      adj += 1
      continue 
    end
    for r in 1:(col-adj)
      row = (col-adj) - r + 1
      S[row+1] = (col-adj-row)/(col-adj+1) * S[row+1] + (row+1)/(col-adj+1) * x[col] * S[row]
    end
    S[1] *= (col-adj)/(col-adj+1)
  end
end

#Regularized summation algorithm where two inputs are zeroed out (for computing derivatives)
function esf_sum_reg3!(S::AbstractArray{T,1}, x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  fill!(S,zero(T))
  S[1] = one(T)
  adj = 0
  @inbounds for col in 1:n
    if x[col] == 0.0
      adj += 1
      continue 
    end
    for r in 1:(col-adj)
      row = (col-adj) - r + 1
      S[row+1] = (col-adj-row)/(col-adj+2) * S[row+1] + (row+2)/(col-adj+2) * x[col] * S[row]
    end
    S[1] *= (col-adj) / (col-adj+2)
  end
end

function esf_sum_dervs_1(x::AbstractVector{T}) where T <: Real
  n = length(x)
  S = Vector{T}(undef,n+1)
  P = Array{T,2}(undef,n,n+1)
  esf_sum!(S, x)
  esf_sum_dervs_1!(P, x)
  return S, P
end

function esf_sum_dervs_1!(P::AbstractArray{T,2}, x::AbstractVector{T}) where T <: Real
  n = length(x)
  xj=zero(T)
  @inbounds for j in 1:n
    xj = x[j]
    x[j] = zero(T)
    @views esf_sum!(P[j,:], x)
    x[j] = xj
  end
end
function esf_sum_dervs_1_reg(x::AbstractVector{T}) where T <: Real
  n = length(x)
  S = Vector{T}(undef,n+1)
  P = Array{T,2}(undef,n,n+1)
  esf_sum_reg!(S, x)
  esf_sum_dervs_1_reg!(P, x)
  return S, P
end

function esf_sum_dervs_1_reg!(P::AbstractArray{T,2}, x::AbstractVector{T}) where T <: Real
  n = length(x)
  xj=zero(T)
  @inbounds for j in 1:n
    xj = x[j]
    x[j] = zero(T)
    @views esf_sum_reg2!(P[j,:], x)
    x[j] = xj
  end
end
function esf_sum_dervs_2(x::AbstractVector{T}) where T <: Real
  n = length(x)
  S = Vector{T}(undef,n+1)
  H = Array{T,3}(undef,n,n,n+1)
  esf_sum!(S, x)
  esf_sum_dervs_2!(H, x)
  return S, H
end

function esf_sum_dervs_2!(H::AbstractArray{T,3}, x::AbstractVector{T}) where T <: Real
  n = length(x)
  xj=zero(T)
  @inbounds for j in 1:n
    xj = x[j]
    x[j] = zero(T)
    for k in j:n
      xk = x[k]
      x[k] = zero(T)
      @views esf_sum!(H[j,k,:], x)
      H[k,j,:] .= H[j,k,:]
      x[k] = xk
    end
    x[j] = xj
  end
end
function esf_sum_dervs_2_reg(x::AbstractVector{T}) where T <: Real
  n = length(x)
  S = Vector{T}(undef,n+1)
  H = Array{T,3}(undef,n,n,n+1)
  esf_sum_reg!(S, x)
  esf_sum_dervs_2_reg!(H, x)
  return S, H
end

function esf_sum_dervs_2_reg!(H::AbstractArray{T,3}, x::AbstractVector{T}) where T <: Real
  n = length(x)
  xj=zero(T)
  @inbounds for j in 1:n
    xj = x[j]
    x[j] = zero(T)
    @views esf_sum_reg2!(H[j,j,:], x)
    for k in j+1:n
      xk = x[k]
      x[k] = zero(T)
      @views esf_sum_reg3!(H[j,k,:], x)
      H[k,j,:] .= H[j,k,:]
      x[k] = xk
    end
    x[j] = xj
  end
end
