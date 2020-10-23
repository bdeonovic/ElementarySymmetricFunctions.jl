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

function esf_sum_log!(S::AbstractArray{T,1}, x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  fill!(S,-Inf)
  S[1] = zero(T)
  @inbounds for col in 1:n
    for r in 1:col
        row = col - r + 1 
        Sr = S[row] + log(x[col])
        Sr1 = S[row+1]
        if (Sr1 > Sr) && (Sr1 > zero(T))
          S[row+1] = Sr1 + log1p(exp(Sr - Sr1))
        elseif (Sr >= Sr1) && (Sr > zero(T))
          S[row+1] = Sr + log1p(exp(Sr1-Sr))
        else
          S[row+1] = log(exp(Sr1) + exp(Sr))
        end
    end
  end
end

function esf_sum_log(x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  S = Vector{T}(undef,n+1)
  esf_sum_log!(S,x)
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

function esf_dc_fft!(S::AbstractArray{T,1}, tempS::AbstractArray{T,2}, 
                       x::AbstractArray{T,1}, si::AbstractArray{T,1},
                       group_sizes::AbstractArray{D,1},
                       group_start_idx::AbstractArray{D,1}) where {T <: Real, D <: Integer}
  n = length(x)
  M = size(tempS)[2]

  #convolve initial subsets
  @inbounds for g in 1:M
    @views esf_sum!(tempS[1:(group_sizes[g]+1),g], 
                    x[group_start_idx[g]:(group_start_idx[g]+group_sizes[g]-1)])
    group_sizes[g] += 1
  end
  
  while M > 1
    next_avail_col = 1
    @inbounds for g in 1:2:M
      m = group_sizes[g] + group_sizes[g+1] - 1
      si .= zero(T)
      @views _filt!(S[1:m], tempS[1:group_sizes[g],g], tempS[1:m,g+1], si[1:(group_sizes[g]-1)])
      @views copyto!(tempS[1:m,next_avail_col], S[1:m]) 
      group_sizes[next_avail_col] = m
      next_avail_col += 1
    end
    M = div(M,2)
  end
end

function esf_dc_fft(x::AbstractArray{T,1}, k::D=2) where {T <: Real, D <: Integer}
  n = length(x)
  k = min(floor(D, log2(n)), k)
  M = 2^k
  L = n/M
  r = rem(n,M) / M

  group_sizes = [fill(floor(D, L), D(M*(1-r))); fill(ceil(D, L), D(M*r))]
  group_start_idx = cumsum(group_sizes) .- (group_sizes .- 1)

  S = Vector{T}(undef,n+1)
  tempS = zeros(T, n+1,M)
  si = zeros(T, n)

  esf_dc_fft!(S, tempS, x, si, group_sizes, group_start_idx)
  return S
end

function esf_dc_fft_reg!(S::AbstractArray{T,1}, tempS::AbstractArray{T,2}, 
                           x::AbstractArray{T,1}, si::AbstractArray{T,1}, 
                           group_sizes::AbstractArray{D,1},
                           group_start_idx::AbstractArray{D,1}) where {T <: Real, D <: Integer}
  n = length(x)
  M = size(tempS)[2]
  tempS .= zero(T)

  #convolve initial subsets
  @inbounds for g in 1:M
    @views esf_sum_reg!(tempS[1:(group_sizes[g]+1),g], 
                        x[group_start_idx[g]:(group_start_idx[g]+group_sizes[g]-1)])
    group_sizes[g] += 1
  end
  
  while M > 1
    next_avail_col = 1
    @inbounds for g in 1:2:M
      m = group_sizes[g] + group_sizes[g+1] - 1
      si .= zero(T)
      @views _filt_reg!(S[1:m], tempS[1:group_sizes[g],g], tempS[1:m,g+1], si[1:(group_sizes[g]-1)], group_sizes[g]-1,group_sizes[g+1]-1)
      @views copyto!(tempS[1:m,next_avail_col], S[1:m]) 
      group_sizes[next_avail_col] = m
      next_avail_col += 1
    end
    M = div(M,2)
  end
end

function esf_dc_fft_reg(x::AbstractArray{T,1}, k::D=2) where {T <: Real, D <: Integer}
  n = length(x)
  k = min(floor(D, log2(n)), k)
  M = 2^k
  L = n/M
  r = rem(n,M) / M

  group_sizes = [fill(floor(D, L), D(M*(1-r))); fill(ceil(D, L), D(M*r))]
  group_start_idx = cumsum(group_sizes) .- (group_sizes .- 1)

  S = Vector{T}(undef,n+1)
  tempS = zeros(T, n+1,M)
  si = zeros(T, n)

  esf_dc_fft_reg!(S, tempS, x, si, group_sizes, group_start_idx)
  return S
end

function esf_diff!(S::AbstractArray{T,1}, P::AbstractArray{T,2}, x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  fill!(S, zero(T))
  S[1] = one(T)
  S[2] = sum(x)
  P[:,1] .= one(T)
  @inbounds for i in 2:n
    for j in 1:n
      P[j,i] = S[i] - x[j] * P[j,i-1]
      S[i+1] = S[i+1] + x[j] * P[j,i]
    end
    S[i+1] /= i
  end
end
"""
    esf_diff(x)

Compute the elementary symmetric functions of order k = 1,...,n
where n = length(x) using the difference algorithm. Also computes
the matrix of first derivatives. 

# Examples
```julia-repl
julia> esf_diff([3.5118, .6219, .2905, .8450, 1.8648])[1]
6-element Array{Float64,1}:
  1.0
  7.134
 16.9493
 16.7781
  7.05289
  0.999736

julia> esf_diff([3.5118, .6219, .2905, .8450, 1.8648])[2]
5×5 Array{Float64,2}:
  1.0  3.6222   4.22884   1.92728  0.284679
  1.0  6.5121  12.8994    8.75598  1.60755
  1.0  6.8435  14.9612   12.4319   3.44143
  1.0  6.289   11.6351    6.94648  1.18312
  1.0  5.2692   7.12328   3.49463  0.536109
```
"""
function esf_diff(x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  S = Vector{T}(undef,n+1)
  P = Matrix{T}(undef,n,n)
  esf_diff!(S,P,x)
  return S, P
end

function esf_diff_onlyP!(S::AbstractArray{T,1}, P::AbstractArray{T,2}, x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  P[:,1] .= one(T)
  @inbounds for i in 2:n
    for j in 1:n
      P[j,i] = S[i] - x[j] * S[j,i-1]
    end
  end
end

function esf_diff_reg!(S::AbstractArray{T,1}, P::AbstractArray{T,2}, x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  fill!(S, zero(T))
  S[1] = one(T)
  S[2] = sum(x)/n
  P[:,1] .= one(T)/n
  @inbounds for i in 2:n
    for j in 1:n
      P[j,i] = (S[i] - x[j] * P[j,i-1]) * (i / (n+1-i))
      S[i+1] = S[i+1] + x[j] * P[j,i]
    end
    S[i+1] /= i
  end
end

function esf_diff_reg(x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  S = Vector{T}(undef,n+1)
  P = Matrix{T}(undef,n,n)
  esf_diff_reg!(S,P,x)
  return S, P
end

function esf_diff_updown(x::AbstractArray{T,1}) where T<:Real
  n = length(x)
  S = Vector{T}(undef,n+1)
  P = Matrix{T}(undef,n,n+1)
  err = zero(T)
  esf_diff_updown!(S,P,err,x)
  return S, P[:,1:end-1], err
end


function esf_diff_updown!(S::AbstractArray{T,1}, P::AbstractArray{T,2}, 
                          fw_bk_err::T, 
                          x::AbstractArray{T,1}) where T <: Real
  n = length(x)
  fill!(S, zero(T))
  S[1] = one(T)
  S[2] = sum(x)
  S[n+1] = prod(x)
  P[:,1] .= one(T)
  P[:,end] .= zero(T)
  mid = div(n, 2) + 1
  @inbounds for i in 2:(mid-1)
    for j in 1:n
      P[j,i] = S[i] - x[j] * P[j,i-1]
      S[i+1] = S[i+1] + x[j] * P[j,i]
    end
    S[i+1] /= i
  end
  S_mid_est = S[mid]
  S[mid] = zero(T)
  @inbounds for ii in mid:n
    i = n - ii + mid
    for j in 1:n
      P[j,i] = (S[i+1] - P[j,i+1]) / x[j]
      S[i] += P[j,i]
    end
    S[i] /= (n - (i-1))
  end
  fw_bk_err = abs(one(T) - S_mid_est / S[mid])
end