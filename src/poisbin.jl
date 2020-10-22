function poisbin_fft!(S::AbstractArray{T,1}, y::AbstractArray{Complex{T},1}, 
                      z::AbstractArray{Complex{T},1},
                      p::AbstractArray{T,1}) where T <: Real
  n = length(p)
  omega = 2pi/(n+1)
  @inbounds for k in 0:n
    for m in 1:n
      y[k+1] *= p[m] * exp(im * omega * k) + (1-p[m])
    end
  end
  _dft!(z,y)
  S .= real.(z) / (n+1)
end

function poisbin_fft(p::AbstractArray{T,1}) where T <: Real
  n = length(p)
  S = Vector{T}(undef,n+1)
  y = ones(Complex{T}, n+1)
  z = Vector{Complex{T}}(undef,n+1)

  poisbin_fft!(S, y, z, p)
  return S
end

#implementation from Distributions.jl to compare
function poisbin_fft_cf!(S::AbstractArray{T,1}, y::AbstractArray{Complex{T},1}, 
                         z::AbstractArray{Complex{T},1}, 
                         p::AbstractArray{T,1}) where T<: Real
  n = length(p)
  y[1] = one(Complex{T}) / (n+1)
  omega = 2 * one(T) / (n+1)
  kmax = ceil(Int, n/2)
  @inbounds for k in 1:kmax
    logz = zero(T)
    argz = zero(T)
    for j in 1:n
      zjl = 1 - p[j] + p[j] * cospi(omega*k) + im * p[j] * sinpi(omega*k)
      logz += log(abs(zjl))
      argz += atan(imag(zjl), real(zjl))
    end
    dl = exp(logz)
    y[k+1] = dl * cos(argz) / (n+1) + im * dl * sin(argz) / (n+1)
    if n + 1 - k > k
      y[n + 1 - k + 1] = conj(y[k+1])
    end
  end
  _dft!(z, y)
  S .= real.(z)
end

function poisbin_fft_cf(p::AbstractArray{T,1}) where T <: Real
  n = length(p)
  S = Vector{T}(undef,n+1)
  y = ones(Complex{T}, n+1)
  z = Vector{Complex{T}}(undef,n+1)

  poisbin_fft_cf!(S, y, z, p)
  return S
end

function poisbin_sum_taub!(S::AbstractArray{T}, p::AbstractArray{T}) where T <: Real
  n = length(p)
  fill!(S,zero(T))
  S[1] = 1-p[1]
  S[2] = p[1]
  @inbounds for col in 2:n
    for r in 1:col
        row = col - r + 1 
        S[row+1] = (1-p[col])*S[row+1] + p[col] * S[row]
    end
    S[1] *= 1-p[col]
  end
end

function poisbin_sum_taub_log!(S::AbstractArray{T}, p::AbstractArray{T}) where T <: Real
  n = length(p)
  fill!(S,-Inf)
  S[1] = log(1-p[1])
  S[2] = log(p[1])
  @inbounds for col in 2:n
    for r in 1:col
        row = col - r + 1 
        Sr = S[row] + log(p[col])
        Sr1 = S[row+1] + log(1-p[col])
        if (Sr1 > Sr) && (Sr1 > zero(T))
          S[row+1] = Sr1 + log1p(exp(Sr - Sr1))
        elseif (Sr >= Sr1) && (Sr > zero(T))
          S[row+1] = Sr + log1p(exp(Sr1-Sr))
        else
          S[row+1] = log(exp(Sr1) + exp(Sr))
        end
    end
    S[1] += log(1-p[col])
  end
end

function poisbin_sum_taub(p::AbstractArray{T,1}) where T <: Real
  n = length(p)
  S = Vector{T}(undef,n+1)
  poisbin_sum_taub!(S,p)
  return S
end

function poisbin_sum_taub_dervs_2(p::AbstractVector{T}) where T <: Real
  n = length(p)
  S = Vector{T}(undef,n+1)
  H = Array{T,3}(undef,n,n,n+1)
  poisbin_sum_taub!(S, p)
  poisbin_sum_taub_dervs_2!(H, p)
  return S, H
end

function poisbin_sum_taub_dervs_2!(H::AbstractArray{T,3}, p::AbstractVector{T}) where T <: Real
  n = length(p)
  pj=zero(T)
  @inbounds for j in 1:n
    pj = p[j]
    p[j] = zero(T)
    for k in j:n
      pk = p[k]
      p[k] = zero(T)
      @views poisbin_sum_taub!(H[j,k,:], p)
      H[k,j,:] .= H[j,k,:]
      p[k] = pk
    end
    p[j] = pj
  end
end

function poisbin_dc_fft!(S::AbstractArray{T,1}, tempS::AbstractArray{T,2}, 
                          p::AbstractArray{T,1}, si::AbstractArray{T,1},
                          group_sizes::AbstractArray{D,1},
                          group_start_idx::AbstractArray{D,1}) where {T <: Real, D <: Integer}
  n = length(p)
  M = size(tempS)[2]

  #convolve initial subsets
  @inbounds for g in 1:M
    @views poisbin_sum_taub!(tempS[1:(group_sizes[g]+1),g], 
                             p[group_start_idx[g]:(group_start_idx[g]+group_sizes[g]-1)])
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

function poisbin_dc_fft(p::AbstractArray{T,1}, k::D=2) where {T <: Real, D <: Integer}
  n = length(p)
  k = min(floor(D, log2(n)), k)
  M = 2^k
  L = n/M
  r = rem(n,M) / M

  group_sizes = [fill(floor(D, L), D(M*(1-r))); fill(ceil(D, L), D(M*r))]
  group_start_idx = cumsum(group_sizes) .- (group_sizes .- 1)

  S = Vector{T}(undef,n+1)
  tempS = zeros(T, n+1,M)
  si = zeros(T, n)

  poisbin_dc_fft!(S, tempS, p, si, group_sizes, group_start_idx)
  return S
end

function poisbin_chen!(S::AbstractArray{T,1}, P::AbstractArray{T,1}, 
                       p::AbstractArray{T,1}) where T <: Real
  n = length(p)
  S[1] = one(T)
  @inbounds for i in 1:n
    S[1] *= (1 - p[i])
    for j in 1:n
      P[j] += (p[i] / (1 - p[i])) ^ j
    end
  end
  @inbounds for i in 2:(n+1)
    k = i-1
    for j in 1:k
      S[i] += (-one(T))^(j-1) * S[k-j+1] * P[j]   
    end
    S[i] /= k
  end
end

function poisbin_chen(p::AbstractArray{T,1}) where T <: Real
  n = length(p)
  S = zeros(T, n+1)
  P = zeros(T, n)
  poisbin_chen!(S,P,p)
  return S
end