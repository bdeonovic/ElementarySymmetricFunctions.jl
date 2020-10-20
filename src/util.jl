function join_groups!(S::AbstractArray{T,1}, gamma1::AbstractArray{T,1}, 
                      gamma2::AbstractArray{T,1}) where T <: Real
  k1 = length(gamma1) - 1
  k2 = length(gamma2) - 1
  fill!(S, zero(T))

  @inbounds for g in 1:(k1+k2+1)
    for i in 1:g
      S[g] += (g-i+1 > length(gamma2)) || (i > length(gamma1)) ? 
        0.0 : gamma1[i] * gamma2[g-i+1]
    end
  end
end

function join_groups_reg!(S::AbstractArray{T,1}, gamma1::AbstractArray{T,1}, 
                          gamma2::AbstractArray{T,1}) where T <: Real
  k1 = length(gamma1) - 1
  k2 = length(gamma2) - 1
  fill!(S, zero(T))
  S[1] = one(T)

  @inbounds for g in 2:(k1+k2+1)
    for i in 1:g
      S[g] += (g-i+1 > length(gamma2)) || (i > length(gamma1)) ? 
        0.0 : gamma1[i] * gamma2[g-i+1] * 
              exp(lbinom(k1,i-1)+lbinom(k2,g-i)-lbinom(k1+k2,g-1))
    end
  end
end

function lbinom(n::T, s::T) where T <: Real
    -log(n+1.0) - logbeta(n-s+1.0, s+1.0)
end

#adopted from DSP.jl
function own_filt!(out::AbstractArray{T,1}, b::AbstractArray{T,1}, x::AbstractArray{T,1}, si::AbstractArray{T,1}) where T <: Real
    silen = length(si)
    @inbounds for i in 1:length(x)
        xi = x[i]
        val = muladd(xi, b[1], si[1])
        for j in 1:(silen-1)
            si[j] = muladd(xi, b[j+1], si[j+1])
        end
        si[silen] = b[silen+1]*xi
        out[i] = val
    end
end