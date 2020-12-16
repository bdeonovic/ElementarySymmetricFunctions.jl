function lbinom(n::T, s::T) where T <: Integer
    if n < 60
        log(binomial(n,s))
    else
        logabsbinomial(n,s)[1]
    end
end

# A simple implementation of a DFT to avoid introducing a dependency
# on an external FFT package just for this one distribution
# adapted from Distributions.jl
function _dft!(y::Vector{T}, x::Vector{T}) where T
    n = length(x)
    y .= zero(T)
    @inbounds for j = 0:n-1, k = 0:n-1
        y[k+1] += x[j+1] * cis(-π * T(2 * mod(j * k, n)) / n)
    end
end

#adopted from DSP.jl
function _filt!(out::AbstractArray{T,1}, b::AbstractArray{T,1}, x::AbstractArray{T,1}, si::AbstractArray{T,1}) where T <: Real
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

function _filt_reg!(out::AbstractArray{T,1}, b::AbstractArray{T,1}, x::AbstractArray{T,1}, si::AbstractArray{T,1}, k1::D, k2::D) where {T <: Real, D <: Integer}
    silen = length(si)
    @inbounds for j in 1:(silen-1)
        adj1 = exp(lbinom(k1,j))
        si[j] = x[1]*b[j+1]*adj1
    end
    adj1 = exp(lbinom(k1,silen))
    si[silen] = b[silen+1]*x[1]*adj1
    out[1] = one(T)

    @inbounds for i in 2:length(x)
        xi = x[i]
        adj1 = exp(lbinom(k2,i-1)-lbinom(k1+k2,i-1))
        adj2 = (i-1) / (k1+k2-i+2)
        val = muladd(xi, b[1]*adj1, si[1]*adj2)
        for j in 1:(silen-1)
            adj1 = exp(lbinom(k1,j)+lbinom(k2,i-1)-lbinom(k1+k2,i-1))
            si[j] = muladd(xi, b[j+1]*adj1, si[j+1]*adj2)
        end
        adj1 = exp(lbinom(k1,silen)+lbinom(k2,i-1)-lbinom(k1+k2,i-1))
        si[silen] = b[silen+1]*xi*adj1
        out[i] = val
    end
end