function esf_gen!(S::AbstractArray{T,1}, x::Array{Array{T,1},1}, a::Array{Array{D,1},1}, m::Array{D,1}, ni::Array{D,1}) where {D <: Integer, T <: Real}
  n = length(x)
  fill!(S,zero(T))
  S[1] = one(T)
  @inbounds for col in 1:n
    for row in m[col]:-1:1
      for k in 1:ni[col]
        if row - a[col][k] >= 0
          S[row+1] += x[col][k] * S[row - a[col][k] + 1]
        end
      end
    end
  end
end

function esf_gen(x::AbstractArray{T,1}, a::AbstractArray{D,1}, lengths::AbstractArray{D,1}) where {D <: Integer, T <: Real}
  n = cumsum(lengths)
  m = cumsum(map(maximum,a))

  S = Vector{T}(undef,m[end]+1)
  esf_gen!(S,x,a,n,m)
  return S
end
