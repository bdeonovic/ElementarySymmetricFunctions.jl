using ElementarySymmetricFunctions
using Test


function naive_esf(x::AbstractVector{T}) where T <: Real
    n = length(x)
    S = zeros(T, n+1)
    states = hcat(reverse.(digits.(0:2^n-1,base=2,pad=n))...)'

    r_states = vec(mapslices(sum, states, dims=2))

    for r in 0:n
        idx = findall(r_states .== r)
        S[r+1] = sum(mapslices(x->prod(x[x .!= 0]), states[idx, :] .* x', dims=2))
    end
    return S
end

@testset "ElementarySymmetricFunctions.jl" begin
    x = [3.5118, .6219, .2905, .8450, 1.8648]
    n = length(x)

    @test esf_sum(x) ≈ naive_esf(x)
    @test esf_sum_reg(x) ≈ (naive_esf(x) ./ binomial.(n,0:n))
end
