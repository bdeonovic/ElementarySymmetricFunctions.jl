using ElementarySymmetricFunctions
using Test


function naive_esf(x::AbstractVector{T}) where T <: Real
    n = sum(x .!= 0)
    xidx = x .!= 0
    S = zeros(T, n+1)
    states = hcat(reverse.(digits.(0:2^n-1,base=2,pad=n))...)'

    r_states = vec(mapslices(sum, states, dims=2))

    for r in 0:n
        idx = findall(r_states .== r)
        S[r+1] = sum(mapslices(x->prod(x[x .!= 0]), states[idx, :] .* x[xidx]', dims=2))
    end
    return S
end

function naive_pb(p::AbstractVector{T}) where T <: Real
    x = p ./ (1 .- p)
    naive_esf(x) * prod(1 ./ (1 .+ x))
end

@testset "ElementarySymmetricFunctions.jl" begin
    x = [3.5118, .6219, .2905, .8450, 1.8648]
    n = length(x)

    naive_sol = naive_esf(x)
    naive_sol_reg = naive_sol ./ binomial.(n,0:n)

    #test recursive and FFT algos
    @test esf_sum(x) ≈ naive_sol
    @test esf_sum_reg(x) ≈ naive_sol_reg
    @test esf_sum_log(x) ≈ log.(naive_sol)

    @test esf_dc_fft(x) ≈ naive_sol
    @test esf_dc_fft_reg(x) ≈ naive_sol_reg

    #test derivatives and difference algos
    G = zeros(n,n)
    [G[:,i] .= x for i in 1:n]
    [G[i,i] = 0 for i in 1:n]
    naive_P = mapslices(naive_esf, G, dims=1)'
    naive_P_reg = mapslices(x->naive_esf(x) ./ binomial.(n, 1:n), G, dims=1)'

    S,P = esf_diff(x)
    @test S ≈ naive_sol
    @test P ≈ naive_P

    S,P,err = esf_diff_updown(x)
    @test S ≈ naive_sol
    @test P ≈ naive_P

    S,P = esf_diff_reg(x)
    @test S ≈ naive_sol_reg
    @test P ≈ naive_P_reg



    y = zeros(n)
    H_naive = zeros(n,n,n+1)
    for i in 1:n
        for j in 1:n
            y .= x; y[i]=0; y[j]=0;
            H_naive[i,j,1:(sum(y .!=0)+1)] .= naive_esf(y)
        end
    end

    @test esf_sum_dervs_1(x)[2] ≈ vcat([H_naive[i,i,:] for i in 1:n]'...)
    @test esf_sum_dervs_2(x)[2] ≈ H_naive

    y = zeros(n)
    H_naive_reg = zeros(n,n,n+1)
    for i in 1:n
        for j in 1:n
            y .= x; y[i]=0; y[j]=0;
            H_naive_reg[i,j,1:(sum(y .!=0)+1)] .= naive_esf(y) ./ binomial.(n, (0+sum(y .== 0)):n)
        end
    end
    @test esf_sum_dervs_1_reg(x)[2] ≈ vcat([H_naive_reg[i,i,:] for i in 1:n]'...)
    @test esf_sum_dervs_2_reg(x)[2] ≈ H_naive_reg

    #test poisson-binomial algos
    p = x ./ (1 .+ x)
    naive_sol = naive_pb(p)

    @test poisbin_sum_taub(p) ≈ naive_sol
    @test poisbin_fft(p) ≈ naive_sol
    @test poisbin_fft_cf(p) ≈ naive_sol
    @test poisbin_chen(p) ≈ naive_sol
    @test poisbin_dc_fft(p) ≈ naive_sol
    @test poisbin_sum_taub_log(p) ≈ log.(naive_sol)

end
