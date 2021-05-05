module ElementarySymmetricFunctions
    import SpecialFunctions.logabsbinomial

    include("util.jl")
    include("esf.jl")
    include("poisbin.jl")

    function esf(x::AbstractVector{T}; scale::Symbol=:identity, method=:recursive, k::D=2) where {T <: Real, D <: Integer}
        if scale == :identity
            if method == :recursive
                return esf_sum(x)
            elseif method == :DCFFT 
                return esf_dc_fft(x,k)
            elseif method == :FFT 
                return poisbin_fft(x ./ (1 .+ x)) ./ prod(1 ./(1 .+ x))
            else 
                error("method $method not implemented with scaling $scale")
            end
        elseif scale == :norm 
            p = x ./ (1 .+ x)
            if method == :recursive
                return poisbin_sum_taub(p)
            elseif method == :DCFFT 
                return poisbin_dc_fft(p,k)
            elseif method == :FFT 
                return poisbin_fft(p)
            else 
                error("method $method not implemented with scaling $scale")
            end
        elseif scale == :log
            if method == :recursive
        end
    end

    export esf_sum, esf_sum_reg, esf_dc_fft, esf_dc_fft_reg, esf_sum_log
    export esf_sum_dervs_1, esf_sum_dervs_2, esf_sum_dervs_1_reg, esf_sum_dervs_2_reg
    export esf_diff, esf_diff_reg, esf_diff_updown
    export poisbin_sum_taub, poisbin_fft, poisbin_chen, poisbin_dc_fft, poisbin_fft_cf, poisbin_sum_taub_log
end
