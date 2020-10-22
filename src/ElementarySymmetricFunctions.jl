module ElementarySymmetricFunctions
    import SpecialFunctions.logabsbinomial

    include("util.jl")
    include("esf.jl")
    include("poisbin.jl")

    export esf_sum, esf_sum_reg, esf_dc_fft, esf_dc_fft_reg
    export poisbin_sum_taub, poisbin_fft, poisbin_chen, poisbin_dc_fft, poisbin_fft_cf
end
