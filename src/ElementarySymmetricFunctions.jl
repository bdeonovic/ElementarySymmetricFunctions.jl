module ElementarySymmetricFunctions
    import SpecialFunctions.logabsbinomial

    include("util.jl")
    include("esf.jl")
    include("poisbin.jl")

    export esf_sum, esf_sum_reg, esf_dc_fft, esf_dc_fft_reg, esf_sum_log
    export esf_sum_dervs_1, esf_sum_dervs_2, esf_sum_dervs_1_reg, esf_sum_dervs_2_reg
    export esf_diff, esf_diff_reg, esf_diff_updown
    export poisbin_sum_taub, poisbin_fft, poisbin_chen, poisbin_dc_fft, poisbin_fft_cf, poisbin_sum_taub_log
end
