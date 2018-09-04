function robustpca_data(seed, n, r)

    # formulation in https://arxiv.org/pdf/1312.3039.pdf
    p = 0.1
    S = sprand(n, n, p)
    L = zeros(n, n)
    for i in 1:r
        v = randn(n)
        L += 5*(rand()+1) * kron(v,v)
    end
    mu = sum(abs, S)

    M = S + L

    return mu, M
end
function robustpca_eval(n, m, x_true, XX)

    nothing
end