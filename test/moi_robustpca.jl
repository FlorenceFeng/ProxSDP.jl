function moi_robustpca(optimizer, seed, n, r)

    srand(seed)
    MOI.empty!(optimizer)
    @test MOI.isempty(optimizer)

    mu, M = robustpca_data(seed, n, r)

    nvars = ProxSDP.sympackedlen(2n+n+n*n)

    X = MOI.addvariables!(optimizer, nvars)

    Xsq = Matrix{MOI.VariableIndex}(nvars, nvars)
    ProxSDP.ivech!(Xsq, X)
    Xsq = full(Symmetric(Xsq,:U))
    W1 = Xsq[1:n,1:n]
    L = Xsq[1:n,n+1:2n]
    W2 = Xsq[n+1:2n,n+1:2n]
    S = Xsq[2n+1:3n,2n+1:3n]
    t = diag(Xsq[3n+1:end,3n+1:end])

    vov = MOI.VectorOfVariables(X)
    cX = MOI.addconstraint!(optimizer, vov, MOI.PositiveSemidefiniteConeTriangle(nvars))

    for k in 1:n*n
        MOI.addconstraint!(optimizer, MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(-1.0, t[i]); MOI.ScalarAffineTerm(-1.0, S[i])], 0.0),
             MOI.LessThan(0.0))
        MOI.addconstraint!(optimizer, MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(-1.0, t[i]); MOI.ScalarAffineTerm(1.0, S[i])], 0.0),
             MOI.LessThan(0.0))
    end
    for k in 1:n*n
        MOI.addconstraint!(optimizer, MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, L[i]); MOI.ScalarAffineTerm(1.0, S[i])], 0.0),
             MOI.EqualTo(M[i]))
    end
    MOI.addconstraint!(optimizer, MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(1.0, t[i]) for i in 1:n*n], 0.0),
             MOI.LessThan(mu))


    objf_t = vcat([MOI.ScalarAffineTerm(-0.5, W1[i,i]) for i in 1:n], [MOI.ScalarAffineTerm(-0.5, W2[i,i]) for i in 1:n])

    MOI.set!(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(objf_t, 0.0))

    MOI.set!(optimizer, MOI.ObjectiveSense(), MOI.MinSense)

    MOI.optimize!(optimizer)

    obj = MOI.get(optimizer, MOI.ObjectiveValue())

    Xsq_s = MOI.get.(optimizer, MOI.VariablePrimal(), Xsq)

    robustpca_eval(n, m, x_true, Xsq_s)

    return nothing
end