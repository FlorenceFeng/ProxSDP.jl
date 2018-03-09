
function mimo(solver)
    srand(0)
    # if Base.libblas_name == "libmkl_rt"
    #     m = Model()
    # else
    #     m = Model(solver=solver) 
    # end
    m = Model()
    
    # Instance size
    n = 3
    # Channel
    H = randn((n, n))
    # Gaussian noise
    v = randn((n, 1))
    # True signal
    s = rand([-1, 1], n)
    # Received signal
    sigma = 0.01
    y = H * s + sigma * v
    L = [hcat(H' * H, -H' * y); hcat(-y' * H, y' * y)]

    # if Base.libblas_name == "libmkl_rt"
    #     @variable(m, X[1:n+1, 1:n+1], PSD)
    # else
    #     @variable(m, X[1:n+1, 1:n+1], SDP)
    # end
    @variable(m, X[1:n+1, 1:n+1], PSD)

    @objective(m, Min, sum(L[i, j] * X[i, j] for i in 1:n+1, j in 1:n+1))
    @constraint(m, ctr[i in 1:n+1], X[i, i] == 1.0)
    # @constraint(m, bla, X[1, 1] <= 1.0)

    # if Base.libblas_name == "libmkl_rt"
    #     JuMP.attach(m, solver)
    # end
    JuMP.attach(m, solver)

    teste = JuMP.solve(m)
end

