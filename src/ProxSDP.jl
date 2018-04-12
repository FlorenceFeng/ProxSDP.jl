module ProxSDP

using MathOptInterface, TimerOutputs

include("mathoptinterface.jl")
include("eigsolver.jl")

immutable Dims
    n::Int  # Size of primal variables
    p::Int  # Number of linear equalities
    m::Int  # Number of linear inequalities
end

type AffineSets{T}
    A::SparseMatrixCSC{Float64,Int64}#AbstractMatrix{T}
    G::SparseMatrixCSC{Float64,Int64}#AbstractMatrix{T}
    b::Vector{T}
    h::Vector{T}
    c::Vector{T}
end

type ConicSets
    sdpcone::Vector{Tuple{Vector{Int},Vector{Int}}}
end

struct CPResult
    status::Int
    primal::Vector{Float64}
    dual::Vector{Float64}
    slack::Vector{Float64}
    primal_residual::Float64
    dual_residual::Float64
    objval::Float64
end

struct CPOptions
    fullmat::Bool
    verbose::Bool
end

type PrimalDual
    x::Vector{Float64}
    x_old::Vector{Float64}
    u::Vector{Float64}
    u_old::Vector{Float64}
    u_aux::Vector{Float64}
    PrimalDual(dims) = new(zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2), zeros(dims.m+dims.p), zeros(dims.m+dims.p), zeros(dims.m+dims.p))
end

type AuxiliaryData
    m::Symmetric{Float64,Matrix{Float64}}
    Mtu::Vector{Float64}
    TMtu::Vector{Float64}
    Mtu_old::Vector{Float64}
    Mtu_diff::Vector{Float64}
    SMx::Vector{Float64}
    SMx_old::Vector{Float64}
    Mx::Vector{Float64}
    Mx_old::Vector{Float64}
    u_1::Vector{Float64}
    u_2::Vector{Float64}
    u_diff::Vector{Float64}
    AuxiliaryData(dims) = new(
        Symmetric(zeros(dims.n, dims.n), :L), zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2), zeros(dims.n*(dims.n+1)/2),
        zeros(dims.n*(dims.n+1)/2), zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m), zeros(dims.p+dims.m)
    )
end

function chambolle_pock(affine_sets::AffineSets, conic_sets::ConicSets, dims::Dims, verbose=true, max_iter=Int(1e+5), tol=1e-4)::CPResult

    if verbose
        println("======================================================================")
        println("          ProxSDP : Proximal Semidefinite Programming Solver          ")
        println("                 (c) Mario Souto and Joaquim D. Garcia, 2018          ")
        println("----------------------------------------------------------------------")
        println(" Initializing Primal-Dual Hybrid Gradient method")
        println("----------------------------------------------------------------------")
        println("|  iter  | comb. res | prim. res |  dual res |    rank   |  time (s)  ")
        println("----------------------------------------------------------------------")
    end

    time0 = time()
    tic()
    @timeit "Init" begin
        opt = CPOptions(false, verbose)
        # Scale objective function
        c_orig, idx = preprocess!(affine_sets, dims, conic_sets)

        # Initialization
        pair = PrimalDual(dims)
        a = AuxiliaryData(dims)
        arc = ARPACKAlloc(Float64)
        target_rank = 1
        
        # logging
        rank_update = 0
        best_prim_residual, best_dual_residual = Inf, Inf
        converged, polishing = false, false
        primal_residual, dual_residual, comb_residual = zeros(max_iter), zeros(max_iter), zeros(max_iter)
        M = vcat(affine_sets.A, affine_sets.G)
        Mt = M'
        rhs = vcat(affine_sets.b, affine_sets.h)

        # Diagonal scaling
        affine_sets, TMt, Tc, S, SM, Sinv, T = diag_scaling(affine_sets, 1.0, dims, M, Mt)

        # Stepsize parameters and linesearch parameters
        # primal_step = 1.0 / svds(M; nsv=1)[1][:S][1]
        primal_step = 1.0
        dual_step = primal_step
        primal_step_start = primal_step
        beta, theta = 1.0, 1.0  # Ratio (dual / primal) and overrelaxation parameter
        adapt_level = 0.9       # Factor by which the stepsizes will be balanced 
        adapt_decay = 0.8       # Rate the adaptivity decreases over time
        l = 500
        max_prim_res, max_dual_res = 1.0, 1.0
        best_comb_residual = Inf
        best_x, best_u = copy(pair.x), copy(pair.u)
        norm_c, norm_rhs = norm(affine_sets.c), norm(rhs)
        # norm_c, norm_rhs = norm(Tc), norm(S * rhs)

        # Initial iterates
        pair.x[1] = 1.0
        # dual_step!(pair, a, dims, affine_sets, M, Mt, dual_step, theta)::Void
        @timeit "dual" dual_step!(pair, a, dims, affine_sets, S, SM, Sinv, Mt, dual_step, theta)::Void
    end

    # Fixed-point loop
    @timeit "CP loop" for k in 1:max_iter

        # Update primal variable
        # @timeit "primal" target_rank, min_eig = primal_step!(pair, a, dims, conic_sets, target_rank, M, Mt, affine_sets.c, primal_step, arc)::Tuple{Int64, Float64}
        @timeit "primal" target_rank, min_eig = primal_step!(pair, a, dims, conic_sets, target_rank, M, TMt, Tc, primal_step, arc)::Tuple{Int64, Float64}
        
        # Dual update with linesearch
        # @timeit "linesearch" primal_step, dual_step, beta, theta = linesearch!(pair, a, dims, affine_sets, S, SM, Sinv, TMt, primal_step, beta, theta)::Tuple{Float64, Float64, Float64, Float64}
        @timeit "dual" dual_step!(pair, a, dims, affine_sets, S, SM, Sinv, Mt, dual_step, theta)::Void

        # Compute residuals and update old iterates
        @timeit "logging" max_prim_res, max_dual_res = compute_residual(pair, a, primal_residual, dual_residual, comb_residual, primal_step, dual_step, k, dims, max_prim_res, max_dual_res, norm_c, norm_rhs)::Tuple{Float64, Float64}

        # Save best incumbent
        if comb_residual[k] < best_comb_residual
            best_comb_residual = comb_residual[k]
            best_x, best_u = copy(pair.x), copy(pair.u)
        end

        # Print progress
        if mod(k, 1000) == 0 && opt.verbose
            print_progress(k, primal_residual[k], dual_residual[k], target_rank, time0)::Void
        end

        # Check convergence
        rank_update += 1
        if primal_residual[k] < tol && dual_residual[k] < tol && k >100
            # Check convergence of inexact fixed-point
            target_rank, min_eig = sdp_cone_projection!(pair.x, a, dims, conic_sets, target_rank + 1, arc)::Tuple{Int64, Float64}
            println(min_eig)

            if min_eig < tol
                converged = true
                best_prim_residual, best_dual_residual = primal_residual[k], dual_residual[k]
                print_progress(k, primal_residual[k], dual_residual[k], target_rank, time0)::Void
                break
            elseif rank_update > l && dims.n > 100 && target_rank <= 10
                target_rank *= 2
                rank_update = 0
            end

        # Check divergence
        elseif k > l && comb_residual[k - l] < comb_residual[k] && rank_update > l
            target_rank *= 2
            rank_update = 0
            # pair.x, pair.u = copy(best_x), copy(best_u)

        # Adaptive beta  
        elseif primal_residual[k] > tol && dual_residual[k] < tol
            beta = max(beta * (1 - adapt_level), 1e-3)
            adapt_level *= adapt_decay
        elseif primal_residual[k] < tol && dual_residual[k] > tol
            beta = min(beta * (1 + adapt_level), 1e+3)
            adapt_level *= adapt_decay  
        end
    end

    # Compute results
    time_ = toq()
    prim_obj = dot(c_orig, T * pair.x)
    dual_obj = dot(rhs, pair.u)
    pair.x = pair.x[idx]
    if verbose
        println("----------------------------------------------------------------------")
        if converged
            println(" Status = solved")
        else
            println(" Status = ProxSDP failed to converge")
        end
        println(" Elapsed time = $(round(time_, 2))s")
        println("----------------------------------------------------------------------")
        println(" Primal objective = $(round(prim_obj, 4))")
        println(" Dual objective = $(round(dual_obj, 4))")
        println(" Duality gap = $(round(prim_obj - dual_obj, 4))")
        println("======================================================================")
    end

    return CPResult(Int(converged), pair.x, pair.u, 0.0*pair.x, best_prim_residual, best_dual_residual, prim_obj)
end

function compute_residual(pair::PrimalDual, a::AuxiliaryData, primal_residual::Array{Float64,1}, dual_residual::Array{Float64,1}, comb_residual::Array{Float64,1}, primal_step::Float64, dual_step::Float64, iter::Int64, dims::Dims, max_prim_res::Float64, max_dual_res::Float64, norm_c::Float64, norm_rhs::Float64)::Tuple{Float64, Float64}    
    # Compute primal residual
    Base.LinAlg.axpy!(-1.0, a.Mtu, a.Mtu_old)
    Base.LinAlg.axpy!((1.0 / (1.0 + primal_step)), pair.x_old, a.Mtu_old)
    Base.LinAlg.axpy!(-(1.0 / (1.0 + primal_step)), pair.x, a.Mtu_old)
    primal_residual[iter] = norm(a.Mtu_old, 2) / (1.0 + norm_c)

    # Compute dual residual
    Base.LinAlg.axpy!(-1.0, a.Mx, a.Mx_old)
    Base.LinAlg.axpy!((1.0 / (1.0 + dual_step)), pair.u_old, a.Mx_old)
    Base.LinAlg.axpy!(-(1.0 / (1.0 + dual_step)), pair.u, a.Mx_old)
    dual_residual[iter] = norm(a.Mx_old, 2) / (1.0 + norm_rhs)

    # Compute combined residual
    comb_residual[iter] = primal_residual[iter] + dual_residual[iter]

    # Keep track of previous iterates
    copy!(pair.x_old, pair.x)
    copy!(pair.u_old, pair.u)
    copy!(a.Mtu_old, a.Mtu)
    copy!(a.Mx_old, a.Mx)
    return max(max_prim_res, primal_residual[iter]), max(max_dual_res, dual_residual[iter])
end

function diag_scaling(affine_sets::AffineSets, alpha::Float64, dims::Dims, M::SparseMatrixCSC{Float64,Int64}, Mt::SparseMatrixCSC{Float64,Int64})
    # Diagonal scaling
    div = vec(sum(abs.(M).^(2.0-alpha), 1))
    div[find(x-> x == 0.0, div)] = 1e-4
    T = spdiagm(1.0 ./ div)
    div = vec(sum(abs.(M).^alpha, 2))
    div[find(x-> x == 0.0, div)] = 1e-4
    S = spdiagm(1.0 ./ div)

    T, S = speye(size(T)...), speye(size(S)...)

    Sinv = div

    # Cache matrix multiplications
    TMt = T * Mt
    Tc = T * affine_sets.c
    SM = S * M
    rhs = vcat(affine_sets.b, affine_sets.h)
    Srhs = S * rhs
    affine_sets.b = Srhs[1:dims.p]
    affine_sets.h = Srhs[dims.p+1:end]

    return affine_sets, TMt, Tc, S, SM, Sinv, T
end

# function dual_step!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, affine_sets::AffineSets, M::SparseMatrixCSC{Float64,Int64}, Mt::SparseMatrixCSC{Float64,Int64}, dual_step::Float64, theta::Float64)::Void
    
#     @inbounds @simd for i in eachindex(pair.u)
#         a.u_1[i] = theta * a.Mx_old[i]
#     end
#     Base.LinAlg.axpy!(-(1.0 + theta), a.Mx, a.u_1) #  (1 + theta) * K * x
#     Base.LinAlg.axpy!(-dual_step, a.u_1, pair.u)   # alpha*x + y
#     @inbounds @simd for i in eachindex(pair.u)
#         a.u_1[i] = pair.u[i] / dual_step
#     end
#     @timeit "box" box_projection!(a.u_1, dims, affine_sets)
#     Base.LinAlg.axpy!(-dual_step, a.u_1, pair.u) # alpha*x + y
#     A_mul_B!(a.Mtu, Mt, pair.u)
#     return nothing
# end

function dual_step!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, affine_sets::AffineSets, S::SparseMatrixCSC{Float64,Int64}, SM::SparseMatrixCSC{Float64,Int64}, Sinv::Vector{Float64}, Mt::SparseMatrixCSC{Float64,Int64}, dual_step::Float64, theta::Float64)::Void
    copy!(a.u_1, a.SMx_old)
    A_mul_B!(a.SMx, SM, pair.x)
    Base.LinAlg.axpy!(-(1.0 + theta), a.SMx, a.u_1) # alpha*x + y
    Base.LinAlg.axpy!(-dual_step, a.u_1, pair.u) # alpha*x + y

    @inbounds @simd for i in eachindex(pair.u)
        a.u_1[i] = Sinv[i] * pair.u[i] / dual_step
    end
    box_projection!(a.u_1, dims, affine_sets)
    A_mul_B!(a.u_2, S, a.u_1)
    Base.LinAlg.axpy!(-dual_step, a.u_2, pair.u) # alpha*x + y
    A_mul_B!(a.Mtu, Mt, pair.u)
    return nothing
end

function box_projection!(v::Vector{Float64}, dims::Dims, aff::AffineSets)::Void
    # Projection onto = b
    @inbounds @simd for i in 1:length(aff.b)
        v[i] = aff.b[i]
    end
    # Projection onto <= h
    @inbounds @simd for i in 1:length(aff.h)
        v[dims.p+i] = min(v[dims.p+i], aff.h[i])
    end
    return nothing
end

function linesearch!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, affine_sets::AffineSets, S::SparseMatrixCSC{Float64,Int64}, M::SparseMatrixCSC{Float64,Int64}, Sinv::Vector{Float64}, Mt::SparseMatrixCSC{Float64,Int64}, primal_step::Float64, beta::Float64, theta::Float64)::Tuple{Float64, Float64, Float64, Float64}
    max_iter_linesearch = 100
    delta = 1.0 - 1e-3
    mu = 0.7
    primal_step_old = primal_step
    primal_step = primal_step * sqrt(1.0 + theta)
    pair.u_aux = copy(pair.u)

    # Linesearch loop
    for i = 1:max_iter_linesearch
        # Inital guess for theta
        theta = primal_step / primal_step_old
        # Update dual variable
        dual_step = primal_step * beta
        @timeit "dual" dual_step!(pair, a, dims, affine_sets, S, SM, Sinv, Mt, dual_step, theta)::Void
        # Check linesearch convergence
        copy!(a.Mtu_diff, a.Mtu)
        Base.LinAlg.axpy!(-1.0, a.Mtu_old, a.Mtu_diff)
        copy!(a.u_diff, pair.u)
        Base.LinAlg.axpy!(-1.0, pair.u_old, a.u_diff)
        if primal_step * sqrt(beta) * norm(a.Mtu_diff) <= delta * norm(a.u_diff)
            return primal_step, beta * primal_step, beta, theta
        else
            pair.u = copy(pair.u_aux)
            primal_step *= mu
        end
    end

    println(":")

    primal_step = primal_step_old
    theta = 1.0
    dual_step = primal_step * 1.0
    @timeit "dual" dual_step!(pair, a, dims, affine_sets, S, SM, Sinv, Mt, dual_step, theta)::Void

    return primal_step, beta * primal_step, beta, theta
end

function primal_step!(pair::PrimalDual, a::AuxiliaryData, dims::Dims, conic_sets::ConicSets, target_rank::Int64, M::SparseMatrixCSC{Float64,Int64}, TMt::SparseMatrixCSC{Float64,Int64}, Tc::Vector{Float64}, primal_step::Float64, arc::ARPACKAlloc)::Tuple{Int64, Float64}
    A_mul_B!(a.TMtu, TMt, pair.u) # (TMt*u)
    Base.LinAlg.axpy!(-primal_step, a.TMtu, pair.x) # x=x+(-t)*(TMt*u)
    Base.LinAlg.axpy!(-primal_step, Tc, pair.x) # x=x+(-t)*Tc

    # Projection onto the psd cone
    target_rank, min_eig = sdp_cone_projection!(pair.x, a, dims, conic_sets, target_rank, arc)::Tuple{Int64, Float64}
    A_mul_B!(a.Mx, M, pair.x)
    return target_rank, min_eig
end

function preprocess!(aff::AffineSets, dims::Dims, conic_sets::ConicSets)
    c_orig = zeros(1)
    M = zeros(Int, dims.n, dims.n)
    iv = conic_sets.sdpcone[1][1]
    im = conic_sets.sdpcone[1][2]
    for i in eachindex(iv)
        M[im[i]] = iv[i]
    end
    X = Symmetric(M, :L)

    n = size(X)[1] # columns or line
    cont = 1
    sdp_vars = zeros(Int, div(n*(n+1),2))
    for j in 1:n, i in j:n
        sdp_vars[cont] = X[i,j]
        cont+=1
    end

    totvars = dims.n
    extra_vars = collect(setdiff(Set(collect(1:totvars)),Set(sdp_vars)))
    ord = vcat(sdp_vars, extra_vars)

    ids = vec(X)
    offdiag_ids = setdiff(Set(ids), Set(diag(X)))
    c_orig = copy(aff.c)
    for i in offdiag_ids
        aff.c[i] /= 2.0
    end  

    aff.A, aff.G, aff.c = aff.A[:, ord], aff.G[:, ord], aff.c[ord]
    return c_orig[ord], sortperm(ord)
end

function sdp_cone_projection!(v::Vector{Float64}, a::AuxiliaryData, dims::Dims, con::ConicSets, target_rank::Int64, arc::ARPACKAlloc)::Tuple{Int64, Float64}

    if target_rank < 1
        @show target_rank
        target_rank = 1
    end

    eig_tol = 0.0
    n = dims.n
    iv = con.sdpcone[1][1]::Vector{Int}
    im = con.sdpcone[1][2]::Vector{Int}
    @timeit "reshape1" begin
        cont = 1
        @inbounds for j in 1:n, i in j:n
            a.m.data[i,j] = v[cont]
            cont+=1
        end
    end

    if target_rank <= max(16, dims.n/100) && dims.n > 100
        @timeit "eigs" begin 
            eig!(arc, a.m, target_rank)
            if hasconverged(arc)
                fill!(a.m.data, 0.0)
                for i in 1:target_rank
                    if unsafe_getvalues(arc)[i] > 0.0
                        vec = unsafe_getvectors(arc)[:, i]
                        Base.LinAlg.BLAS.gemm!('N', 'T', unsafe_getvalues(arc)[i], vec, vec, 1.0, a.m.data)
                    end
                end
            end
        end
        if hasconverged(arc)
            @timeit "reshape2" begin
                cont = 1
                @inbounds for j in 1:n, i in j:n
                    v[cont] = a.m.data[i,j]
                    cont+=1
                end
            end
            return target_rank, minimum(unsafe_getvalues(arc))
        end
    end
    
    @timeit "eigfact" begin
        cont_ = 0
        fact = eigfact!(a.m, max(dims.n-target_rank, 1):dims.n)
        # fact = eigfact!(a.m)
        fill!(a.m.data, 0.0)
        for i in 1:length(fact[:values])
            if fact[:values][i] > 0.0
                cont_ += 1
                Base.LinAlg.BLAS.gemm!('N', 'T', fact[:values][i], fact[:vectors][:, i], fact[:vectors][:, i], 1.0, a.m.data)
            end
        end
    end

    @timeit "reshape2" begin
        cont = 1
        @inbounds for j in 1:n, i in j:n
            v[cont] = a.m.data[i,j]
            cont+=1
        end
    end
    if cont_ == 0
        return target_rank, 0.0
    else
        return target_rank, minimum(fact[:values])
    end
end

function print_progress(k::Int64, primal_res::Float64, dual_res::Float64, target_rank::Int64, time0::Float64)::Void
    s_k = @sprintf("%d", k)
    s_k *= " |"
    s_s = @sprintf("%.4f", primal_res + dual_res)
    s_s *= " |"
    s_p = @sprintf("%.4f", primal_res)
    s_p *= " |"
    s_d = @sprintf("%.4f", dual_res)
    s_d *= " |"
    s_target_rank = @sprintf("%.0f", target_rank)
    s_target_rank *= " |"
    s_time = @sprintf("%.4f", time() - time0)
    s_time *= " |"
    a = "|"
    a *= " "^max(0, 9 - length(s_k))
    a *= s_k
    a *= " "^max(0, 12 - length(s_s))
    a *= s_s
    a *= " "^max(0, 12 - length(s_p))
    a *= s_p
    a *= " "^max(0, 12 - length(s_d))
    a *= s_d
    a *= " "^max(0, 12 - length(s_target_rank))
    a *= s_target_rank
    a *= " "^max(0, 12 - length(s_time))
    a *= s_time
    println(a)
    return nothing
end

end