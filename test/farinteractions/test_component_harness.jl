module TestComponentHarness

using Test
using Random
using LinearAlgebra
using StaticArrays
using OhMyThreads
using H2Factory
using H2Trees
using H2FactoryTestUtils

include("component_harness_fixtures.jl")

# Fresh moment/collection types minted for the harness via the `@createmoment` DSL, mirroring the
# Toeplitz harness's `@createtoeplitzmoment` types. Driving the fakes with a user-defined type (not the
# built-in `Moment`) proves the aggregation/disaggregation machinery is generic over any moment the DSL
# can create, not just H2Factory's own -- exactly the contract a consumer like iFMM relies on.
@createmoment TestHarnessMoment
@createmomentcollection TestHarnessMomentCollection
@linkmomenttocollection TestHarnessMoment TestHarnessMomentCollection

const TEST_MOMENT_TYPES = (;
    MomentType=TestHarnessMoment, MomentCollectionType=TestHarnessMomentCollection
)

# Assert every moment/collection the fake built is the fresh type, then return the map. `momentcollections`
# is Galerkin-only; a Petrov map splits it into `test`/`trialmomentcollections`.
function checktestworkflow(A)
    @test all(m -> m isa TestHarnessMoment, values(H2Factory.trialmoments(A)))
    @test all(m -> m isa TestHarnessMoment, values(H2Factory.testmoments(A)))
    # Both sides, always: for a Galerkin map the two accessors resolve to the same shared
    # dictionary, so this is the Petrov check and the Galerkin one at once -- and it no longer
    # depends on which FIELDS the map happens to have, which a workspace does not expose.
    @test all(
        m -> m isa TestHarnessMomentCollection, values(H2Factory.trialmomentcollections(A))
    )
    @test all(
        m -> m isa TestHarnessMomentCollection, values(H2Factory.testmomentcollections(A))
    )
    return A
end

# Every harness map is built through these, so the whole suite runs on the fresh DSL type and each
# construction is type-checked -- the direct mirror of `testfaketoeplitzmap`/`testfakepetrovtoeplitzmap`.
testfakegalerkinmap(args...; kwargs...) =
    checktestworkflow(fakegalerkinmap(args...; merge((; kwargs...), TEST_MOMENT_TYPES)...))
testfakepetrovmap(args...; kwargs...) =
    checktestworkflow(fakepetrovmap(args...; merge((; kwargs...), TEST_MOMENT_TYPES)...))

# Component-level aggregation/disaggregation harness (plan C.0.4). Unlike B.4's single composed
# `farmul!` check, this exercises aggregation and disaggregation *individually*, the traversal
# *structure*, and both aggregation modes, against two independent oracles from H2FactoryTestUtils:
# the naive `ref_*` reference (C.0.2) and the mini 1D interpolation H2 matrix (C.0.3). This is the
# safety net the C.1/C.3/C.4 restructuring phases lean on.
#
# Seeded for reproducibility -- the assertions use random inputs, so a fixed seed keeps a failure
# deterministic instead of intermittently seed-dependent.
Random.seed!(0x2c0d_ab11)

@testset "Reference oracle agreement (ref_farmul == farmul!, complex + real)" begin
    # Fake Galerkin maps: scalar coupling on the B.4 tree, plus the deep uniform tree. Both a
    # complex and a real Float64 coupling.
    for (T, c1, c2) in ((ComplexF64, 2.0 - 1.0im, 1.5 + 0.5im), (Float64, 2.0, 1.5))
        tree, N = b4tree()
        A = testfakegalerkinmap(
            tree;
            T=T,
            i2i=identitytranslator(),
            o2o=identitytranslator(),
            i2o=scalartranslator(c1),
        )
        x = randn(T, N)
        @test ref_farmul(A, x) ≈ H2Factory.farmul!(similar(x), A, x)

        A2 = testfakegalerkinmap(
            H2Trees.buildtree(
                [SVector(c, 0.0, 0.0) for c in uniformpts(256)];
                builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
            );
            T=T,
            i2o=scalartranslator(c2),
        )
        x2 = randn(T, 256)
        @test ref_farmul(A2, x2) ≈ H2Factory.farmul!(similar(x2), A2, x2)
    end
end

@testset "mul! closed form (multi-cluster scalar: forward/transpose/adjoint, complex + real)" begin
    # Six well-separated clusters (120 points) -> every cross-cluster pair is far, every
    # within-cluster pair near, so the far matrix has the CLOSED FORM  F[i,j] = c  iff  i,j are in
    # different clusters. Checks `farmul!` against analytic values, not another impl. Complex `c`
    # (adjoint conjugates) and real `c` (adjoint == transpose).
    pts, cid = manyclusters(6, 20)
    N = length(pts)
    tree = H2Trees.buildtree(
        pts; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=1.0, minvalues=0, root=1)
    )
    for (T, c) in ((ComplexF64, 2.0 - 1.0im), (Float64, 2.0))
        A = testfakegalerkinmap(tree; T=T, i2o=scalartranslator(c))
        F = [cid[i] != cid[j] ? c : zero(T) for i in 1:N, j in 1:N]
        x = randn(T, N)
        @testset "$T" begin
            @test H2Factory.farmul!(similar(x), A, x) ≈ F * x                       # forward
            @test H2Factory.farmul!(similar(x), transpose(A), x) ≈ transpose(F) * x # transpose
            @test H2Factory.farmul!(similar(x), adjoint(A), x) ≈ F' * x             # adjoint
        end
    end
    # If the geometry produced any within-cluster far pair the closed forms above would have failed;
    # this pins that the far partition is exactly the cross-cluster structure. `ref_farpairs`
    # returns tree-node ids, so map each far node to the cluster of one of its (tight) subtree DOFs.
    A = testfakegalerkinmap(tree; T=ComplexF64, i2o=scalartranslator(1.0 + 0.0im))
    @test all(
        cid[first(H2Trees.values(tree, t))] != cid[first(H2Trees.values(tree, s))] for
        (t, s) in ref_farpairs(A)
    )
    # Note: for a Galerkin scalar map F is symmetric so transpose == forward; the genuinely
    # non-symmetric transpose is covered by the Petrov overlapping case below (test != trial).
end

@testset "Complete dense matrix (denseh2matrix): mul! all modes, complex + real, uniform + non-uniform" begin
    # The fake is a linear operator whose exact far matrix is fully determined by its translators +
    # tree, so `denseh2matrix(A)` assembles it independently of `mul!` (via reftransfer/refcoupling,
    # never calling the map). `mul!` in ALL three modes must match F / transpose(F) / adjoint(F).
    # Run both a COMPLEX coupling (adjoint != transpose) and a REAL Float64 one (adjoint == transpose).
    for (Tname, T, coupling) in
        (("ComplexF64", ComplexF64, 1.5 + 0.5im), ("Float64", Float64, 1.5))
        trees = Dict(
            "uniform" => (
                H2Trees.buildtree(
                    [SVector(c, 0.0, 0.0) for c in uniformpts(256)];
                    builder=H2Trees.TwoNTreeBuilder(;
                        minhalfsize=0.008, minvalues=0, root=1
                    ),
                ),
                256,
            ),
            "non-uniform" => nonuniformtree(),
        )
        for (name, (tree, N)) in trees
            A = testfakegalerkinmap(tree; T=T, i2o=scalartranslator(coupling))
            F = denseh2matrix(A)
            x = randn(T, N)
            @testset "$Tname / $name tree" begin
                @test H2Factory.farmul!(similar(x), A, x) ≈ F * x
                @test H2Factory.farmul!(similar(x), transpose(A), x) ≈ transpose(F) * x
                @test H2Factory.farmul!(similar(x), adjoint(A), x) ≈ F' * x
                @test ref_farmul(A, x) ≈ F * x           # reference agrees with the matrix too
                # Per-pair I2O translation (real single-pair pipeline, all modes) vs dense coupling.
                @test checkdensei2o(A; n=3) > 0
            end
        end
    end
    # Confirm the "non-uniform" tree genuinely has leaves at different levels (else it wouldn't
    # exercise the cross-level traversal this test is here for).
    @test length(leaflevels(nonuniformtree()[1])) > 1
end

@testset "Petrov (BlockTree): all modes vs complete matrix; balanced + unbalanced + asymmetric" begin
    # A Petrov map has SEPARATE test/trial trees (a `BlockTree`); aggregation runs on the trial
    # side, disaggregation on the test side, so `F` is `Ntest × Ntrial`. H2Factory's Petrov
    # traversal must match the independently-assembled complete matrix in ALL modes. (That it does
    # localizes the known iFMM Petrov bug to iFMM's construction, not the H2Factory traversal.)
    #
    # Geometries covered, for BALANCED trees (uniform points) AND UNBALANCED trees (dense cluster +
    # isolated points, `minvalues=1` -> leaves at many levels):
    #   * coincident   -- test == trial (F symmetric);
    #   * disjoint-far -- regions far apart -> single root<->root far pair -> F = c*ones;
    #   * overlapping  -- partial overlap -> mixed near/far -> genuinely NON-symmetric F.
    # Plus ASYMMETRIC DOF counts: many more test points than trial (rectangular F, `Ntest >> Ntrial`)
    # and vice versa -- the finer-test/coarser-trial shape typical of a real Petrov discretization,
    # which also gives the test and trial trees genuinely DIFFERENT depths.
    u = uniformpts(128)
    b = petrovunb()
    for (T, c) in ((ComplexF64, 2.0 - 1.0im), (Float64, 2.0))
        for (name, tp, rp, tv, rv, mhs) in dense_petrov_point_configs()
            A = testfakepetrovmap(
                tp,
                rp;
                T=T,
                i2o=scalartranslator(c),
                minhalfsize=mhs,
                testminvalues=tv,
                trialminvalues=rv,
            )
            F = denseh2matrix(A)
            x = randn(T, size(A, 2))
            w = randn(T, size(A, 1))
            @testset "$T / $name" begin
                @test H2Factory.farmul!(similar(x, size(A, 1)), A, x) ≈ F * x
                @test H2Factory.farmul!(similar(w, size(A, 2)), transpose(A), w) ≈
                    transpose(F) * w
                @test H2Factory.farmul!(similar(w, size(A, 2)), adjoint(A), w) ≈ F' * w
                @test ref_farmul(A, x) ≈ F * x
                # Per-pair I2O translation (real single-pair pipeline, all modes) vs dense coupling.
                @test checkdensei2o(A; n=3) > 0
            end
        end
    end
    # disjoint-far closed form: entirely-far regions -> a single root<->root far pair with
    # identity transfers + all-ones bases -> F = c * ones(Ntest, Ntrial).
    c = 3.0 - 2.0im
    Afar = testfakepetrovmap(
        mkpts(u), mkpts(100.0 .+ u); T=ComplexF64, i2o=scalartranslator(c), minhalfsize=0.02
    )
    @test denseh2matrix(Afar) ≈ fill(c, size(Afar))
    # overlapping really is non-symmetric (so the transpose/adjoint tests above have teeth).
    Aov = testfakepetrovmap(
        mkpts(u),
        mkpts(0.5 .+ u);
        T=ComplexF64,
        i2o=scalartranslator(1.0 + 0.0im),
        minhalfsize=0.02,
    )
    @test !issymmetric(denseh2matrix(Aov))
    # The unbalanced trees genuinely span multiple levels (else the "unbalanced" configs wouldn't be
    # exercising the cross-level traversal they claim to).
    Aunb = testfakepetrovmap(
        mkpts(b),
        mkpts(b);
        T=ComplexF64,
        i2o=scalartranslator(1.0 + 0.0im),
        minhalfsize=0.001,
        testminvalues=1,
        trialminvalues=1,
    )
    @test length(leaflevels(H2Trees.testtree(Aunb))) > 1
    # The many-test config genuinely has many more test DOFs than trial DOFs (rectangular F) AND,
    # from the differing point densities (`minvalues=1`), test and trial trees of different depths.
    Amany = testfakepetrovmap(
        mkpts(uniformpts(256)),
        mkpts(uniformpts(16));
        T=ComplexF64,
        i2o=scalartranslator(1.0 + 0.0im),
        minhalfsize=0.004,
        testminvalues=1,
        trialminvalues=1,
    )
    @test size(Amany) == (256, 16)
    @test leaflevels(H2Trees.testtree(Amany)) != leaflevels(H2Trees.trialtree(Amany))
end

@testset "Mini H2 matrix: all modes vs Ã (complex + real kernels)" begin
    pts = uniformpts(256)
    realkernel = (x, y) -> 1 / (1 + (x - y)^2)
    # Default is a COMPLEX kernel (so adjoint != transpose is meaningful); a real kernel gives the
    # Float64 path. The element type is inferred from the kernel and used consistently.
    for (kname, kernel) in
        (("complex", H2FactoryTestUtils.COMPLEXKERNEL), ("real Float64", realkernel))
        @testset "$kname kernel" begin
            for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
                for rank in (1, 2)
                    mini = minih2matrix(
                        pts;
                        rank=rank,
                        minhalfsize=0.008,
                        kernel=kernel,
                        aggregationmode=mode,
                    )
                    A = mini.A
                    F = mini.dense_approx
                    x = randn(eltype(A), length(pts))
                    # The traversal reproduces the self-assembled Ã in ALL modes (machine precision).
                    @test A * x ≈ F * x                             # forward
                    @test transpose(A) * x ≈ transpose(F) * x       # transpose
                    @test adjoint(A) * x ≈ F' * x                   # adjoint
                    @test ref_farmul(A, x) ≈ F * x                  # naive reference agrees too
                    @test denseh2matrix(A) ≈ F                      # independent assembly == Ã
                end
            end
        end
    end

    # Approximation quality (complex kernel): rank-2 (linear) beats rank-1 (constant) vs the TRUE
    # kernel -- the classic interpolation-order improvement.
    x = randn(ComplexF64, length(pts))
    eA = let m = minih2matrix(pts; rank=1, minhalfsize=0.008)
        norm((m.dense_true - m.dense_approx) * x) / norm(m.dense_true * x)
    end
    eB = let m = minih2matrix(pts; rank=2, minhalfsize=0.008)
        norm((m.dense_true - m.dense_approx) * x) / norm(m.dense_true * x)
    end
    @test eB < eA

    # A COMPLEX bilinear kernel is reproduced EXACTLY by rank-2 linear interpolation, so the H2 far
    # field equals the DIRECTLY kernel-sampled matrix `K[i,j]=kernel(x_i,x_j)` -- now in ALL modes,
    # since the mini's transpose/adjoint is correct (o2i applies transpose(S(transl,recv))).
    mbil = minih2matrix(
        pts; rank=2, minhalfsize=0.008, kernel=(a, b) -> a * b * (1.0 + 0.5im)
    )
    K = mbil.dense_true
    xb = randn(eltype(mbil.A), length(pts))
    @test mbil.dense_approx ≈ K                        # interpolation exact -> Ã == sampled kernel
    @test mbil.A * xb ≈ K * xb                         # forward matches kernel sampling
    @test transpose(mbil.A) * xb ≈ transpose(K) * xb   # transpose
    @test adjoint(mbil.A) * xb ≈ K' * xb               # adjoint (conj: genuinely distinct)

    A = mbil.A
    for α in [false, true]
        for β in [false, true]
            y = randn(eltype(A), size(A, 1))
            y2 = deepcopy(y)

            @test LinearAlgebra.mul!(y, A, x, α, β) ≈ LinearAlgebra.mul!(y2, K, x, α, β)

            w = randn(eltype(A), size(A, 2))
            w2 = deepcopy(w)

            @test LinearAlgebra.mul!(w, transpose(A), y, α, β) ≈
                LinearAlgebra.mul!(w2, transpose(K), y, α, β)

            w2 = deepcopy(w)

            @test LinearAlgebra.mul!(w, adjoint(A), y, α, β) ≈
                LinearAlgebra.mul!(w2, K', y, α, β)
        end
    end
end

@testset "Mini Petrov H2 matrix: all modes vs sampled affine kernel" begin
    # Petrov analogue of the mini Galerkin accuracy oracle, across the SAME Petrov geometry matrix as
    # the fake BlockTree test above. Rank-2 interpolation exactly reproduces affine kernels, so the
    # assembled H2 far field must equal the directly sampled far kernel for balanced, unbalanced,
    # overlapping, disjoint-far, and rectangular/asymmetric test-trial trees.
    u = uniformpts(128)
    b = petrovunb()
    for mode in (H2Trees.AggregateMode(), H2Trees.AggregateTranslateMode())
        for (kname, kernel) in affine_kernel_cases()
            for (name, tp, rp, tv, rv, mhs) in dense_petrov_coordinate_configs()
                mini = minipetrovh2matrix(
                    tp,
                    rp;
                    rank=2,
                    kernel=kernel,
                    minhalfsize=mhs,
                    testminvalues=tv,
                    trialminvalues=rv,
                    aggregationmode=mode,
                )
                A = mini.A
                K = mini.dense_true
                x = randn(eltype(A), size(A, 2))
                w = randn(eltype(A), size(A, 1))
                @testset "$(typeof(mode)) / $kname / $name" begin
                    @test mini.dense_approx ≈ K
                    @test denseh2matrix(A) ≈ K
                    @test H2Factory.farmul!(similar(x, size(A, 1)), A, x) ≈ K * x
                    @test H2Factory.farmul!(similar(w, size(A, 2)), transpose(A), w) ≈
                        transpose(K) * w
                    @test H2Factory.farmul!(similar(w, size(A, 2)), adjoint(A), w) ≈ K' * w
                    @test ref_farmul(A, x) ≈ K * x

                    for α in [false, true]
                        for β in [false, true]
                            y = randn(eltype(A), size(A, 1))
                            y2 = deepcopy(y)

                            @test LinearAlgebra.mul!(y, A, x, α, β) ≈
                                LinearAlgebra.mul!(y2, K, x, α, β)

                            w = randn(eltype(A), size(A, 2))
                            w2 = deepcopy(w)

                            @test LinearAlgebra.mul!(w, transpose(A), y, α, β) ≈
                                LinearAlgebra.mul!(w2, transpose(K), y, α, β)

                            w2 = deepcopy(w)

                            @test LinearAlgebra.mul!(w, adjoint(A), y, α, β) ≈
                                LinearAlgebra.mul!(w2, K', y, α, β)
                        end
                    end
                end
            end
        end
    end

    # Approximation quality on a genuinely non-symmetric Petrov geometry.
    smooth = H2FactoryTestUtils.COMPLEXKERNEL
    e2 = let m = minipetrovh2matrix(u, 0.5 .+ u; rank=2, kernel=smooth, minhalfsize=0.006)
        xx = randn(eltype(m.A), size(m.A, 2))
        norm((m.dense_true - m.dense_approx) * xx) / norm(m.dense_true * xx)
    end
    e3 = let m = minipetrovh2matrix(u, 0.5 .+ u; rank=3, kernel=smooth, minhalfsize=0.006)
        xx = randn(eltype(m.A), size(m.A, 2))
        norm((m.dense_true - m.dense_approx) * xx) / norm(m.dense_true * xx)
    end
    @test e3 < e2
end

@testset "Aggregation makes sense (per-node moments == reference)" begin
    tree, N = b4tree()
    T = ComplexF64
    A = testfakegalerkinmap(tree; T=T, i2o=scalartranslator(2.0 - 1.0im))
    x = randn(T, N)
    H2Factory.aggregate!(A, x)
    refagg = ref_aggregate(A, x)
    for (node, refm) in refagg
        @test H2Factory.trialmoments(A)[node][1] ≈ refm
    end
end

@testset "Aggregation closed form (identity-sum -> subtree sums, incl. top)" begin
    # With identity I2I + all-ones rank-1 leaf basis, aggregation is pure summation, so EVERY
    # aggregated node's moment has the CLOSED FORM  sum(x over that node's subtree DOFs)  --
    # checked analytically, not against another implementation. In particular the top (highest)
    # aggregation nodes hold per-cluster sums that together cover all of x.
    tree, N = b4tree()
    T = ComplexF64
    A = testfakegalerkinmap(tree; T=T, i2o=scalartranslator(1.0 + 0.0im))
    x = randn(T, N)
    H2Factory.aggregate!(A, x)
    plan = H2Factory.trialaggregationplan(A)
    for level in H2Trees.aggregationlevels(plan)
        for node in H2Trees.aggregationnodes(plan, level)
            @test H2Factory.trialmoments(A)[node][1][1] ≈ sum(x[H2Trees.values(tree, node)])
        end
    end
    # `aggregationlevels` runs leaf-to-root, so `last` is the top; its nodes partition all DOFs
    # (two clusters here), so their moments sum to sum(x) -- the closed form "at the top".
    toplevel = last(collect(H2Trees.aggregationlevels(plan)))
    topnodes = collect(H2Trees.aggregationnodes(plan, toplevel))
    @test sum(H2Factory.trialmoments(A)[n][1][1] for n in topnodes) ≈ sum(x)
end

@testset "Disaggregation makes sense (isolated, injected moments)" begin
    # Drive disaggregation from a known set of trial (aggregation) moments and compare the real
    # `disaggregate!` output with the reference translate+disaggregate of the same moments.
    pts = uniformpts(128)
    mini = minih2matrix(pts; rank=2, minhalfsize=0.03)
    A = mini.A
    x = randn(eltype(A), length(pts))   # match the map's element type (complex kernel by default)
    # Populate aggregation moments via the real aggregate!, then run the real disaggregate!.
    H2Factory.aggregate!(A, x)
    y = zeros(eltype(A), length(pts))
    H2Factory.disaggregate!(y, A)
    # Reference: same aggregation moments -> translate -> disaggregate.
    agg = ref_aggregate(A, x)
    yref = ref_disaggregate(A, ref_translate(A, agg))
    @test y ≈ yref
end

@testset "Transpose/adjoint duality (reference-free)" begin
    # Scalar fake on the deep (asymmetric-`F`) tree with a non-real coupling. The defining identity
    # of the adjoint over ℂ is ⟨u, A x⟩ == ⟨A' u, x⟩ (adjoint, NOT transpose). (The mini's own
    # transpose/adjoint is now also correct and covered against its complete matrix above.)
    T = ComplexF64
    A = testfakegalerkinmap(
        H2Trees.buildtree(
            [SVector(c, 0.0, 0.0) for c in uniformpts(256)];
            builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.02, minvalues=0, root=1),
        );
        T=T,
        i2o=scalartranslator(1.5 + 0.5im),
    )
    x = randn(T, 256)
    u = randn(T, 256)
    yx = similar(x)
    yu = similar(u)
    lhs = dot(u, H2Factory.farmul!(yx, A, x))
    rhs = dot(H2Factory.farmul!(yu, adjoint(A), u), x)
    @test lhs ≈ rhs
end

@testset "Structural correctness (spy translator)" begin
    tree, N = b4tree()
    T = ComplexF64
    i2ospy = spytranslator(scalartranslator(2.0 - 1.0im))
    i2ispy = spytranslator(identitytranslator())
    A = testfakegalerkinmap(tree; T=T, i2i=i2ispy, o2o=identitytranslator(), i2o=i2ospy)
    x = randn(T, N)
    y = similar(x)
    H2Factory.farmul!(y, A, x)

    farset = Set(ref_farpairs(A))
    i2orecs = filter(r -> r.role == :i2o, records(i2ospy))
    # Every far pair translated exactly once; no near pair ever translated.
    @test length(i2orecs) == length(farset)
    @test Set((r.a, r.b) for r in i2orecs) == farset

    # Every non-leaf aggregated strictly after all its children: for each i2i transfer record
    # (parent, child), every record where `child` acts as the parent appears earlier in the log.
    transfers = filter(r -> r.role == :transfer, records(i2ispy))
    for (idx, r) in enumerate(transfers)
        childasparent = findall(s -> s.a == r.b, transfers)
        @test all(<(idx), childasparent)
    end
end

@testset "Determinism (real traversal == single-threaded reference)" begin
    # The reference is single-threaded; agreement with the (possibly multithreaded) real traversal
    # is the per-run determinism guard. Repeated calls must also be bit-stable.
    pts = uniformpts(512)
    mini = minih2matrix(pts; rank=2, minhalfsize=0.02)
    A = mini.A
    x = randn(eltype(A), length(pts))   # match the map's element type (complex kernel by default)
    y1 = similar(x)
    y2 = similar(x)
    H2Factory.farmul!(y1, A, x)
    H2Factory.farmul!(y2, A, x)
    @test y1 == y2
    @test y1 ≈ ref_farmul(A, x)
end

end # module TestComponentHarness
