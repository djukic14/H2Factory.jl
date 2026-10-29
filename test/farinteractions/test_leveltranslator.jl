module TestLevelTranslator

using Test
using Random
using LinearAlgebra
using StaticArrays
using H2Factory
using H2Trees

# Unit + direction tests for the generic level-transfer protocol (SectorSelector / LevelTranslator /
# applytransfer! / leveltranslation!). These pin the two things the protocol must own so the traversal
# no longer has to: (1) how a tree edge maps to a stored transfer key (the selector), and (2) the
# forward/transpose/adjoint math at the transfer boundary (applytransfer!). All matrices here are
# deliberately NON-symmetric and (in the direction tests) complex, so a wrong transpose/adjoint cannot
# masquerade as a right one.

# A moment whose single array is a length-n vector, filled with `v`.
mkm(v) = (m=H2Factory.Moment1D{ComplexF64}((length(v),)); m[1].=v; m)
ctx() = H2Factory.TranslationContext(H2Factory.SerialScheduler(), nothing)

# A genuinely subdivided 3D octree: random points over a wide box with a small `minhalfsize`, so it
# splits several levels deep and populates many sectors at levels 1 and 2. (A handful of corner points
# with a large minhalfsize never subdivides -- the root stays childless and every sector loop is
# vacuous.)
function octree()
    rng = MersenneTwister(20)
    pts = [SVector(8 .* rand(rng, 3) .- 4 ...) for _ in 1:400]
    return H2Trees.buildtree(
        pts; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.5, root=1, minvalues=1)
    )
end
const TREE = octree()

# Every genuine (parent, child) edge. `DepthFirstIterator` yields the start node itself, so the root
# is filtered out -- without that, one "edge" would be `(parent(root), root)`, which passes only
# because sector lookup ignores the parent, blessing a weaker contract than "acts on real tree edges".
edges(tree) = [
    (H2Trees.parent(tree, n), n) for
    n in H2Trees.DepthFirstIterator(tree, H2Trees.root(tree)) if n != H2Trees.root(tree)
]
levelspresent(tree) = sort(unique(H2Trees.level(tree, c) for (_, c) in edges(tree)))

# A deliberately BROKEN selector for the mutation test below: it resolves a valid-but-WRONG sector key
# (rotated by half the sector count, so the key is always in range yet never the child's real sector).
# It models "the traversal picked the wrong sector" and exists only to prove the fake harness catches
# that class of bug -- it is never wired into a real map. Rotating by `2^D ÷ 2` guarantees the mutated
# key differs from the correct one for every sector (the shift is nonzero mod `2^D`).
struct MisroutingSelector{D} <: H2Factory.AbstractTransferSelector end
function H2Factory.transferkey(
    ::MisroutingSelector{D}, tree, parent::Int, child::Int
) where {D}
    return mod1(H2Trees.sector(tree, child) + 1 + (2^D ÷ 2), 2^D)
end

@testset "SectorSelector: nsectors = 2^D" begin
    @test H2Factory.nsectors(H2Factory.SectorSelector{1}()) == 2
    @test H2Factory.nsectors(H2Factory.SectorSelector{2}()) == 4
    @test H2Factory.nsectors(H2Factory.SectorSelector{3}()) == 8
end

@testset "SectorSelector: bad type parameter fails at construction" begin
    # A public protocol type should reject a nonsensical `D` when constructed, not deep inside a later
    # `nsectors`/`transferkey` call.
    @test_throws ArgumentError H2Factory.SectorSelector{:bad}()
    @test_throws ArgumentError H2Factory.SectorSelector{-1}()
    @test_throws ArgumentError H2Factory.SectorSelector{2.5}()
    @test H2Factory.SectorSelector{0}() isa H2Factory.SectorSelector{0}   # 0 is valid (2^0 = 1)
end

@testset "transferkey returns the child sector (1-based) and bounds-checks" begin
    # `H2Trees.sector` is 0-based; the selector must return it 1-based (the stored transfer vectors
    # are 1-based). The tree is genuinely subdivided, so this loop is non-vacuous.
    sel = H2Factory.SectorSelector{3}()
    @test all(child != H2Trees.root(TREE) for (_, child) in edges(TREE))  # real edges only
    ran = 0
    for (parent, child) in edges(TREE)
        @test H2Factory.transferkey(sel, TREE, parent, child) ==
            H2Trees.sector(TREE, child) + 1
        @test 1 <= H2Factory.transferkey(sel, TREE, parent, child) <= 8
        ran += 1
    end
    @test ran > 8   # guard against a vacuous (non-subdividing) tree
end

@testset "SectorSelector lookup depends only on the child (the real contract)" begin
    # The sector-indexed key is a property of the child alone; the `parent` argument is positional (it
    # exists for future selectors that key on the edge). Pin that: any parent value yields the same key.
    sel = H2Factory.SectorSelector{3}()
    for (parent, child) in edges(TREE)
        key = H2Factory.transferkey(sel, TREE, parent, child)
        for bogusparent in (parent, H2Trees.root(TREE), parent + 1)
            @test H2Factory.transferkey(sel, TREE, bogusparent, child) == key
        end
    end
end

@testset "transferkey: out-of-range sector errors under bounds checking" begin
    # A SectorSelector{1} expects sectors in 0:1 (key 1:2). Feeding it a node from a 3D tree (sectors
    # up to 7) must throw rather than silently index out of a 2-element transfer vector.
    sel1 = H2Factory.SectorSelector{1}()
    highedge = findfirst(e -> H2Trees.sector(TREE, e[2]) > 1, edges(TREE))
    @test highedge !== nothing
    parent, child = edges(TREE)[highedge]
    @test_throws BoundsError H2Factory.transferkey(sel1, TREE, parent, child)
end

@testset "LevelTranslator resolves the same matrix as a direct sector/level lookup" begin
    # Distinct matrices per (level, sector) so a wrong key is visible, for every level present.
    levels = levelspresent(TREE)
    rng = MersenneTwister(21)
    mats = Dict(lvl => [randn(rng, ComplexF64, 3, 3) for _ in 1:8] for lvl in levels)
    sel = H2Factory.SectorSelector{3}()
    translator = H2Factory.LevelTranslator(mats, sel)

    for (parent, child) in edges(TREE)
        lvl = H2Trees.level(TREE, child)
        expected = mats[lvl][H2Trees.sector(TREE, child) + 1]
        @test H2Factory.leveltransfer(translator, TREE, parent, child) === expected
    end
end

@testset "applytransfer!: forward=T, transpose=Tᵀ, adjoint=Tᴴ (non-symmetric, complex)" begin
    # A non-symmetric, complex T so transpose(T), adjoint(T), and T are three genuinely different
    # linear maps -- the whole point of making direction explicit at the transfer boundary.
    T = ComplexF64[1+2im 3-1im; 0 4+0im]      # upper-triangular, non-symmetric, complex
    x = ComplexF64[5 - 1im, 2 + 3im]

    for (dir, Tapplied) in (
        (H2Factory.ForwardTranslation(), T),
        (H2Factory.TransposeTranslation(), transpose(T)),
        (H2Factory.AdjointTranslation(), adjoint(T)),
    )
        # Overwrite form.
        out = mkm(zeros(ComplexF64, 2))
        H2Factory.applytransfer!(out, mkm(x), T, dir, H2Factory.Overwrite(), ctx())
        @test out[1] ≈ Tapplied * x

        # Accumulate form (β=true): adds onto the prefilled target.
        pre = ComplexF64[7 + 0im, -3 + 2im]
        outacc = mkm(copy(pre))
        H2Factory.applytransfer!(outacc, mkm(x), T, dir, H2Factory.Accumulate(), ctx())
        @test outacc[1] ≈ pre .+ Tapplied * x
    end
end

@testset "applytransfer!: transpose vs adjoint differ for complex T, agree for real T" begin
    Tc = ComplexF64[1+2im 3-1im; 0 4+1im]     # complex, non-symmetric
    x = ComplexF64[5 - 1im, 2 + 3im]

    outT = mkm(zeros(ComplexF64, 2))
    outA = mkm(zeros(ComplexF64, 2))
    H2Factory.applytransfer!(
        outT, mkm(x), Tc, H2Factory.TransposeTranslation(), H2Factory.Overwrite(), ctx()
    )
    H2Factory.applytransfer!(
        outA, mkm(x), Tc, H2Factory.AdjointTranslation(), H2Factory.Overwrite(), ctx()
    )
    @test !(outT[1] ≈ outA[1])                # complex: transpose != adjoint
    @test outA[1] ≈ conj(transpose(Tc)) * x   # adjoint = conj(transpose)

    # Real T: transpose == adjoint (the currently-supported level-transfer case).
    Tr = ComplexF64[1 3; 0 4]                 # real values in a complex array, non-symmetric
    outTr = mkm(zeros(ComplexF64, 2))
    outAr = mkm(zeros(ComplexF64, 2))
    H2Factory.applytransfer!(
        outTr, mkm(x), Tr, H2Factory.TransposeTranslation(), H2Factory.Overwrite(), ctx()
    )
    H2Factory.applytransfer!(
        outAr, mkm(x), Tr, H2Factory.AdjointTranslation(), H2Factory.Overwrite(), ctx()
    )
    @test outTr[1] ≈ outAr[1]                 # real: transpose == adjoint
end

@testset "leveltranslation!: end-to-end edge -> keyed transfer -> direction math" begin
    # Distinct per-sector matrices; drive real tree edges through the full entry point and confirm each
    # equals the hand-resolved matrix applied in the requested direction.
    levels = levelspresent(TREE)
    rng = MersenneTwister(22)
    mats = Dict(lvl => [randn(rng, ComplexF64, 2, 2) for _ in 1:8] for lvl in levels)
    sel = H2Factory.SectorSelector{3}()
    translator = H2Factory.LevelTranslator(mats, sel)

    x = ComplexF64[1 - 1im, 2 + 2im]
    for (parent, child) in edges(TREE)
        lvl = H2Trees.level(TREE, child)
        T = mats[lvl][H2Trees.sector(TREE, child) + 1]
        for (dir, Tapplied) in (
            (H2Factory.ForwardTranslation(), T),
            (H2Factory.TransposeTranslation(), transpose(T)),
            (H2Factory.AdjointTranslation(), adjoint(T)),
        )
            out = mkm(zeros(ComplexF64, 2))
            H2Factory.leveltranslation!(
                out,
                mkm(x),
                translator,
                TREE,
                parent,
                child,
                dir,
                H2Factory.Overwrite(),
                ctx(),
            )
            @test out[1] ≈ Tapplied * x
        end
    end
end

@testset "mutation: a mis-routed sector is caught (wrong transfer, wrong result)" begin
    # The plan's fake-harness mutation test for the SECTOR selector: deliberately route the wrong sector
    # and confirm both the structural lookup and the numerical result diverge from the correct routing.
    # This is the discriminating counterpart to the poisoned-O2O (direction) and side-ownership (side)
    # tests -- together they cover the sector/side/direction triple the refactor must not silently mix
    # up. If distinct per-sector transfers could be swapped without the harness noticing, the selector's
    # core guarantee ("the right stored transfer for each edge") would be untestable.
    levels = levelspresent(TREE)
    rng = MersenneTwister(23)
    # Distinct per-(level, sector) matrices, so any wrong key resolves a genuinely different transfer.
    mats = Dict(lvl => [randn(rng, ComplexF64, 2, 2) for _ in 1:8] for lvl in levels)

    correct = H2Factory.LevelTranslator(mats, H2Factory.SectorSelector{3}())
    mutant = H2Factory.LevelTranslator(mats, MisroutingSelector{3}())

    x = ComplexF64[1 - 1im, 2 + 2im]
    checked = 0
    for (parent, child) in edges(TREE)
        Tcorrect = H2Factory.leveltransfer(correct, TREE, parent, child)
        Tmutant = H2Factory.leveltransfer(mutant, TREE, parent, child)
        # Structural: the mis-route resolves a DIFFERENT stored transfer (not just an equal-valued one).
        @test Tcorrect !== Tmutant
        @test !(Tcorrect ≈ Tmutant)

        # Numerical: driven through the same entry point, the mis-route yields a different moment -- so a
        # wrong sector selection cannot pass the harness's numerical checks.
        outc = mkm(zeros(ComplexF64, 2))
        outm = mkm(zeros(ComplexF64, 2))
        H2Factory.leveltranslation!(
            outc,
            mkm(x),
            correct,
            TREE,
            parent,
            child,
            H2Factory.ForwardTranslation(),
            H2Factory.Overwrite(),
            ctx(),
        )
        H2Factory.leveltranslation!(
            outm,
            mkm(x),
            mutant,
            TREE,
            parent,
            child,
            H2Factory.ForwardTranslation(),
            H2Factory.Overwrite(),
            ctx(),
        )
        @test !(outc[1] ≈ outm[1])
        checked += 1
    end
    @test checked > 8   # guard against a vacuous (non-subdividing) tree hiding the mutation
end

end # module TestLevelTranslator
