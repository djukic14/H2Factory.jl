module TestComponentHarnessSymmetry

using Test
using Random
using LinearAlgebra
using StaticArrays
using FFTW
using OhMyThreads: SerialScheduler
using H2Factory
using H2Trees
using H2FactoryTestUtils

# SYMMETRY-REDUCED FAR-FIELD I2O, THROUGH THE FULL TRAVERSAL.
#
# `test/i2otranslationcollection/test_symmetrycollection.jl` covers what H2Factory STORES: the tags
# reconstruct, and applying a tagged collection through the plain path is refused. It stops there,
# because acting with a symmetry needs an angular representation and H2Factory has none.
#
# This file supplies one -- `H2FactoryTestUtils`' fourteen lattice directions, closed under the cubic
# group, with a diagonal coupling `d_q[j] = f(k̂_j ⋅ q)` -- and so closes the loop the storage tests
# leave open: read the tag, invert it, permute, and get the right matrix. The reduced map's coupling
# comes from one diagonal per ORBIT read through a permutation; the reference `denseh2matrix` reads
# the per-pair diagonals of an unreduced collection built independently. So every
# `farmul! == denseh2matrix` below is the statement that the symmetry convention is right, checked on
# every far pair in forward, transpose and adjoint at once.
#
# Exactly, not approximately: `k̂ ⋅ q` is an integer dot product, so unlike MLFMA (which evaluates the
# same translation operator from ROTATED trigonometric inputs and agrees to ~1e-14) the two paths
# here are bit-for-bit. Where that is what is being asserted the test says `==`, not `≈`.

# A cubic point lattice, not a random cloud: the far-field offsets then land on many DISTINCT orbits
# of the group, which is what gives the permutation tables something to get wrong. A random cloud
# still works but concentrates the offsets on fewer orbits.
#
# `minhalfsize` is what decides how many levels carry translations, and both matter here: with
# `1.0` this tree bottoms out at a single translating level, which is exactly the case a per-level
# indexing mistake survives. At `0.6` there are two, with 316 -> 16 unique translations on each --
# a 19.75x reduction, which is the full cubic group's theoretical ceiling over a level's 316
# far-field offsets (the group has 48 elements; the ceiling is below that because offsets lying on
# a mirror plane or an axis have short orbits).
function latticetree(n=8; minhalfsize=0.6)
    pts = [
        SVector(Float64(i), Float64(j), Float64(k)) for i in 0:(n - 1) for j in 0:(n - 1)
        for k in 0:(n - 1)
    ]
    return H2Trees.buildtree(
        pts; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=minhalfsize, minvalues=0, root=1)
    ),
    length(pts)
end

# The unstructured counterpart: the offsets are the same lattice (a `TwoNTree` is a cubic
# subdivision whatever the points are), but the OCCUPIED offsets are irregular, so which orbits
# appear and how many members each has is no longer uniform. `minhalfsize` is small enough that two
# levels translate; a coarser one leaves the plan empty and `buildplans` refuses it outright.
function cloudtree(N=800; minhalfsize=0.1, seed=13)
    rng = MersenneTwister(seed)
    pts = [SVector{3,Float64}(rand(rng), rand(rng), rand(rng)) for _ in 1:N]
    return H2Trees.buildtree(
        pts; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=minhalfsize, minvalues=0, root=1)
    ),
    length(pts)
end

# How many stored couplings this pair reaches through a genuinely non-identity permutation. Zero here
# would mean every interaction canonicalized to itself and nothing below tests anything.
function npermuted(setup)
    identity = collect(1:(setup.rank))
    return count(keys(setup.reduced.dict)) do key
        return setup.reduced[key].permutation != identity
    end
end

@testset "The reduction actually happens" begin
    # Non-vacuity first. Every assertion further down is satisfied by a "reduction" that stored one
    # translation per direction and permuted by the identity, so the reduction has to be shown to be
    # real before anything is concluded from its correctness.
    for (name, (tree, _)) in ("lattice" => latticetree(), "cloud" => cloudtree())
        setup = symmetryi2osetup(tree, symmetrytranslatingplan(tree))
        @testset "$name" begin
            @test setup.stats.npairs == setup.plainstats.npairs      # same interactions
            @test setup.stats.nunique < setup.plainstats.nunique     # fewer stored translations
            @test npermuted(setup) > 0                               # ... and they are read permuted
            # More than one level carries translations, so a per-level indexing mistake has somewhere
            # to show. (`pairsbylevel` counts entries, not levels -- counting entries is how an
            # earlier version of this assertion passed while testing one level.)
            @test count(>(0), setup.stats.pairsbylevel) >= 2
        end
    end
end

@testset "Reduced store reproduces every pair's own coupling" begin
    # The narrowest statement, before any traversal is involved: what the reduced collection hands
    # back for a pair equals what an unreduced collection built that pair's coupling to be. Bit for
    # bit -- see the header.
    tree, _ = latticetree()
    setup = symmetryi2osetup(tree, symmetrytranslatingplan(tree))
    @test all(key -> collect(setup.reduced[key]) == setup.plain[key], keys(setup.plain))
    @test length(keys(setup.reduced.dict)) == length(keys(setup.plain.dict))
end

@testset "Full traversal: all modes vs the complete matrix" begin
    for (name, (tree, N)) in ("lattice" => latticetree(), "cloud" => cloudtree())
        setup = symmetryi2osetup(tree, symmetrytranslatingplan(tree))
        reduced = fakesymmetrymap(tree; setup=setup, reduced=true)
        plain = fakesymmetrymap(tree; setup=setup, reduced=false)

        F = denseh2matrix(reduced)
        x = randn(MersenneTwister(5), ComplexF64, N)
        @testset "$name" begin
            # The reduced map's I2O reaches each pair's coupling through the permutation; `F` was
            # assembled from the per-pair couplings directly. Agreement in all three directions is
            # the symmetry convention being right.
            @test H2Factory.farmul!(similar(x), reduced, x) ≈ F * x
            @test H2Factory.farmul!(similar(x), transpose(reduced), x) ≈ transpose(F) * x
            @test H2Factory.farmul!(similar(x), adjoint(reduced), x) ≈ F' * x
            @test ref_farmul(reduced, x) ≈ F * x

            # Per-pair, through H2Factory's real single-pair I2O pipeline in all three modes.
            @test checkdensei2o(reduced; n=3) > 0

            # And against the unreduced map built over the SAME couplings: not merely close, equal.
            # A tolerance here would hide exactly the small systematic error a subtly wrong
            # permutation on a near-symmetric orbit produces.
            @test H2Factory.farmul!(similar(x), reduced, x) ==
                H2Factory.farmul!(similar(x), plain, x)
            @test denseh2matrix(plain) == F
        end
    end
end

@testset "The wrong convention is caught" begin
    # THE NEGATIVE CONTROL. `translationsymmetry` returns the symmetry mapping CANONICAL to ACTUAL,
    # and `T(k̂, S d) == T(S⁻¹ k̂, d)`, so the consumer applies the INVERSE. Getting that backwards is
    # the mistake the MLFMA comments warn about, and it is INVISIBLE on the antipodal map, which is
    # its own inverse. Everything above would pass under the wrong convention on a group without
    # chiral elements, so: check the group has some, then check the wrong tables are caught.
    tree, N = latticetree()
    setup = symmetryi2osetup(tree, symmetrytranslatingplan(tree))
    group = H2Factory.translationsymmetrygroup(setup.canonical)

    encountered = unique([
        H2Factory.translationsymmetry(setup.canonical, key[1], key[2]) for
        key in keys(setup.canonical)
    ])
    # At least one interaction carries an element that is NOT its own inverse; otherwise the control
    # below is guaranteed to pass for a reason that has nothing to do with the code.
    @test any(s -> H2Trees.inversesymmetry(s) != s, encountered)
    @test any(s -> H2Trees.inversesymmetry(s) != s, collect(group))

    wrong = symmetryi2osetup(tree, symmetrytranslatingplan(tree); useinverse=false)
    @test any(key -> collect(wrong.reduced[key]) != wrong.plain[key], keys(wrong.plain))

    wrongmap = fakesymmetrymap(tree; setup=wrong, reduced=true)
    F = denseh2matrix(wrongmap)
    x = randn(MersenneTwister(6), ComplexF64, N)
    y = H2Factory.farmul!(similar(x), wrongmap, x)
    @test norm(y - F * x) / norm(F * x) > 1e-3
end

@testset "Policies: a smaller group reduces less, and stays correct" begin
    # The policy is the CONSUMER's choice -- it names the subgroup that consumer's representation can
    # act with exactly -- so all three must work, and a smaller group must store more.
    tree, N = latticetree()
    counts = Int[]
    for policy in (
        H2Trees.OppositeSymmetry(),
        H2Trees.AxisPreservingSymmetry(3),
        H2Trees.FullLatticeSymmetry(),
    )
        setup = symmetryi2osetup(tree, symmetrytranslatingplan(tree); policy=policy)
        push!(counts, setup.stats.nunique)
        A = fakesymmetrymap(tree; setup=setup)
        F = denseh2matrix(A)
        x = randn(MersenneTwister(7), ComplexF64, N)
        @testset "$(nameof(typeof(policy)))" begin
            @test H2Factory.farmul!(similar(x), A, x) ≈ F * x
            @test H2Factory.farmul!(similar(x), adjoint(A), x) ≈ F' * x
            @test all(
                key -> collect(setup.reduced[key]) == setup.plain[key], keys(setup.plain)
            )
        end
    end
    # Strictly decreasing: the three policies are nested, so a larger group can only merge more
    # orbits. Equality would mean a policy bought nothing on this geometry and the sweep is not
    # covering what it claims to.
    @test counts[1] > counts[2] > counts[3]
end

@testset "Petrov (BlockTree): all modes, and the lattice condition on the two roots" begin
    # THE CASE THE GALERKIN FIXTURE CANNOT REACH. One tree means one box grid, so every displacement
    # is a lattice vector automatically. A `BlockTree` has TWO roots, and a lattice symmetry can only
    # be applied when their offset is itself a lattice vector: the stored translation is
    # `halfsize * canonical + offset` while the interaction needs `halfsize * (S canonical) + offset`,
    # and applying `S` to the stored one gives `halfsize * (S canonical) + S * offset`. Those agree
    # only when `S * offset == offset`.
    #
    # This is where the symmetry path was silently WRONG in MLFMA before `NonLatticeTranslationError`
    # existed, and the test that should have caught it was vacuous (one pair, identity symmetry). It
    # is checked here instead of only there because it needs no solver, so it is fast and hermetic.
    lattice(n, offset) = [
        (Float64(i) + offset, Float64(j), Float64(k)) for i in 0:(n - 1) for
        j in 0:(n - 1) for k in 0:(n - 1)
    ]

    # WHICH OFFSETS QUALIFY IS NOT READABLE OFF THE POINTS, and the two admitted cases below are
    # asserted rather than derived for that reason: the builder rounds the root halfsize up to a
    # power-of-two multiple of `minhalfsize`, so 12.0 is admitted while 8.0 -- also an integer, also
    # a multiple of the leaf halfsize 0.6 divided out, and entirely plausible-looking -- is not. Two
    # attempts to derive the rule from the mesh were wrong last time; measurement is the source here.
    for (name, offset) in (("coincident roots", 0.0), ("lattice-offset roots", 12.0))
        setup = petrovsymmetrysetup(lattice(6, 0.0), lattice(6, offset); minhalfsize=0.6)
        reduced = fakepetrovsymmetrymap(
            lattice(6, 0.0), lattice(6, offset); setup=setup, minhalfsize=0.6, reduced=true
        )
        plain = fakepetrovsymmetrymap(
            lattice(6, 0.0), lattice(6, offset); setup=setup, minhalfsize=0.6, reduced=false
        )
        F = denseh2matrix(reduced)
        x = randn(MersenneTwister(11), ComplexF64, size(reduced, 2))
        w = randn(MersenneTwister(12), ComplexF64, size(reduced, 1))
        @testset "$name" begin
            # Non-vacuity FIRST, and specifically here: the Petrov testset that missed the original
            # bug passed while covering a single pair carrying the identity.
            @test setup.stats.nunique < setup.plainstats.nunique
            @test npermuted(setup) > 0
            @test setup.stats.npairs > 100

            # `F` is `Ntest x Ntrial`: aggregation runs on the trial side, disaggregation on the
            # test side, so the transpose and adjoint take a vector of the OTHER length.
            @test H2Factory.farmul!(similar(x, size(reduced, 1)), reduced, x) ≈ F * x
            @test H2Factory.farmul!(similar(w, size(reduced, 2)), transpose(reduced), w) ≈
                transpose(F) * w
            @test H2Factory.farmul!(similar(w, size(reduced, 2)), adjoint(reduced), w) ≈
                F' * w
            @test checkdensei2o(reduced; n=3) > 0

            # Against the unreduced map over the same couplings: equal, not merely close.
            @test H2Factory.farmul!(similar(x, size(reduced, 1)), reduced, x) ==
                H2Factory.farmul!(similar(x, size(plain, 1)), plain, x)
        end
    end

    # And the refusal. Both of these are rejected, which is the point: an offset that LOOKS like a
    # lattice vector is not one, and the machinery says so instead of returning a plausible field.
    for offset in (8.0, 5.37)
        @test_throws H2Trees.NonLatticeTranslationError petrovsymmetrysetup(
            lattice(6, 0.0), lattice(6, offset); minhalfsize=0.6
        )
    end
    # The message names the coordinate and says why, rather than being a bare assertion failure.
    @test_throws "not a multiple of the level halfsize" petrovsymmetrysetup(
        lattice(6, 0.0), lattice(6, 5.37); minhalfsize=0.6
    )

    # A non-lattice block tree is a legitimate map -- it simply cannot be symmetry-reduced. The
    # UNREDUCED Petrov fake over the same geometry must still build and still be correct, otherwise
    # the refusal above would be taking a working configuration away.
    plainonly = fakepetrovmap(
        lattice(4, 0.0),
        lattice(4, 5.37);
        T=ComplexF64,
        rank=3,
        i2o=scalartranslator(2.0 - 1.0im),
        minhalfsize=0.6,
    )
    Fplain = denseh2matrix(plainonly)
    xp = randn(MersenneTwister(13), ComplexF64, size(plainonly, 2))
    @test H2Factory.farmul!(similar(xp, size(plainonly, 1)), plainonly, xp) ≈ Fplain * xp
end

@testset "Toeplitz: an unconsumed tag is refused, and the FFT scope is narrower than it looks" begin
    # WHAT THIS TESTSET IS FOR, restated because an earlier version of it claimed more than it
    # showed. It asserts two things:
    #   1. a collection still CARRYING symmetry tags is refused whatever its translation trait,
    #      because nothing has permuted anything yet; and
    #   2. THIS package's fourteen-direction diagonal coupling cannot be FFT'd and permuted.
    # It does NOT show that the Toeplitz path is incompatible with the reduction. It is not: a
    # Toeplitz coupling stores the DFT of a difference-domain kernel on a REGULAR TENSOR GRID, where
    # the symmetry is an exact index permutation that commutes with the transform. Fourteen
    # arbitrary directions are not a regular grid, which is the whole reason (2) holds here.
    symmetry = H2Trees.LatticeSymmetry((2, 3, 1), (1, -1, 1))     # a chiral element, not an involution
    @test H2Trees.inversesymmetry(symmetry) != symmetry
    images = symmetryfrequencyimages(symmetry, SVector{3,Int}(3, 1, -2))

    # The premise: in the SAMPLE domain the two couplings hold the same values, reordered.
    @test images.samplemultiset ≈ images.imagemultiset
    @test images.sample != images.sampleimage                      # genuinely reordered, not fixed

    # And in the FREQUENCY domain they do not. A permutation preserves the multiset of magnitudes,
    # so a difference here rules out ANY permutation -- for THIS irregular direction set.
    @test !isapprox(images.spectrummultiset, images.spectrumimagemultiset; rtol=1e-8)

    # So the combination must be refused, not silently applied. `assertnosymmetry` dispatches on the
    # collection carrying a symmetry table, NOT on the translation trait, which is what makes this
    # hold for the Toeplitz trait without a second guard. Pinned here because it currently holds by
    # construction rather than by intent, and a later trait-based refactor could lose it.
    tree, _ = latticetree()
    translatingplan = symmetrytranslatingplan(tree)
    toeplitztagged = symmetrytoeplitzcollection(tree, translatingplan)
    @test H2Factory.i2otrait(toeplitztagged) isa H2Factory.ToeplitzI2OTranslationTrait
    @test_throws "symmetry-reduced" H2Factory.assertnosymmetry(toeplitztagged)

    # BOTH accumulation modes, not just the accessor -- accumulate is the one the traversal actually
    # calls. Same coverage the collection-level test gives the diagonal trait, asserted here for the
    # Toeplitz one so a future trait-based refactor cannot lose it for Toeplitz alone while the
    # diagonal case keeps passing.
    pair = first(keys(toeplitztagged))
    buffer = zeros(ComplexF64, 4)
    context = H2Factory.TranslationContext(SerialScheduler(), nothing)
    for accumulation in (H2Factory.Overwrite(), H2Factory.Accumulate())
        @test_throws "symmetry-reduced" H2Factory.applycoupling!(
            buffer,
            buffer,
            toeplitztagged,
            pair[1],
            pair[2],
            H2Factory.ForwardTranslation(),
            accumulation,
            context,
        )
    end

    # AND THROUGH THE WRAPPER. `ToeplitzI2OTranslationCollection` wraps an `I2OTranslationCollection`
    # and is itself an `AbstractI2OTranslationCollection`, so before the forwarding methods existed a
    # generic no-op `assertnosymmetry(::AbstractI2OTranslationCollection)` matched it and the tags
    # were silently ignored -- the bare collection refused, the wrapped one did not (measured). That
    # fallback is gone now (see `test_symmetrycollection.jl`, which pins that an undeclared wrapper
    # fails), but this assertion is what says the forwarding is real rather than inherited.
    # Not reachable through today's constructors, which all pass `DirectionInvariance()`, so this is
    # a guard against a future one rather than a live bug.
    wrapped = H2Factory.ToeplitzI2OTranslationCollection(toeplitztagged, tree)
    @test_throws "symmetry-reduced" H2Factory.assertnosymmetry(wrapped)
    @test H2Factory.translationsymmetrygroup(wrapped) ===
        H2Factory.translationsymmetrygroup(toeplitztagged)
    @test H2Factory.translationsymmetry(wrapped, pair[1], pair[2]) ==
        H2Factory.translationsymmetry(toeplitztagged, pair[1], pair[2])

    # The reduced collection, by contrast, must NOT be refused: its permutation is already baked into
    # each stored value, so there is no symmetry left for a consumer to act on. A refusal here would
    # mean the wrapper was carrying a table it had already consumed.
    setup = symmetryi2osetup(tree, translatingplan)
    @test H2Factory.assertnosymmetry(setup.reduced) === nothing
    @test H2Factory.translationsymmetrygroup(setup.reduced) === nothing
    # ... while still reporting the ORBIT count it was built from, rather than the placeholder stats
    # the dict-only constructor would invent.
    @test H2Factory.translationstats(setup.reduced).nunique == setup.stats.nunique
    @test H2Factory.translationstats(setup.reduced).nunique <
        length(keys(setup.reduced.dict))
end

@testset "A tagged collection is refused in every mode, not just forward" begin
    # THE REFUSAL HAS TO COVER THE WHOLE MATVEC, and for a long time it did not. `assertnosymmetry`
    # was called from the two FORWARD entries and from nowhere else, so the four transpose and adjoint
    # entries applied a tagged collection without asking. The tags describe the STORE; transposing a
    # translation does not consume one any more than conjugating it does.
    #
    # There is one entry now, so a direction cannot be missed by omission -- but the matvec-level
    # check below stays, because it verifies the guard is reached through the real traversal in each
    # mode, which a unit test on the entry cannot show.
    #
    # MEASURED BEFORE THE FIX, on this fixture: forward refused, `transpose` and `adjoint` ran and
    # returned relative errors of 1.4297 and 1.3291 against the correctly reduced map. Not a
    # perturbation -- a different answer of the same magnitude, which is exactly the plausible wrong
    # field the guard exists to prevent, and why this needs no new type to reach: one transposed
    # matvec over a collection the forward path already refuses.
    tree, N = latticetree()
    setup = symmetryi2osetup(tree, symmetrytranslatingplan(tree))
    tagged = fakegalerkinmap(
        tree;
        T=ComplexF64,
        rank=setup.rank,
        i2o=fakesymmetryi2o(setup.canonical, setup.couplings),
        leafbasis=symmetryleafbasis,
    )
    x = randn(MersenneTwister(5), ComplexF64, N)
    for A in (tagged, transpose(tagged), adjoint(tagged))
        @test_throws "symmetry-reduced" H2Factory.farmul!(similar(x), A, x)
    end

    # The control, on the same couplings: the CONSUMED collection is refused in no mode and is right
    # in all three. Without this the testset above would pass on a guard that refused everything.
    reduced = fakesymmetrymap(tree; setup=setup, reduced=true)
    F = denseh2matrix(reduced)
    @test H2Factory.farmul!(similar(x), reduced, x) ≈ F * x
    @test H2Factory.farmul!(similar(x), transpose(reduced), x) ≈ transpose(F) * x
    @test H2Factory.farmul!(similar(x), adjoint(reduced), x) ≈ F' * x
end

@testset "Toeplitz fake MAP: still works, and the reduction did not leak into it" begin
    # The refusals above are about a COLLECTION. This is the map, which is what a user builds --
    # and the property that matters is not just "the combination errors" but "the Toeplitz path is
    # untouched". A change to the diagonal reduction that broke the real FFT path would show up
    # here and nowhere else in this file.
    points = [Float64(i) / 8 for i in 0:63]
    kernelfn(z) = 1 / (1 + z^2) + 0.3im * z          # complex, so adjoint != transpose
    tree = H2Trees.buildtree(
        [SVector(p, 0.0, 0.0) for p in points];
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.06, minvalues=0, root=1),
    )
    A = faketoeplitzmap(tree, kernelfn; rank=3)

    F = denseh2matrix(A)
    x = randn(MersenneTwister(21), ComplexF64, size(A, 2))
    @test H2Factory.farmul!(similar(x), A, x) ≈ F * x
    @test H2Factory.farmul!(similar(x), transpose(A), x) ≈ transpose(F) * x
    @test H2Factory.farmul!(similar(x), adjoint(A), x) ≈ F' * x

    # The map's own I2O collection carries NO symmetry -- it was built through the unreduced trait,
    # so applying it is allowed and `assertnosymmetry` must stay silent. If a future change made the
    # symmetry trait the default anywhere in the Toeplitz path, this is what would catch it: the
    # accessors below now see THROUGH the `ToeplitzI2OTranslationCollection` wrapper rather than
    # stopping at it, which is the whole point of the forwarding.
    collection = H2Factory.i2otranslator(A).collection
    @test collection isa H2Factory.ToeplitzI2OTranslationCollection
    @test H2Factory.translationsymmetrygroup(collection) === nothing
    @test H2Factory.assertnosymmetry(collection) === nothing
    @test H2Factory.i2otrait(collection) isa H2Factory.ToeplitzI2OTranslationTrait
end

end # module TestComponentHarnessSymmetry
