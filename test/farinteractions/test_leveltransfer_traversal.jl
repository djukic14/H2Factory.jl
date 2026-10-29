module TestLevelTransferTraversal

using Test
using LinearAlgebra
using Random
using H2Factory
using H2Trees
using H2FactoryTestUtils

# Traversal-level proof for the level-transfer refactor: a real 1D interpolation H2 matrix driven
# through H2Factory's actual aggregate!/translate/disaggregate traversal, with the I2I/O2O level
# transfers supplied as H2Factory `LevelTranslator`s (sector-indexed matrices + SectorSelector) instead
# of the callable mini translators.
#
# The traversal now routes level transfers through `leveltranslation!(..., direction, ...)`: aggregation
# applies the side's I2I forward, disaggregation applies `transpose(I2I)`/`adjoint(I2I)` at the transfer
# boundary, and the former `conjmomentTransposeAdjoint!` moment-conjugation convention is gone. So the
# forward/transpose/adjoint results match the reference dense matrix, computed by a twin callable
# `minih2matrix` over identical geometry. (These assertions were written test-first as `@test_broken`
# -- they errored while `LevelTranslator` was not a traversal translator -- and flipped to `@test` with
# the switch.)
#
# The transfers are the real, NON-SYMMETRIC interpolation weights (`transpose(E)` for I2I, `E` for
# O2O), and the default kernel is COMPLEX, so a wrong transpose/adjoint (or a leftover moment
# conjugation) cannot hide: transpose != adjoint for the complex coupling, and neither equals forward.
#
# COMPLEX LEVEL TRANSFERS: intentionally not exercised here. The interpolation transfers are real, so
# transpose==adjoint at the transfer boundary; the future adjoint-vs-transpose shape for a genuinely
# complex level transfer is kept visible at the PROTOCOL level (`test_leveltranslator.jl`'s complex
# applytransfer! tests), where it belongs -- a complex-transfer traversal fixture would need a dense
# reference the real-transfer twin mini cannot provide, so it is a documented future follow-up, not a
# gap in this slice.

# A genuinely DISCRIMINATING kernel: the default COMPLEXKERNEL is conjugate-symmetric
# (K(y,x) = conj(K(x,y))), which makes the far matrix Hermitian, so adjoint(F) == F and the adjoint
# matvec is indistinguishable from forward. This asymmetric phase breaks both symmetries: F is neither
# symmetric (transpose has teeth) nor Hermitian (adjoint has teeth, and differs from transpose).
# [[feedback_symmetric_masking]]
kern(x, y) = exp(im * (2x - y)) / (1 + (x - y)^2)

@testset "LevelTranslator far-field matvec: forward/transpose/adjoint vs dense reference" begin
    pts = collect(range(0.3, 2.7; length=64))
    m = minih2matrix_leveltranslator(pts; rank=2, kernel=kern, minhalfsize=0.03)
    A = m.A
    F = m.dense_approx
    rng = MersenneTwister(30)
    x = randn(rng, eltype(A), size(F, 2))

    # Sanity: the reference far matrix is non-trivial, non-symmetric, and non-Hermitian (so transpose
    # AND adjoint both have teeth and differ from each other and from forward).
    @test norm(F) > 0
    @test !(F ≈ transpose(F))
    @test !(F ≈ adjoint(F))

    # Sanity: the LEVEL TRANSFERS themselves are non-symmetric, so this slice's transpose/adjoint
    # direction is actually being exercised at the transfer boundary (not masked by symmetric E).
    i2i = H2Factory.triali2itranslator(A)
    edge = first(
        (H2Trees.parent(m.tree, n), n) for
        n in H2Trees.DepthFirstIterator(m.tree, H2Trees.root(m.tree)) if
        n != H2Trees.root(m.tree)
    )
    Trep = H2Factory.leveltransfer(i2i, m.tree, edge[1], edge[2])
    @test !(Trep ≈ transpose(Trep))

    # The far-field matvec matches the dense reference in every orientation: aggregation applies the
    # side's I2I forward, disaggregation applies transpose(I2I)/adjoint(I2I) at the transfer boundary.
    @test H2Factory.farmul!(similar(x), A, x) ≈ F * x                    # forward
    @test H2Factory.farmul!(similar(x), transpose(A), x) ≈ transpose(F) * x  # transpose
    @test H2Factory.farmul!(similar(x), adjoint(A), x) ≈ adjoint(F) * x   # adjoint
end

@testset "O2O is not semantic: poisoned O2O slot, result still correct via transpose(I2I)" begin
    # The demotion test. The O2O slot's matrices are NaN; because disaggregation derives its transfer
    # from `transpose(I2I)` and never reads the O2O slot, the result stays NaN-free. A traversal that
    # used the O2O slot semantically (or a regression that reintroduced it) would produce NaN here.
    pts = collect(range(0.3, 2.7; length=64))
    m = minih2matrix_leveltranslator(
        pts; rank=2, kernel=kern, minhalfsize=0.03, poisono2o=true
    )
    A = m.A
    F = m.dense_approx
    rng = MersenneTwister(31)
    x = randn(rng, eltype(A), size(F, 2))

    # Forward and transpose/adjoint all traverse disaggregation (O2O role), so all three would go NaN
    # if the O2O slot were consulted. Broken until the switch derives disagg from transpose(I2I).
    @test H2Factory.farmul!(similar(x), A, x) ≈ F * x                    # forward
    @test H2Factory.farmul!(similar(x), transpose(A), x) ≈ transpose(F) * x  # transpose
    @test H2Factory.farmul!(similar(x), adjoint(A), x) ≈ adjoint(F) * x   # adjoint
end

end # module TestLevelTransferTraversal
