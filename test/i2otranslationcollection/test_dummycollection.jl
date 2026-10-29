module TestDummyCollection

using Test
using H2Factory
using OhMyThreads
@testset "DummyI2OTranslationCollection" begin
    collection = DummyI2OTranslationCollection()

    in = 1
    out = 2

    # IT PARTICIPATES IN THE PROTOCOL, it does not short-circuit it. A dummy collection is the type
    # with the best excuse to skip the guard -- it stores nothing, so no symmetry tag can possibly be
    # left unconsumed -- which is exactly why it must not be the in-package example of how to skip it.
    # It declares both answers `applycoupling!` asks of every coupling collection and supplies a
    # kernel, like anything else.
    @test H2Factory.assertnosymmetry(collection) === nothing
    @test H2Factory.i2otrait(collection) === H2Factory.DummyTranslation()

    # Declared, not inherited: `which` is what tells those apart, since H2Factory's abstract method
    # errors and an inherited one would therefore not return `nothing` at all.
    @test which(H2Factory.assertnosymmetry, Tuple{typeof(collection)}).sig.parameters[2] ===
        typeof(collection)

    # And it adds NO `applycoupling!` method, so the guard runs for it through H2Factory's single
    # entry rather than being displaced by a convenience override.
    @test !any(
        m -> m.sig.parameters[4] === typeof(collection), methods(H2Factory.applycoupling!)
    )

    # `applycoupling!` is the coupling entry (not exported); the public per-pair entry is `applyi2o!`.
    # This unit test drives the dummy collection directly, so it uses the qualified name.
    #
    # A dummy collection returns `out` untouched in EVERY direction and accumulation mode, which is
    # now one kernel rather than six methods. Checking all six combinations here, because "applies
    # nothing" is the entire contract and a mode that quietly did something else would be invisible.
    context = H2Factory.TranslationContext(SerialScheduler(), nothing)
    for direction in (
        H2Factory.ForwardTranslation(),
        H2Factory.TransposeTranslation(),
        H2Factory.AdjointTranslation(),
    )
        for accumulation in (H2Factory.Overwrite(), H2Factory.Accumulate())
            @test H2Factory.applycoupling!(
                out, in, collection, 1, 2, direction, accumulation, context
            ) == out
        end
    end
end

end # module TestDummyCollection
