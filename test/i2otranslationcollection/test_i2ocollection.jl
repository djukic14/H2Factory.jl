module TestI2OCollection

using Test, InteractiveUtils
using BEAST, CompScienceMeshes, ClusterTrees
using StaticArrays
using LinearAlgebra
using H2Factory, H2FactoryTestUtils
using OhMyThreads
using H2Trees
@testset "I2OTranslationCollection" begin
    struct DummyElementwiseI2OTranslator end

    function (::DummyElementwiseI2OTranslator)(
        receivingnode, translatingnode, translationdirection
    )
        return translationdirection
    end

    struct DummyDenseI2OTranslator end

    function (::DummyDenseI2OTranslator)(
        receivingnode, translatingnode, translationdirection
    )
        return randn(3, 3)
    end

    λ = 1.0

    m = meshsphere(2 * λ, λ / 5)
    m1 = translate!(meshsphere(2 * λ, λ / 5), SVector(10 * λ, 0.0, 0.0))
    m2 = translate!(meshsphere(λ, λ / 5), SVector(0.5 * λ, 0.0, 0.0))
    m3 = translate!(meshsphere(2 * λ, λ / 5), SVector(2 * λ, 0.0, 0.0))

    out = H2Factory.Moment1D(SVector{1}([[0.0, 0.0, 0.0]]))
    in = H2Factory.Moment1D(SVector{1}([[1.0, 2.0, 3.0]]))

    context = H2Factory.TranslationContext(SerialScheduler(), nothing)

    for mx in [m, m1, m2, m3]
        X = raviartthomas(mx)
        for my in [m, m1, m2, m3]
            Y = raviartthomas(my)
            tree = buildtree(
                X,
                Y;
                builder=BlockTreeBuilder(;
                    test=TwoNTreeBuilder(; minhalfsize=λ / 10, minvalues=0),
                    trial=TwoNTreeBuilder(; minhalfsize=λ / 10, minvalues=0),
                ),
            )
            TFIterator = H2Trees.TranslatingNodesIterator(; isnear=H2Trees.isnear())
            aggregatenode = H2Trees.istranslatingnode(; TranslatingNodesIterator=TFIterator)

            testtree = H2Trees.testtree(tree)
            trialtree = H2Trees.trialtree(tree)

            trialaggregateplan = H2Trees.AggregatePlan(
                trialtree,
                H2Trees.PetrovAggregationFunctor(aggregatenode, tree, testtree, trialtree),
            )
            testdisaggregatetranslateplan = H2Trees.DisaggregateTranslatePlan(
                testtree, trialtree, H2Trees.TranslatingNodesIterator
            )

            trialaggregatetranslateplan = H2Trees.AggregateTranslatePlan(
                trialtree, H2Trees.PetrovDisaggregationFunctor(TFIterator, tree, testtree)
            )

            testdisaggregateplan = H2Trees.DisaggregatePlan(
                testtree,
                H2Trees.PetrovAggregationFunctor(aggregatenode, tree, trialtree, testtree),
            )

            aggregationdisaggregationpairs = (
                (trialaggregateplan, testdisaggregatetranslateplan),
                (trialaggregatetranslateplan, testdisaggregateplan),
            )

            i2otranslationtrait = H2Factory.DiagonalTranslation()

            for (aggregationplan, disaggregationplan) in aggregationdisaggregationpairs
                for treetranslationtypes in
                    InteractiveUtils.subtypes(H2Trees.AbstractTranslationTrait)
                    # A symmetry-reduced trait stores one translation per ORBIT, so the stored
                    # value is the canonical direction rather than this pair's own and the
                    # assertions below (which compare against the pair's direction) do not apply
                    # to it. The coupling entry refuses it outright rather than returning the
                    # canonical translation quietly; that refusal and the trait's own contract
                    # are covered in `test_symmetrycollection.jl`.
                    treetranslationtypes <: H2Trees.SymmetryDirectionInvariancePerLevel &&
                        continue
                    treetranslationtrait = treetranslationtypes()
                    for verbose in [true, false]
                        tfcollection, i2ochannel = I2OTranslationCollection(
                            tree,
                            aggregationplan,
                            disaggregationplan,
                            treetranslationtrait,
                            DummyElementwiseI2OTranslator();
                            i2otranslationtrait=i2otranslationtrait,
                            verbose=verbose,
                        )

                        dummycollection = DummyI2OTranslationCollection(
                            tree,
                            aggregationplan,
                            disaggregationplan,
                            treetranslationtrait,
                            DummyElementwiseI2OTranslator();
                            i2otranslationtrait=i2otranslationtrait,
                            verbose=verbose,
                        )

                        for (i, k) in enumerate(keys(tfcollection.dict))
                            out = H2Factory.Moment1D(SVector{1}([[0.0, 0.0, 0.0]]))

                            direction =
                                H2Trees.center(H2Trees.testtree(tree), k[1]) -
                                H2Trees.center(H2Trees.trialtree(tree), k[2])
                            @test tfcollection[k] ≈ direction

                            # The two accumulation modes the protocol has. These replace five
                            # `(α, β)` calls, of which only three ever asserted anything: the
                            # `(false, true)` and `(false, false)` cases compared `out[1]` against
                            # the return value, which IS `out` mutated in place, so they held
                            # whatever the kernel did. `(true, false)` is the same overwrite as the
                            # no-α/β form. The one remaining pair is the real contract, and the
                            # traversal only ever asked for these two.
                            @test H2Factory.applycoupling!(
                                out,
                                in,
                                tfcollection,
                                k[1],
                                k[2],
                                H2Factory.ForwardTranslation(),
                                H2Factory.Overwrite(),
                                context,
                            )[1] ≈ direction .* in[1]

                            @test direction .* in[1] + out[1] ≈ H2Factory.applycoupling!(
                                out,
                                in,
                                tfcollection,
                                k[1],
                                k[2],
                                H2Factory.ForwardTranslation(),
                                H2Factory.Accumulate(),
                                context,
                            )[1]
                            i > 50 && break
                        end

                        @test H2Factory.i2otrait(tfcollection) == i2otranslationtrait
                    end
                end
            end

            i2otranslationtrait = H2Factory.DenseTranslation()

            for (aggregationplan, disaggregationplan) in aggregationdisaggregationpairs
                for treetranslationtypes in
                    InteractiveUtils.subtypes(H2Trees.AbstractTranslationTrait)
                    # A symmetry-reduced trait stores one translation per ORBIT, so the stored
                    # value is the canonical direction rather than this pair's own and the
                    # assertions below (which compare against the pair's direction) do not apply
                    # to it. The coupling entry refuses it outright rather than returning the
                    # canonical translation quietly; that refusal and the trait's own contract
                    # are covered in `test_symmetrycollection.jl`.
                    treetranslationtypes <: H2Trees.SymmetryDirectionInvariancePerLevel &&
                        continue
                    treetranslationtrait = treetranslationtypes()
                    for verbose in [true, false]
                        tfcollection, i2ochannel = I2OTranslationCollection(
                            tree,
                            aggregationplan,
                            disaggregationplan,
                            treetranslationtrait,
                            DummyDenseI2OTranslator();
                            i2otranslationtrait=i2otranslationtrait,
                            verbose=verbose,
                        )
                        dummycollection = DummyI2OTranslationCollection(
                            tree,
                            aggregationplan,
                            disaggregationplan,
                            treetranslationtrait,
                            DummyDenseI2OTranslator();
                            i2otranslationtrait=i2otranslationtrait,
                            verbose=verbose,
                        )
                        for (i, k) in enumerate(keys(tfcollection.dict))
                            # See the elementwise testset above for why five `(α, β)` calls become
                            # these two modes.
                            @test H2Factory.applycoupling!(
                                out,
                                in,
                                tfcollection,
                                k[1],
                                k[2],
                                H2Factory.ForwardTranslation(),
                                H2Factory.Overwrite(),
                                context,
                            )[1] ≈ tfcollection[k] * in[1]

                            @test tfcollection[k] * in[1] + out[1] ≈
                                H2Factory.applycoupling!(
                                out,
                                in,
                                tfcollection,
                                k[1],
                                k[2],
                                H2Factory.ForwardTranslation(),
                                H2Factory.Accumulate(),
                                context,
                            )[1]
                            i > 50 && break
                        end

                        @test H2Factory.i2otrait(tfcollection) == i2otranslationtrait
                    end
                end
            end
        end
    end
end

end # module TestI2OCollection
