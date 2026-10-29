module TestPublicAPI

using Test
using H2Factory

# H2FACTORY'S API MODEL: TWO INDEPENDENT AXES.
#
# A first attempt sorted every exported name into one audience and it produced artificial answers,
# because the two questions being asked are not the same question:
#
#   STABILITY  what may we change, and how is the name reached?
#   AUDIENCE   who is it for?
#
# These are orthogonal. A concrete type can be a user's configuration choice AND a legitimate
# dispatch target downstream; forcing it to be only one invents a distinction the code does not
# have. So a name carries one stability class and one or more audiences.
#
# STABILITY
#   PUBLIC              exported and promised.
#   QUALIFIED_EXTENSION NOT exported, but supported extension surface reached as `H2Factory.name`.
#                       Being unexported is a statement about reach, not about support: these are
#                       the hooks downstream packages implement, and breaking one breaks them.
#   INTERNAL            exported today, promised never. Reachable, unsupported.
#
# There used to be a fourth class, REEXPORT, for the `H2Trees` module binding and eleven H2Trees
# names H2Factory re-exported. The class existed to say "not ours to promise"; the names are no
# longer exported at all, so it is gone rather than empty. Consumers declare H2Trees themselves.
#
# AUDIENCE
#   USERFACING          you call it to build and run an operator, or you select it as policy.
#   IMPLEMENTERFACING   you implement, subtype or dispatch on it when extending H2Factory.
#   (a name in both sets is BOTH; that overlap is the point of splitting the axes.)
#
# NOTHING HERE CHANGES AT RUNTIME. Every exported name stays exported and every unexported one stays
# unexported. This records intent so later slices act on a decision. In particular INTERNAL is not a
# removal list: a name is there because it is unpromised, not because it is unused.
#
# QUALIFIED_EXTENSION is what #179's `Interfaces` namespace is built from, together with the
# IMPLEMENTERFACING half of PUBLIC. The exported surface alone was never enough: the most important
# extension points in this package (`buildmoments`, `buildi2ochannel`, `applycoupling!`,
# `modulate_in!`) are not exported at all.

# --- STABILITY: PUBLIC ---------------------------------------------------------------------------

const PUBLIC = [
    # Maps, cores and workspaces. Downstream packages subtype and dispatch on these; a user reaches
    # an operator through the constructors, not by naming the types.
    :GalerkinH2Map,
    :PetrovGalerkinH2Map,
    :AbstractH2Core,
    :GalerkinH2Core,
    :PetrovGalerkinH2Core,
    :AbstractH2Workspace,
    :GalerkinH2Workspace,
    :PetrovGalerkinH2Workspace,
    :H2Workspace,
    :similarworkspace,
    :WorkspaceBuffers,
    :VectorBuffers,
    :SidedChannels,
    # The workspace ownership guard: machinery for implementers of new top-level operations.
    :acquireworkspace!,
    :releaseworkspace!,
    :withworkspace,
    :ConcurrentWorkspaceError,
    # Execution entry points.
    :nearmul,
    :nearmul!,
    :nearinteractions,
    :farmul,
    :farmul!,
    :farinteractions,
    :operator,
    # Execution scheduling: documented user configuration.
    :ExecutionScheduling,
    :defaultexecutionscheduling,
    :serialexecutionscheduling,
    :executionscheduling,
    # Diagnostics.
    :Diagnostics,
    :checkmomentaliasing,
    :MomentAliasingReport,
    # The symmetry reduction: what you build and the policy objects you pick, never its parts.
    :SymmetryReducedI2OTranslationCollection,
    :symmetryreducedcollection,
    :symmetryreducedtoeplitzcollection,
    :symmetryreduceddensecollection,
    :VectorField,
    :TrueTensor,
    :PseudoTensor,
    # Moment type generation and its abstracts.
    Symbol("@createmoment"),
    Symbol("@createmomentcollection"),
    Symbol("@createlinked"),
    Symbol("@linkmomenttocollection"),
    Symbol("@createtoeplitzmoment"),
    Symbol("@createlinkedtoeplitz"),
    :AbstractMoment,
    :AbstractMomentCollection,
    # The store protocol.
    :storetrait,
    :buildstorer,
    :StoreAsIs,
    # Moment assembly, provided by the H2BEAST extension.
    :assemblemoments!,
    :superspace,
    :assemblymap,
    # Near-field: a user SELECTS a strategy; an implementer writes an assembler. Same group in the
    # export block, different audiences, which is why the audience axis exists.
    :AbstractInteractionStrategy,
    :InteractionStrategy,
    :AbstractNearInteractionsAssembler,
    :AbstractNearInteractionsStrategy,
    :PassThroughNearInteractionsAssembler,
    :PassThroughNearInteractionsStrategy,
    :DummyNearInteractionsAssembler,
    :DummyNearInteractionsStrategy,
    :RandomNearInteractionsAssembler,
    :SparseBEASTNearInteractionsAssembler,
    :BlockBEASTNearInteractionsAssembler,
    :BlockBEASTNearInteractionsStrategy,
    :nearcriterion,
    :translatingnodesiterator,
    :aggregatenode,
    :aggregationmode,
    :buildnearinteractionsassembler,
    :assemblenearinteractions,
    :assembleplans,
    # Translation collections: reference implementations and dispatch vocabulary, not end-user
    # configuration. Concrete does not imply user-facing.
    :I2OTranslationCollection,
    :DummyI2OTranslationCollection,
    :I2ITranslationCollection,
    :DenseI2I,
    :SeparableI2I,
    :i2itrait,
    :AbstractI2OTranslationTrait,
    :ToeplitzI2OTranslationTrait,
    # Leaf multiplication, and the chunk count a downstream buffer builder must satisfy.
    :elementwisemul!,
    :nchunks,
    # BEAST operator wrappers. Their methods arrive with the H2BEAST extension, which is a property
    # of every H2BEAST-backed name rather than a reason to classify them apart.
    Symbol("@efie"),
    Symbol("@mw3Dhypersingular"),
    Symbol("@mw3Dweaklysingular"),
    Symbol("@mw3Ddoublelayer"),
    Symbol("@hh3Ddoublelayer"),
    Symbol("@hh3Ddoublelayer_transposed"),
]

# --- STABILITY: QUALIFIED_EXTENSION (unexported, still supported) ---------------------------------

const QUALIFIED_EXTENSION = [
    # Workspace construction hooks.
    :buildmoments,
    :buildi2ochannel,
    # The I2O protocol: coupling, modulation and their composition. `applyi2o!` and `applycoupling!`
    # are ORCHESTRATION, not extension points -- stable qualified names a caller may use, but not
    # declared in `Interfaces` and not to be overridden. `applycoupling!` in particular owns the
    # symmetry guard, and a more specific method would shadow it.
    :applyi2o!,
    :applycoupling!,
    :modulate_in!,
    :modulate_out!,
    :coupling,
    :modulation,
    # The kernel a coupling collection implements, below the guarded `applycoupling!` entry. It
    # replaced the three direction-named per-pair primitives
    # (`i2otranslation!`/`o2itranslation!`/`adjointo2itranslation!`, two arities each), which #180
    # removed: direction and accumulation are arguments now.
    :applycouplingkernel!,
    :couplingkey,
    # The operator transform a direction applies at the coupling boundary, called by kernels that
    # serve several directions from one method.
    :directedoperation,
    # The symmetry guard every collection on the plain path must answer.
    :assertnosymmetry,
    # Moment multiplication downstream implements alongside the generated methods.
    :conjmul!,
    :applyconjugatedleafmul!,
    # Map accessors extensions read.
    :i2otranslator,
    :i2ochannel,
    :maptrait,
    :core,
    :plans,
    :scheduling,
]

# --- STABILITY: INTERNAL -------------------------------------------------------------------------

const INTERNAL = [
    # Scheduling operation tags.
    :AggregateOp,
    :DisaggregateOp,
    :I2IOp,
    :I2OOp,
    :O2OOp,
    :ModulationOp,
    :ReverseModulationOp,
    :LeafMulOp,
    :TrialAggregationOp,
    :TestAggregationOp,
    :TestDisaggregationOp,
    :TrialDisaggregationOp,
    # Scheduling plumbing below the documented `ExecutionScheduling` surface.
    :OuterScheduling,
    :InnerScheduling,
    :outerscheduling,
    :outerscheduler,
    :innerscheduler,
    :operationscheduler,
    :perlevel,
    :serialinnerscheduling,
    # Chunk bookkeeping. `nchunks` itself is PUBLIC (downstream buffer builders must answer it);
    # these scoped helpers are how this package drives it, and nothing downstream dispatches on them.
    :Schedules,
    :PlanSchedule,
    :checkexecutiondata,
    :withnchunks,
    :withschedulechunks,
    :schedulechunkcounts,
    :requestedntasks,
    # Translator functors and raw translations.
    :I2ITranslatorFunctor,
    :I2OTranslatorFunctor,
    :O2OTranslatorFunctor,
    :DenseTranslation,
    :DiagonalTranslation,
    # Level transfer machinery.
    :LevelTranslator,
    :LevelTranslators,
    :SidedLevelTranslators,
    :SectorSelector,
    :nsectors,
    :transferkey,
    :leveltransfer,
    :leveltranslation!,
    :applytransfer!,
    # Moment containers and small helpers.
    :Moments,
    :MomentCollections,
    :subset,
    :add!,
]

# --- AUDIENCE (deliberately overlapping) ----------------------------------------------------------

const USERFACING = [
    :H2Workspace,
    :similarworkspace,
    :ConcurrentWorkspaceError,
    :nearmul,
    :nearmul!,
    :nearinteractions,
    :farmul,
    :farmul!,
    :farinteractions,
    :operator,
    :ExecutionScheduling,
    :defaultexecutionscheduling,
    :serialexecutionscheduling,
    :executionscheduling,
    :Diagnostics,
    :checkmomentaliasing,
    :MomentAliasingReport,
    :SymmetryReducedI2OTranslationCollection,
    :symmetryreducedcollection,
    :symmetryreducedtoeplitzcollection,
    :symmetryreduceddensecollection,
    :VectorField,
    :TrueTensor,
    :PseudoTensor,
    :InteractionStrategy,
    :PassThroughNearInteractionsStrategy,
    :DummyNearInteractionsStrategy,
    :BlockBEASTNearInteractionsStrategy,
    Symbol("@efie"),
    Symbol("@mw3Dhypersingular"),
    Symbol("@mw3Dweaklysingular"),
    Symbol("@mw3Ddoublelayer"),
    Symbol("@hh3Ddoublelayer"),
    Symbol("@hh3Ddoublelayer_transposed"),
]

const IMPLEMENTERFACING = [
    # Dual-audience names. Their membership here does NOT remove them from `USERFACING`: audience is
    # deliberately non-exclusive, so these appear in both lists on purpose.
    #   `H2Workspace`         users construct executable workspaces; implementers construct them
    #                         over custom cores and rely on the workspace protocol.
    #   `ExecutionScheduling` users configure execution; implementers inspect, propagate and
    #                         validate the materialized scheduling object.
    #   `ConcurrentWorkspaceError` users catch it; implementers may throw it.
    :H2Workspace,
    :ExecutionScheduling,
    :GalerkinH2Map,
    :PetrovGalerkinH2Map,
    :AbstractH2Core,
    :GalerkinH2Core,
    :PetrovGalerkinH2Core,
    :AbstractH2Workspace,
    :GalerkinH2Workspace,
    :PetrovGalerkinH2Workspace,
    :WorkspaceBuffers,
    :VectorBuffers,
    :SidedChannels,
    :acquireworkspace!,
    :releaseworkspace!,
    :withworkspace,
    :ConcurrentWorkspaceError,
    Symbol("@createmoment"),
    Symbol("@createmomentcollection"),
    Symbol("@createlinked"),
    Symbol("@linkmomenttocollection"),
    Symbol("@createtoeplitzmoment"),
    Symbol("@createlinkedtoeplitz"),
    :AbstractMoment,
    :AbstractMomentCollection,
    :storetrait,
    :buildstorer,
    :StoreAsIs,
    :assemblemoments!,
    :superspace,
    :assemblymap,
    :AbstractInteractionStrategy,
    :AbstractNearInteractionsAssembler,
    :AbstractNearInteractionsStrategy,
    :PassThroughNearInteractionsAssembler,
    :DummyNearInteractionsAssembler,
    :RandomNearInteractionsAssembler,
    :SparseBEASTNearInteractionsAssembler,
    :BlockBEASTNearInteractionsAssembler,
    :nearcriterion,
    :translatingnodesiterator,
    :aggregatenode,
    :aggregationmode,
    :buildnearinteractionsassembler,
    :assemblenearinteractions,
    :assembleplans,
    :I2OTranslationCollection,
    :DummyI2OTranslationCollection,
    :I2ITranslationCollection,
    :DenseI2I,
    :SeparableI2I,
    :i2itrait,
    :AbstractI2OTranslationTrait,
    :ToeplitzI2OTranslationTrait,
    :elementwisemul!,
    :nchunks,
    QUALIFIED_EXTENSION...,
]

@testset "stability classes partition the exported surface" begin
    exportedclasses = [PUBLIC, INTERNAL]
    for t in exportedclasses
        @test length(t) == length(unique(t))
    end
    for (i, a) in pairs(exportedclasses), (j, b) in pairs(exportedclasses)
        i < j && @test isempty(intersect(a, b))
    end

    # `names` includes the module's own name, which belongs to no class.
    exported = setdiff(Set(names(H2Factory)), Set([:H2Factory]))
    classified = union(Set.(exportedclasses)...)

    @test isempty(setdiff(exported, classified))   # nothing exported is unclassified
    @test isempty(setdiff(classified, exported))   # nothing classified has stopped being exported
end

@testset "QUALIFIED_EXTENSION is unexported but real" begin
    @test length(QUALIFIED_EXTENSION) == length(unique(QUALIFIED_EXTENSION))
    for n in QUALIFIED_EXTENSION
        # Supported without being exported: that is the whole class. Both halves have to hold, and
        # the second is what would catch a name quietly gaining an export and drifting classes.
        @test isdefined(H2Factory, n)
        @test !(n in names(H2Factory))
    end
end

@testset "the audience axis is independent of stability" begin
    # Audience membership is non-exclusive. The three dual names are pinned INDIVIDUALLY and by
    # identity rather than by counting: the axes are justified because stability and audience answer
    # different questions, not because some quota of names occupies the intersection. A bare
    # "the overlap is nonempty" assertion would add nothing once these exist, and would quietly
    # start passing on the wrong name if one of them were reclassified.
    both = intersect(USERFACING, IMPLEMENTERFACING)
    @test :ConcurrentWorkspaceError in both
    @test :H2Workspace in both
    @test :ExecutionScheduling in both

    # The converse for the two nearest neighbours, which are deliberately single-audience: without
    # this the dual list could grow by accident and nothing would notice.
    @test :similarworkspace in USERFACING
    @test !(:similarworkspace in IMPLEMENTERFACING)
    @test :nchunks in IMPLEMENTERFACING
    @test !(:nchunks in USERFACING)

    # Every supported name has at least one audience.
    supported = union(Set(PUBLIC), Set(QUALIFIED_EXTENSION))
    audienced = union(Set(USERFACING), Set(IMPLEMENTERFACING))
    @test isempty(setdiff(supported, audienced))

    # And nothing unsupported claims one: an INTERNAL name has no promised audience.
    @test isempty(intersect(Set(INTERNAL), audienced))
end

@testset "every classified name resolves" begin
    # NEGATIVE CONTROL: a name that exists nowhere must be seen as absent, otherwise the checks
    # above could be satisfied by lists of typos that `names` also fails to contain.
    @test !isdefined(H2Factory, :nosuchbindinghere)
    @test !(:nosuchbindinghere in names(H2Factory))

    for class in (PUBLIC, QUALIFIED_EXTENSION, INTERNAL), n in class
        @test isdefined(H2Factory, n)
    end
end

@testset "the Interfaces namespace owns what it declares" begin
    # Generics migrated into `H2Factory.Interfaces`. The parent brings each in with
    # `import .Interfaces: f`, so `H2Factory.f` and `H2Factory.Interfaces.f` must be ONE binding.
    #
    # Identity is the assertion that matters. Were the parent to use `using .Interfaces` instead, a
    # later `function buildmoments(...)` in an implementation file would define a SECOND function
    # shadowing the imported one; `H2Factory.buildmoments` would still resolve, still be callable and
    # still look right, while every downstream method on `Interfaces.buildmoments` silently left
    # dispatch. Comparing the two bindings is what distinguishes those cases.
    # name => whether it is exported. Migrating a declaration must not change export status, and a
    # blanket "none of these is exported" would not say that: the workspace hooks are unexported and
    # five of the six near-field names are exported, so the expectation has to be per name or it
    # stops being an assertion about preservation at all.
    migrated = Dict(
        # Workspace construction: zero H2Factory methods, six downstream methods each.
        :buildmoments => false,
        :buildvectorbuffers => false,
        :buildi2ichannels => false,
        :buildo2ochannels => false,
        :buildi2ochannel => false,
        :schedulinglevels => false,
        # Near-field strategy. The first two are protocol by OBSERVED downstream extension. The four
        # accessors have NO downstream methods at all and are protocol by an explicit SUBTYPE
        # CONTRACT: their documentation promises a strategy need not store its data under a fixed
        # field name. Both kinds keep their H2Factory defaults, which stay in implementation files.
        :buildnearinteractionsassembler => true,
        :isgalerkinsymmetric => false,
        :nearcriterion => true,
        :translatingnodesiterator => true,
        :aggregatenode => true,
        :aggregationmode => true,
        # Moment storage. Protocol by observed extension: MLFMA supplies `buildstorer` methods and iFMM a
        # `storetrait`. Both keep their H2Factory defaults, and both had NO documentation before
        # `Interfaces` existed, so their contracts there are written rather than relocated.
        :storetrait => true,
        :buildstorer => true,
        # Leaf multiplication: the busiest extension surface by method count, MLFMA supplying 25
        # `applyleafmul!` methods and 10 conjugated ones. All four keep their H2Factory defaults.
        :applyleafmul! => false,
        :applyconjugatedleafmul! => false,
        :leafmul => false,
        :leafmulscheduler => false,
        # Moment multiplication. Registered by LINKING a moment/collection pair rather than by
        # implementing anything: `@linkmomenttocollection` generates the methods, 84 of them in MLFMA
        # and 42 in iFMM, none visible as text in either source. Declared here only once #181 had
        # removed its universal `(Any, Any, Any)` fallback, so the declaration describes a generic
        # that answers `hasmethod` honestly.
        :conjmul! => false,
        # I2O composition, modulation and scratch. Seven are protocol by observed iFMM extension;
        # `coupling`/`modulation` are the two halves of the composition protocol a custom wrapper must
        # override together. `i2ochannels` qualifies twice over, by contract and by the method the
        # H2Toeplitz EXTENSION supplies.
        #
        # `applycouplingkernel!` is here and `applycoupling!` deliberately is NOT. The kernel is the
        # only I2O coupling extension point; the entry is orchestration that owns the symmetry guard,
        # and declaring it as protocol would invite the more specific method that shadows the guard.
        # #180 collapsed the old direction-named per-pair primitives onto the kernel.
        :coupling => false,
        :modulation => false,
        :applycouplingkernel! => false,
        :modulate_in! => false,
        :modulate_out! => false,
        :i2ochannels => false,
        :borrowbuffer => false,
        :takefftplan! => false,
        :takeifftplan! => false,
        :assertnosymmetry => false,
        # Traversal optimization, its own family of one: a different KIND of hook, saying how a
        # backend would rather the traversal call it rather than what the backend is. MLFMA supplies
        # three production kernels; a specialization changes performance, never the result.
        :aggregateleafchunk! => false,
        # Near-interaction assembly: the other half of the near-field protocol, found by the
        # EXTENSION audit rather than the downstream one. H2BEAST and H2BlockSparse each supply two
        # methods, and the abstract-typed H2Factory ones have empty bodies, so an assembler that
        # forgets to implement it silently assembles nothing.
        :assemble => false,
    )
    for (n, isexported) in migrated
        @test getfield(H2Factory, n) === getfield(H2Factory.Interfaces, n)
        @test parentmodule(getfield(H2Factory, n)) === H2Factory.Interfaces
        @test (n in names(H2Factory)) == isexported
    end

    # DECLARATIONS ONLY. `Interfaces` exists to say what the protocol is, not to implement it, so
    # nothing it owns may carry a method defined inside it. Without this the module would drift into
    # an ordinary implementation file that happens to be named after an interface.
    for n in names(H2Factory.Interfaces; all=true)
        isdefined(H2Factory.Interfaces, n) || continue
        v = getfield(H2Factory.Interfaces, n)
        v isa Function || continue
        parentmodule(v) === H2Factory.Interfaces || continue
        @test all(m -> m.module !== H2Factory.Interfaces, methods(v))
    end

    # EXACT ownership, in both directions. `length(migrated) == 6` would have been tautological: it
    # tests only the literal written above it. Comparing the list against what the module ACTUALLY
    # owns catches the two real mistakes instead, a declaration that failed to move and a new
    # generic added to `Interfaces` without review.
    owned = Set(
        n for
        n in names(H2Factory.Interfaces; all=true) if isdefined(H2Factory.Interfaces, n) &&
            getfield(H2Factory.Interfaces, n) isa Function &&
            parentmodule(getfield(H2Factory.Interfaces, n)) === H2Factory.Interfaces
    )
    @test owned == Set(keys(migrated))
end

@testset "H2Factory exports nothing of H2Trees'" begin
    # The REEXPORT class used to live here, asserting these twelve bindings were H2Trees' and not
    # H2Factory's. That check did its job: it confirmed, every run, that removing them would drop
    # nothing of ours. Now it asserts the removal instead, so a stray `export` cannot put them back.
    #
    # `isdefined` is NOT the test. All eleven names remain perfectly defined inside H2Factory, which
    # does `using H2Trees` for its own use; what changed is that H2Factory no longer EXPORTS them, so
    # a consumer writing `using H2Factory` no longer silently acquires them. `names` is what
    # distinguishes those two, and it is the one a consumer actually feels.
    for n in [
        :H2Trees,
        :TwoNTree,
        :SimpleHybridTree,
        :buildtree,
        :buildforest,
        :TwoNTreeBuilder,
        :BlockTreeBuilder,
        :BoundingBallTreeBuilder,
        :KMeansTreeBuilder,
        :MetisTreeBuilder,
        :MetisForestBuilder,
        :SimpleHybridTreeBuilder,
    ]
        @test !(n in names(H2Factory))
    end

    # NEGATIVE CONTROL for the loop above: `names` must actually be able to see an H2Factory export,
    # otherwise a bug making it return nothing would let every assertion above pass vacuously.
    @test :farmul! in names(H2Factory)

    # The converse: nothing in the promised classes is a FOREIGN binding. Two owners qualify as
    # ours, and they are named explicitly rather than accepted by walking the module tree, because
    # accepting any H2Factory submodule would let this pass for a name that had drifted into some
    # future implementation submodule.
    #
    # `H2Factory.Interfaces` has to be admitted: a generic migrated there is still H2Factory's, and
    # requiring `=== H2Factory` would turn every successful migration into a failure here. That is
    # the third place in this refactor where an ownership check assumed a single owner, after the
    # two downstream extension-surface guards, so it is worth stating rather than quietly widening.
    for class in (PUBLIC, INTERNAL), n in class
        startswith(string(n), "@") && continue   # macro bindings resolve differently
        @test Base.which(H2Factory, n) in (H2Factory, H2Factory.Interfaces)
    end
end

end # module TestPublicAPI
