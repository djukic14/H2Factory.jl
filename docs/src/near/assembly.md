# Computing near interactions

Near interactions are evaluated directly rather than through the far-field
expansion. There are three choices: which interactions are near, how their
entries are computed, and how the results are stored.

## Select the interactions

```julia
interaction = H2Factory.InteractionStrategy(isnear=nearfunctor)
plans = H2Factory.assembleplans(tree, interaction)
predicate = H2Factory.nearcriterion(interaction)(tree)
```

`nearfunctor` is a tree-independent H2Trees criterion. Calling it on `tree`
binds it to that geometry. Use the same partition for near assembly and far
plans; otherwise interactions can be omitted or counted twice.

`InteractionStrategy` also sets the translating-node iterator, aggregation-node
selection, and traversal mode. It belongs to the whole interaction partition,
not only the near field. H2Trees documents the geometric predicates and plans.

## Select an assembler

```text
H2Factory.assemblenearinteractions(op, testspace, trialspace, tree, γ,
    nearstrategy, interaction; verbose=false)
```

This builds an assembler and calls it with `(tree, predicate)`. The built-in
`BlockBEASTNearInteractionsStrategy(quadstrategy)` selects block-sparse BEAST
assembly when its [extension](../extensions.md) is loaded.

A custom strategy derives from `H2Factory.AbstractNearInteractionsStrategy`
and implements:

```text
H2Factory.buildnearinteractionsassembler(op, testspace, trialspace, γ,
    strategy::MyNearStrategy)
```

Return a subtype of `H2Factory.AbstractNearInteractionsAssembler{T}`.
It supplies `size(assembler)` and the assembly methods below. Default operator
and space accessors read `operator`, `testspace`, and `trialspace` fields.

```text
H2Factory.assemble(assembler::MyAssembler, values, nearvalues, verbose)
H2Factory.assemble(assembler::MyAssembler, selfvalues, values, nearvalues, verbose)
```

The first receives matching row and column groups of global basis indices.
The second separates self interactions for symmetric Galerkin assembly.
It is used only when both `supportsgalerkinsymmetric(typeof(assembler))`
and `isgalerkinsymmetric(typeof(op))` opt in, on H2Factory's generics.
Both default to false. An omitted assembly method raises `MethodError`.

### Symmetric Galerkin assembly

The two declarations answer different questions:

| Query on H2Factory | Declares |
| :--- | :--- |
| `isgalerkinsymmetric(::Type{MyOperator})` | The operator has transpose symmetry for this assembly route |
| `supportsgalerkinsymmetric(::Type{MyAssembler})` | The assembler implements the symmetric assembly form |

The value-level `supportsgalerkinsymmetric(assembler)` combines both.
This example checks selection only; it does not implement or execute an assembler:

```jldoctest symmetric-trait-guide
using H2Factory

struct GuideSymmetricOperator end
struct GuideOtherOperator end
struct GuideSymmetricAssembler{O} <: H2Factory.AbstractNearInteractionsAssembler{Float64}
    operator::O
end
H2Factory.isgalerkinsymmetric(::Type{GuideSymmetricOperator}) = true
a = GuideSymmetricAssembler(GuideSymmetricOperator())
before = H2Factory.supportsgalerkinsymmetric(a)
H2Factory.supportsgalerkinsymmetric(::Type{<:GuideSymmetricAssembler}) = true
b = GuideSymmetricAssembler(GuideOtherOperator())
(before, H2Factory.supportsgalerkinsymmetric(a), H2Factory.supportsgalerkinsymmetric(b))

# output

(false, true, false)
```

An actual opt-in also requires the self/off-diagonal assembly method shown
above and correct symmetric storage. Hermitian symmetry alone is not a
substitute for transpose symmetry. The block-tree route uses the ordinary
assembly form, not this single-tree Galerkin optimization.

## Use existing data

`H2Factory.PassThroughNearInteractionsStrategy(N)` returns an already assembled
near matrix unchanged. It does not check or filter N against the tree.
`DummyNearInteractionsStrategy(quadstrategy)` produces a zero near block for
tests or an intentionally far-only map; it is not a complete physical near field.

Next: [storage and multiplication](storage.md).
