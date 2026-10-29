
function ToeplitzI2OTranslationCollection(
    tree, aggregationplan::A, disaggregationplan::D, translationtrait, Translator; kwargs...
) where {A<:H2Trees.AbstractAggregationPlan,D<:H2Trees.AbstractDisaggregationPlan}
    return ToeplitzI2OTranslationCollection(
        tree,
        H2Trees.translatingplan(aggregationplan, disaggregationplan),
        translationtrait,
        Translator;
        kwargs...,
    )
end

function H2Factory.toeplitzi2ocollection(
    tree, translatingplan::A, translationtrait, Translator; kwargs...
) where {A<:H2Trees.AbstractPlan}
    i2otfplan, _ = I2OTranslationCollection(
        tree, translatingplan, translationtrait, Translator; kwargs...
    )
    return ToeplitzI2OTranslationCollection(i2otfplan, H2Trees.tree(translatingplan))
end

# Build the coupling collection and a fresh workspace channel.
function ToeplitzI2OTranslationCollection(
    tree, translatingplan::A, translationtrait, Translator; kwargs...
) where {A<:H2Trees.AbstractPlan}
    collection = H2Factory.toeplitzi2ocollection(
        tree, translatingplan, translationtrait, Translator; kwargs...
    )
    return collection, H2Factory.i2ochannels(collection, translatingplan)
end

# Build one workspace's FFT plan pools from the stored coupling sizes.
function H2Factory.i2ochannels(
    collection::ToeplitzI2OTranslationCollection, translatingplan
)
    fftplans = []
    ifftplans = []
    fftlevels = Int[]

    # Couplings at a level share a size; one representative is enough.
    for level in H2Trees.levels(translatingplan)
        representative = _firsttranslation(collection, translatingplan, level)
        representative === nothing && continue
        # FFT plans require a mutable dense buffer with the stored tensor's size and element type.
        tensortype = H2Factory.toeplitzeltype(representative)
        tensorsize = H2Factory.toeplitzsize(representative)
        buffer = Array{eltype(tensortype),ndims(tensortype)}(undef, tensorsize)
        # Allocate one plan per scheduled chunk.
        nc = H2Factory.nchunks(level)
        push!(fftplans, [plan_fft!(buffer; num_threads=nothing) for _ in 1:nc])
        push!(ifftplans, [plan_ifft!(buffer; num_threads=nothing) for _ in 1:nc])
        push!(fftlevels, level)
    end

    return ToeplitzI2OChannel(
        _converttotypestable(fftplans), _converttotypestable(ifftplans), fftlevels
    )
end

# Return the first coupling at this level for plan sizing, or nothing if there are none.
function _firsttranslation(i2otfplan, translatingplan, level)
    for rnode in H2Trees.receivingnodes(translatingplan, level)
        for tnode in H2Trees.translatingnodes(translatingplan, rnode, level)
            return i2otfplan[(rnode, tnode)]
        end
    end
    return nothing
end

function _converttotypestable(buffers)
    isempty(buffers) && return buffers
    return convert(Vector{typeof(buffers[1])}, buffers)
end
