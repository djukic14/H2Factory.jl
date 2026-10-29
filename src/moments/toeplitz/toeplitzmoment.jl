function toeplitzmoment(moment::A, i::Int) where {A<:AbstractToeplitzMoment}
    return moment.toeplitzmoments[i]
end

"""
    toeplitzvector(moment::AbstractToeplitzMoment, i::Int)

Return the cached vector view of padded component `i`. Use this instead of creating a new
`vec(toeplitzmoment(moment, i))` view on each call.
"""
function toeplitzvector(moment::A, i::Int) where {A<:AbstractToeplitzMoment}
    return moment.toeplitzvectors[i]
end

function numberofentries(moment::A) where {A<:AbstractToeplitzMoment}
    numentries = 0
    for i in eachindex(moment)
        @assert moment[i] isa DenseArray
        numentries += length(moment[i])
    end
    for i in eachindex(moment)
        @assert toeplitzmoment(moment, i) isa DenseArray
        numentries += length(toeplitzmoment(moment, i))
    end
    return numentries
end

function standardtoeplitzsize(size)
    return 2 .* size .- 1
end

"""
    @createtoeplitzmoment Name [AbstractType]

Create a moment type with ordinary component arrays and padded arrays for Toeplitz products.
The default parent is `H2Factory.AbstractToeplitzMoment`.
"""
macro createtoeplitzmoment(name, AbstractType)
    quote
        # Resolve the abstract type in the calling module.
        local mod = @__MODULE__

        local AbstractType = try
            $AbstractType
        catch
            mod.$AbstractType
        end

        struct $name{D,T,P,V} <: AbstractType{D,T}
            moments::SVector{D,T}
            toeplitzmoments::SVector{D,T} # enough memory to also include padding #TODO: think about aliasing this to moments
            pointers::SVector{D,P} # pointers to the data of the arrays
            vectors::SVector{D,V} # vectors wrapping the data of the arrays
            toeplitzpointers::SVector{D,P} # pointers to the data of the toeplitzmoments arrays
            toeplitzvectors::SVector{D,V} # cached vector views of the toeplitzmoments arrays

            function $name{D,T}(sizes::NTuple{N,I}) where {D,T,I<:Tuple,N}
                return $name{D,T}(sizes[1], sizes[2])
            end

            # Allocate D ordinary and D padded arrays.
            function $name{D,T}(
                size::NTuple{N,I}, tsize=standardtoeplitzsize(size)
            ) where {D,T,I,N}
                moments = [Array{T,N}(undef, size) for _ in 1:D]
                toeplitzmoments = [Array{T,N}(undef, tsize) for _ in 1:D]
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)
                toeplitzpointers = getpointers(toeplitzmoments)
                toeplitzvectors = getvectors(toeplitzmoments, toeplitzpointers)

                return new{
                    D,Base.eltype(moments),Base.eltype(pointers),Base.eltype(vectors)
                }(
                    SVector{D}(moments),
                    SVector{D}(toeplitzmoments),
                    SVector{D}(pointers),
                    SVector{D}(vectors),
                    SVector{D}(toeplitzpointers),
                    SVector{D}(toeplitzvectors),
                )
            end

            # The pooled allocator supplies both sets of arrays. Infer D and T from them.
            function $name{D2,T2}(
                moments::SVector{D,T}, tmoments::SVector{D,T}
            ) where {D,D2,T2,T}
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)
                toeplitzpointers = getpointers(tmoments)
                toeplitzvectors = getvectors(tmoments, toeplitzpointers)

                return new{D,T,Base.eltype(pointers),Base.eltype(vectors)}(
                    moments,
                    tmoments,
                    SVector{D}(pointers),
                    SVector{D}(vectors),
                    SVector{D}(toeplitzpointers),
                    SVector{D}(toeplitzvectors),
                )
            end
        end

        #TODO: add doc string
        # $name
    end
end

macro createtoeplitzmoment(name)
    return quote
        @createtoeplitzmoment $name H2Factory.AbstractToeplitzMoment
    end
end

"""
    @createlinkedtoeplitz MomentName CollectionName [AbstractMomentType AbstractCollectionType]

Create a Toeplitz moment type and a matching collection type, then link their multiplication
methods. Without parent types, the new types use H2Factory's abstract moment types.
"""
macro createlinkedtoeplitz(
    momentname, momentcollectionname, AbstractTypeMoment, AbstractTypeMomentCollection
)
    return quote
        @createtoeplitzmoment $momentname $AbstractTypeMoment
        @createmomentcollection $momentcollectionname $AbstractTypeMomentCollection
        @linkmomenttocollection $momentname $momentcollectionname
    end
end

macro createlinkedtoeplitz(momentname, momentcollectionname)
    return quote
        @createlinkedtoeplitz $momentname $momentcollectionname H2Factory.AbstractToeplitzMoment H2Factory.AbstractMomentCollection
    end
end
