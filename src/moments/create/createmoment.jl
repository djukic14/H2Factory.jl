"""
    getpointers(moments)

Return pointers to dense component arrays, or `nothing` entries for other arrays.
"""
function getpointers(moments)
    pointers = if eltype(moments) <: DenseArray
        ps = Vector{Ptr{eltype(eltype(moments))}}(undef, length(moments))
        for i in eachindex(moments)
            ps[i] = pointer(moments[i])
        end
        ps
    else
        Vector{Nothing}(undef, length(moments))
    end

    return pointers
end

function getvectors(moments, pointers)
    eltype(moments) <: DenseArray ||
        error("getvectors only works for moments with DenseArray element type")
    MemoryBuffer = @memorybuffer
    return MemoryBuffer{eltype(eltype(moments))}[
        unsafe_wrap(
            MemoryBuffer{eltype(eltype(moments))}, pointers[i], prod(size(moments[i]))
        ) for i in eachindex(pointers)
    ]
end

"""
@createmoment StructName AbstractType

Define a moment type with `D` component arrays of the same size. It subtypes
`AbstractType{D,T}`, where `T` is the component array type.

# Example

```julia
abstract type NewAbstractMoment{D,T} <: H2Factory.AbstractMoment{D,T} end
@createmoment MyMoment NewAbstractMoment
@show m = MyMoment{2,Float64}((10,))
@show m = MyMoment(m.moments)
@show m = MyMoment(m)
```
"""
macro createmoment(name, AbstractType)
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
            pointers::SVector{D,P} # pointers to the data of the arrays
            vectors::SVector{D,V} # vectors wrapping the data of the arrays

            # Allocate D arrays of the requested size.
            function $name{D,T,P}(size::NTuple{N,I}) where {D,T,P,I,N}
                moments = [Array{T,N}(undef, size) for _ in 1:D]
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)

                return new{
                    D,Base.eltype(moments),Base.eltype(pointers),Base.eltype(vectors)
                }(
                    SVector{D}(moments), SVector{D}(pointers), SVector{D}(vectors)
                )
            end

            # Allocate D arrays of the requested size.
            function $name{D,T}(size::NTuple{N,I}) where {D,T,I,N}
                moments = [Array{T,N}(undef, size) for _ in 1:D]
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)

                return new{
                    D,Base.eltype(moments),Base.eltype(pointers),Base.eltype(vectors)
                }(
                    SVector{D}(moments), SVector{D}(pointers), SVector{D}(vectors)
                )
            end

            # Use the supplied component arrays.
            function $name{D,T,P}(moments::SVector{D,T2}) where {D,T,T2,P}
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)

                return new{D,T2,Base.eltype(pointers),Base.eltype(vectors)}(
                    moments, SVector{D}(pointers), SVector{D}(vectors)
                )
            end

            # Use the supplied component arrays.
            function $name{D2,T2}(moments::SVector{D,T}) where {D,T,D2,T2}
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)
                return new{D,T,Base.eltype(pointers),Base.eltype(vectors)}(
                    moments, SVector{D}(pointers), SVector{D}(vectors)
                )
            end

            # Infer D from the supplied arrays.
            function $name{D}(moments::SVector{D,T}) where {D,T}
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)
                return new{D,T,Base.eltype(pointers),Base.eltype(vectors)}(
                    moments, SVector{D}(pointers), SVector{D}(vectors)
                )
            end
            # Infer D and T from the supplied arrays.
            function $name(moments::SVector{D,T}) where {D,T}
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)
                return new{D,T,Base.eltype(pointers),Base.eltype(vectors)}(
                    moments, SVector{D}(pointers), SVector{D}(vectors)
                )
            end

            # Allocate new arrays with the same shape and type as m.
            function $name{D,T,P,V}(m::$name{D,T,P,V}) where {D,T,P,V}
                moments = [T(undef, size(first(m.moments))) for _ in 1:D]
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)
                return new{D,T,P,V}(
                    SVector{D}(moments), SVector{D}(pointers), SVector{D}(vectors)
                )
            end

            # Match m's shape without copying its data.
            function $name{D,T}(m::$name{D,T,P}) where {D,T,P}
                moments = [T(undef, size(first(m.moments))) for _ in 1:D]
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)

                return new{D,T,P,Base.eltype(vectors)}(
                    SVector{D}(moments), SVector{D}(pointers), SVector{D}(vectors)
                )
            end

            # Match m's shape without copying its data.
            function $name{D}(m::$name{D,T}) where {D,T}
                moments = [T(undef, size(first(m.moments))) for _ in 1:D]
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)
                return new{D,T,Base.eltype(pointers),Base.eltype(vectors)}(
                    SVector{D}(moments), SVector{D}(pointers), SVector{D}(vectors)
                )
            end

            # Match m's shape without copying its data.
            function $name(m::$name{D,T}) where {D,T}
                moments = [T(undef, size(first(m.moments))) for _ in 1:D]
                pointers = getpointers(moments)
                vectors = getvectors(moments, pointers)
                return new{D,T,Base.eltype(pointers),Base.eltype(vectors)}(
                    SVector{D}(moments), SVector{D}(pointers), SVector{D}(vectors)
                )
            end
        end

        stringname = string(mod.$name)
        """
            struct $stringname{D,T} <: $AbstractType{D,T}

        The `$stringname` struct represents a moment consisting of `D` arrays of type `T`.
        This struct was created with the `@createmoment` macro of `H2Factory.jl`.

        # Fields

          - `moments::SVector{D,T}`: `D` arrays of type `T`.

        # Constructors

          - `$(stringname){D,T}(size::NTuple{N,I}) where {D,T,I,N}`: Creates a new moment with
            `D` arrays of type `T` and size `size`.
          - `$(stringname){D,T}(moments::SVector{D,T}) where {D,T}`: Creates a new moment with
            the given `moments` in an `SVector`.
          - `$(stringname){D}(moments::SVector{D,T}) where {D,T}`: Creates a new moment with
            the given `moments` in an `SVector`, inferring `D` from the input.
          - `$(stringname)(moments::SVector{D,T}) where {D,T}`: Creates a new moment with the
            given `moments` in an `SVector`, inferring `T` and `D` from the input.
          - `$(stringname){D,T}(m::$stringname{D,T}) where {D,T}`: Creates a new moment of the
            same type as `m`, with arrays of the same size as the first array of `m`. The
            fields of `m` are not copied.
          - `$(stringname){D}(m::$stringname{D,T}) where {D,T}`: Creates a new moment of the
            same type as `m`, with arrays of the same size as the first array of `m`,
            inferring `T` from the input. The fields of `m` are not copied.
          - `$(stringname)(m::$stringname{D,T}) where {D,T}`: Creates a new moment of the
            same type as `m`, with arrays of the same size as the first array of `m`,
            inferring `T` and `D` from the input.

        # Example

        ```julia
        m = $stringname{2,Float64}((3, 3))

        # access the first array of the moment
        m[1] == m.moments[1]

        # create a new moment of the same type as `m` without copying the fields of `m`
        m2 = $stringname(m)

        typeof(m2) == typeof(m)
        ```
        """
        $name
    end
end

"""
@createmoment StructName

Similar to `@createmoment StructName AbstractType`, but the abstract type is implied to be
`H2Factory.AbstractMoment`.

# Example

```julia
@createmoment MyMoment
@show m = MyMoment{4,Float64}((3, 3, 3))
@show m = MyMoment(m.moments)
@show m = MyMoment(m)

@show H2Factory.dim(m)

# iterator over each array of the moment
for i in eachindex(m)
    # access each array of the moment
    println(m[i])
end
```
"""
macro createmoment(name)
    return quote
        @createmoment $name H2Factory.AbstractMoment
    end
end
