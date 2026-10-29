"""
@createmomentcollection StructName AbstractType

Define a collection type with `D` moment matrices of the same size. It subtypes
`AbstractType{D,T}`, where `T` is the matrix type.

# Example

```julia
abstract type NewAbstractMomentCollection{D,T} <: H2Factory.AbstractMomentCollection{D,T} end
@createmomentcollection MyMomentCollection NewAbstractMomentCollection
@show m = MyMomentCollection{2,Float64}((10, 10))
@show m = MyMomentCollection(m.moments)
@show m = MyMomentCollection(m)

# not allowed since arrays in collection are supposed to be abstract matrices
@show m = MyMomentCollection{2,Float64}((10,))
```
"""
macro createmomentcollection(name, AbstractType)
    quote
        # Resolve the abstract type in the calling module.
        local mod = @__MODULE__
        local AbstractType = try
            $AbstractType
        catch
            mod.$AbstractType
        end

        struct $name{D,T} <: AbstractType{D,T}
            # Each component is a matrix.
            moments::SVector{D,T}

            # Allocate D matrices of the requested size.
            function $name{D,T}(size::NTuple{2,I}) where {D,T,I}
                moments = [Matrix{T}(undef, size) for _ in 1:D]
                return new{D,Base.eltype(moments)}(SVector{D}(moments))
            end

            # Use the supplied matrices.
            function $name{D,T}(moments::SVector{D,T}) where {D,T}
                return new{D,T}(moments)
            end

            # Use the supplied matrices.
            function $name{D}(moments::SVector{D,T}) where {D,T}
                return new{D,T}(moments)
            end

            # Use the supplied matrices.
            function $name(moments::SVector{D,T}) where {D,T}
                return new{D,T}(moments)
            end

            # Match m's shape without copying its data.
            function $name{D,T}(m::$name{D,T}) where {D,T}
                moments = [T(undef, size(first(m.moments))) for _ in 1:D]
                return new{D,T}(SVector{D}(moments))
            end

            # Match m's shape without copying its data.
            function $name{D}(m::$name{D,T}) where {D,T}
                moments = [T(undef, size(first(m.moments))) for _ in 1:D]
                return new{D,T}(SVector{D}(moments))
            end

            # Match m's shape without copying its data.
            function $name(m::$name{D,T}) where {D,T}
                moments = [T(undef, size(first(m.moments))) for _ in 1:D]
                return new{D,T}(SVector{D}(moments))
            end
        end

        stringname = string(mod.$name)

        """
        struct $stringname{D,T} <: AbstractType{D,T}

            A collection of `D` moment matrices of type `T`, stored in a static vector.
            Created by `@createmomentcollection`.

        # Fields

          - `moments`::SVector{D,T}: A static vector of length `D` that holds the moments of
            the cluster. The elements of the static vector are abstract matrices of type `T`.

        # Constructors

          - `$stringname{D,T}(size::NTuple{2,I}) where {D,T,I}`: Create a new moment collection
            with `D` matrices of size `size` of type `T`.
          - `$stringname{D,T}(moments::SVector{D,T}) where {D,T}`: Create a new moment collection
            with the given `moments`.
          - `$stringname{D}(moments::SVector{D,T}) where {D,T}`: Create a new moment collection
            with the given `moments`.
          - `$stringname(moments::SVector{D,T}) where {D,T}`: Create a new moment collection
            with the given `moments`.
          - `$stringname{D,T}(m::$stringname{D,T}) where {D,T}`: Create a new moment collection
            of the same type as `m`. The matrices have the same size as the first matrix of `m`.
            The data of `m` is not copied.
          - `$stringname{D}(m::$stringname{D,T}) where {D,T}`: Create a new moment collection
            of the same type as `m`. The matrices have the same size as the first matrix of `m`.
            The data of `m` is not copied.
          - `$stringname(m::$stringname{D,T}) where {D,T}`: Create a new moment collection
            of the same type as `m`. The matrices have the same size as the first matrix of `m`.
            The data of `m` is not copied.
        """
        $name
    end
end

"""
@createmomentcollection StructName

Define a moment collection using `H2Factory.AbstractMomentCollection` as its parent.

# Example

```julia
@createmomentcollection MyMomentCollection
@show m = MyMomentCollection{2,Float64}((3, 3))
@show m = MyMomentCollection(m.moments)
@show m = MyMomentCollection(m)

@show H2Factory.dim(m)

# iterator over each matrix of the moment
for i in eachindex(m)
    # access each matrix of the moment
    println(m[i])
end
```
"""
macro createmomentcollection(name)
    return quote
        @createmomentcollection $name H2Factory.AbstractMomentCollection
    end
end
