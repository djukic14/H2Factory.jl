macro mw3Dhypersingular(name, AbstractType)
    quote
        local mod = @__MODULE__
        local AbstractType = try
            $AbstractType
        catch
            mod.$AbstractType
        end

        struct $name{T,U} <: AbstractType{T,U}
            operator::MWSingleLayer3D{T,U}

            function $name(singlelayer)
                iszero(singlelayer.α) ||
                    throw(ArgumentError("Maxwell hypersingular requires α = 0"))
                !iszero(singlelayer.β) ||
                    throw(ArgumentError("Maxwell hypersingular requires β ≠ 0"))
                return new{typeof(singlelayer.gamma),typeof(singlelayer.α)}(singlelayer)
            end

            function $name(; gamma=nothing, wavenumber=nothing, beta=nothing)
                γ, _ = BEAST.gamma_wavenumber_handler(gamma, wavenumber)

                β = if isnothing(beta)
                    -inv(γ)
                else
                    beta
                end

                alphabetatype = promote_type(typeof(γ), typeof(β))

                β = convert(alphabetatype, β)

                singlelayer = MWSingleLayer3D(γ, zero(typeof(β)), β)
                return $name(singlelayer)
            end

            function H2Factory.isgalerkinsymmetric(::Type{<:$name})
                return true
            end
        end

        stringname = string(mod.$name)

        """
            struct $stringname{T,U} <: $AbstractType{T,U}

        Wrap the divergence term of a `BEAST.MWSingleLayer3D` (`α = 0`, `β ≠ 0`). Accepts a
        compatible operator or the keywords `gamma`/`wavenumber` and `beta`.
        """
        $name
    end
end

macro mw3Dhypersingular(name)
    quote
        @mw3Dhypersingular $name Maxwell3DWrapper

        function H2Factory.operator(op::$(esc(name)))
            return op.operator
        end
    end
end
