macro mw3Dweaklysingular(name, AbstractType)
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
                !iszero(singlelayer.α) ||
                    throw(ArgumentError("Maxwell weakly singular requires α ≠ 0"))
                iszero(singlelayer.β) ||
                    throw(ArgumentError("Maxwell weakly singular requires β = 0"))
                return new{typeof(singlelayer.gamma),typeof(singlelayer.α)}(singlelayer)
            end

            function $name(; gamma=nothing, wavenumber=nothing, alpha=nothing)
                γ, _ = BEAST.gamma_wavenumber_handler(gamma, wavenumber)

                α = if isnothing(alpha)
                    -γ
                else
                    alpha
                end

                alphabetatype = promote_type(typeof(γ), typeof(α))

                α = convert(alphabetatype, α)

                singlelayer = MWSingleLayer3D(γ, α, zero(typeof(α)))
                return $name(singlelayer)
            end

            function H2Factory.isgalerkinsymmetric(::Type{<:$name})
                return true
            end
        end

        stringname = string(mod.$name)

        """
            struct $stringname{T,U} <: $AbstractType{T,U}

        Wrap the weakly singular term of a `BEAST.MWSingleLayer3D` (`α ≠ 0`, `β = 0`). Accepts a
        compatible operator or the keywords `gamma`/`wavenumber` and `alpha`.
        """
        $name
    end
end

macro mw3Dweaklysingular(name)
    quote
        @mw3Dweaklysingular $name Maxwell3DWrapper

        function H2Factory.operator(op::$(esc(name)))
            return op.operator
        end
    end
end
