macro mw3Ddoublelayer(name, AbstractType)
    quote
        local mod = @__MODULE__
        local AbstractType = try
            $AbstractType
        catch
            mod.$AbstractType
        end

        struct $name{T,U} <: AbstractType{T,U}
            operator::BEAST.MWDoubleLayer3D{T,U}

            function $name(doublelayer)
                return new{typeof(doublelayer.alpha),typeof(doublelayer.gamma)}(doublelayer)
            end

            function $name(; gamma=nothing, wavenumber=nothing, alpha=nothing)
                γ, _ = BEAST.gamma_wavenumber_handler(gamma, wavenumber)

                α = if isnothing(alpha)
                    if isnothing(γ)
                        α = one(typeof(wavenumber))
                    else
                        one(typeof(γ))
                    end
                else
                    alpha
                end
                doublelayer = BEAST.MWDoubleLayer3D(α, γ)
                return $name(doublelayer)
            end

            function H2Factory.isgalerkinsymmetric(::Type{<:$name})
                return true
            end
        end

        stringname = string(mod.$name)

        """
            struct $stringname{T,U} <: $AbstractType{T,U}

        Wrap a `BEAST.MWDoubleLayer3D`. Accepts an existing operator or the keywords
        `gamma`/`wavenumber` and `alpha`.
        """
        $name
    end
end

macro mw3Ddoublelayer(name)
    quote
        @mw3Ddoublelayer $name Maxwell3DWrapper

        function H2Factory.operator(op::$(esc(name)))
            return op.operator
        end
    end
end
