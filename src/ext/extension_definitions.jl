macro requiredocstringfunction(requiredpkg, ext, functioname)
    return quote
        @eval function $functioname end

        """
        $($functioname) requires $($requiredpkg) to load the $($ext) extension.
        """
        $functioname
    end
end

@requiredocstringfunction "BEAST" "H2BEAST" assemblemoments!
@requiredocstringfunction "BEAST" "H2BEAST" buffer
@requiredocstringfunction "BEAST" "H2BEAST" quaddata
@requiredocstringfunction "BEAST" "H2BEAST" assemblydata
@requiredocstringfunction "BEAST" "H2BEAST" globaltoactiveelements
@requiredocstringfunction "BEAST" "H2BEAST" superspace
@requiredocstringfunction "BEAST" "H2BEAST" assemblymap

"""
    @efie name
    @efie name Parent

Define an EFIE wrapper named `name`. `Parent` defaults to `Maxwell3DWrapper`.
Requires BEAST.
"""
macro efie end

"""
    @mw3Dhypersingular name
    @mw3Dhypersingular name Parent

Define a Maxwell divergence-term wrapper named `name`. `Parent` defaults to `Maxwell3DWrapper`.
Requires BEAST.
"""
macro mw3Dhypersingular end

"""
    @mw3Dweaklysingular name
    @mw3Dweaklysingular name Parent

Define a Maxwell weakly singular wrapper named `name`. `Parent` defaults to `Maxwell3DWrapper`.
Requires BEAST.
"""
macro mw3Dweaklysingular end

"""
    @mw3Ddoublelayer name
    @mw3Ddoublelayer name Parent

Define a Maxwell double-layer wrapper named `name`. `Parent` defaults to `Maxwell3DWrapper`.
Requires BEAST.
"""
macro mw3Ddoublelayer end

"""
    @hh3Ddoublelayer name
    @hh3Ddoublelayer name Parent

Define a Helmholtz double-layer wrapper named `name`. `Parent` defaults to `Helmholtz3DWrapper`.
Requires BEAST.
"""
macro hh3Ddoublelayer end

"""
    @hh3Ddoublelayer_transposed name
    @hh3Ddoublelayer_transposed name Parent

Define a transposed Helmholtz double-layer wrapper named `name`.
`Parent` defaults to `Helmholtz3DWrapper`. Requires BEAST.
"""
macro hh3Ddoublelayer_transposed end

# Load a concrete FFT backend, such as FFTW, to provide the AbstractFFTs operations.
@requiredocstringfunction "AbstractFFTs" "H2Toeplitz" toeplitzmul!
