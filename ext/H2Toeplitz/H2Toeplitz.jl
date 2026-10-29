module H2Toeplitz
using AbstractFFTs
using H2Factory
using H2Trees
using LinearAlgebra
using LinearMaps
import H2Factory:
    ToeplitzI2OTranslationCollection,
    ToeplitzI2OChannel,
    ToeplitzI2OTranslationTrait,
    I2OTranslationCollection,
    ToeplitzMatrixRepresentation,
    AdjointToeplitzMatrixRepresentation,
    AbstractToeplitzI2OTranslationTrait,
    toeplitzsize

include("toeplitzi2ocollection.jl")

function LinearMaps._unsafe_mul!(y, A::ToeplitzMatrixRepresentation, x)
    ts = toeplitzsize(A)
    s = floor.(Ref(Int), (ts .+ 1) ./ 2)

    xbuffer = zeros(eltype(A), ts)
    xreshaped = reshape(x, s)
    H2Factory._copytobuffer!(xbuffer, xreshaped)
    xfft = fft(xbuffer)
    res = ifft(A.toeplitz .* xfft)

    yreshaped = reshape(y, s)
    yreshaped .= view(res, axes(yreshaped)...)
    return y
end

function LinearMaps._unsafe_mul!(y, A::AdjointToeplitzMatrixRepresentation, x)
    x = conj(x)
    ts = toeplitzsize(A)
    s = floor.(Ref(Int), (ts .+ 1) ./ 2)
    xbuffer = zeros(eltype(A), ts)
    xreshaped = reshape(x, s)
    H2Factory._copytobuffer!(xbuffer, xreshaped)
    xfft = ifft(xbuffer)
    res = fft(A.toeplitz .* xfft)

    yreshaped = reshape(y, s)
    yreshaped .= view(res, axes(yreshaped)...)
    return conj!(y)
end

end # module H2Toeplitz
