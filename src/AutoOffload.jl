module AutoOffload

using LinearAlgebra, AbstractFFTs, FFTW

@static if Base.find_package("CuArrays") !== nothing
    using CuArrays
    if Float64(CuArrays.CUDAdrv.totalmem(first(CuArrays.CUDAdrv.devices()))) > 1e9
        @info("CUDA support found, automatic GPU acceleration will be enabled.")
        const GPU_SUPPORTED = true
        const AUTO_GPU_SIZE = 100
        cuify(x) = CuArrays.CuArray(x)

        # Piracy, should get upstreamed
        function Base.ldiv!(x::CuArrays.CuArray,_qr::CuArrays.CUSOLVER.CuQR,b::CuArrays.CuArray)
          _x = UpperTriangular(_qr.R) \ (_qr.Q' * reshape(b,length(b),1))
          x .= vec(_x)
        end
    else
        @info("CUDA support not found, GPU acceleration will not be available.")
        const GPU_SUPPORTED = false
        const AUTO_GPU_SIZE = 100
        cuify(x) = x
    end
else
    @info("CUDA support not found, GPU acceleration will not be available.")
    const GPU_SUPPORTED = false
    const AUTO_GPU_SIZE = 100
    cuify(x) = x
end

function accelerated_mul!(X,A,B)
    if GPU_SUPPORTED && size(A,1) > AUTO_GPU_SIZE && b isa Array
        A,B = cuify(A),cuify(B)
        X .= Array(A*B)
    else
        mul!(X,A,B)
    end
end

function accelerated_ldiv!(X,A,B)
    if GPU_SUPPORTED && size(A,1) > AUTO_GPU_SIZE && b isa Array
        _X,A,B = cuify(X),cuify(A),cuify(B)
        X .= Array(ldiv!(_X,A,B))
    else
        ldiv!(X,A,B)
    end
end

function accelerated_factorize!(A)
    if GPU_SUPPORTED && size(A,1) > AUTO_GPU_SIZE && b isa Array
        qr(cuify(A))
    else
        factorize(A)
    end
end

function accelerated_fft!(A)
    if GPU_SUPPORTED && size(A,1) > AUTO_GPU_SIZE && b isa Array
        A .= Array(fft!(cuify(A)))
    else
        FFTW.fft!(A)
    end
end

export accelerated_mul!, accelerated_ldiv!, accelerated_factorize!,
       accelerated_fft!

module Pirate

    # Somehow take over mul!, *, etc to autooffload to GPU?

end

end # module
