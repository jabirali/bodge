using HDF5
# using Threads
using SparseArrays

"""
    load()

Load a Hamiltonian matrix in the SciPy CSC format from an HDF5 file.

This matrix is then converted to a Julia sparse matrix.
"""
function load()
	fid = h5open("bodge.hdf5", "r")
	data = read(fid["hamiltonian/data"])
	indptr = read(fid["hamiltonian/indptr"])
	indices = read(fid["hamiltonian/indices"])
	dim = read(fid["hamiltonian/dim"])
	close(fid)

	H = SparseMatrixCSC(dim, dim, indptr .+ 1, indices .+ 1, data)

	return H
end

function save(G)
	println("test")
end

function cheb(H)
	println(H.m)
	Threads.@threads for j in 1:8
		b = 32
		v1 = spzeros(H.m, b)
		for k in 1:b
			v1[b*(j-1)+k+1, k] = 1
		end

		v2 = H * v1

		for i in 1:500
			v1 = 2 * (H * v2) - v1 
			v2 = 2 * (H * v1) - v2 
		end
		display(v2[j])
	end
end

H = load()
cheb(H)


# function from_scipy(data, indices

# 	SparseMatrixCSC(N, N, indptr .+1, indices .+ 1, data)
# 	return SparseMatrixCSC(N, N, indptr .+1, indices .+ 1, data)
# end

# """
#     kernel(x)

# TBW
# """
# function kernel(x)
# 	println("test")
# end
