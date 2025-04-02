// Repository: locuoco/coulomb_oscillators
// File: Simulation/fmm_cart_base3.cuh

//  Base functions used in Fast Multipole Method (FMM) in 3d
//  Copyright (C) 2024 Alessandro Lo Cuoco (alessandro.locuoco@gmail.com)

//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.

//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.

//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef FMM_CART_BASE3_CUDA_H
#define FMM_CART_BASE3_CUDA_H

#include <atomic> // std::atomic_ref

#include "fmm_cart_base.cuh"

__host__ __device__
inline SCAL coeff13(int n, int m)
{
// a coefficient used in the calculation of the gradient
// returns (-1)^m * (2n - 2m - 1)!!
// assumes m <= n/2 and n >= 1
	return (SCAL)paritysign(m) * static_odfactorial(2 * (n - m) - 1);
	// for DIM > 3, see Shanker, Huang, 2007
}

__host__ __device__
inline SCAL dyn_coeff13(int n, int m)
{
// a coefficient used in the calculation of the gradient
// returns (-1)^m * (2n - 2m - 1)!!
// assumes m <= n/2 and n >= 1
	return paritysign(m) * dfactorial(2 * (n - m) - 1);
	// for DIM > 3, see Shanker, Huang, 2007
}

/**

Let's define a (totally) symmetric tensor in 3 dimensions with order p. Then it has (p+1)(p+2)/2 independent
coefficients.
Usually a tensor A is denoted with the notation:
	A_a,b,c,d...
where a,b,c,d... are indices that can take only three values since we are in 3D. However,
it would be pointless to store all 3^p elements. Since the tensor is symmetric, swapping
any two indices won't change the result. So, an element of a symmetric tensor is completely
determined by the multiplicities of the values that indices take. Then we use a new set of indices x,y,z
to denote the elements of a symmetric tensor A:
	A_x,y,z
where p = x + y + z. With just 3 indices we can notate symmetric tensors of any order in 3 dimensions.
It is usually convenient in a program to have just one index for an array. We use an index i that
corresponds to the multiplicities x,y,z through an 1-1 relationship:
(we consider all elements where z >= 2 to be dependent in traceless symmetric tensors)
x,y,z | i | j | p
0,0,0 | 0 | 0 | 0
1,0,0 | 0 | 1 | 1
0,1,0 | 1 | 2 | 1
0,0,1 | 2 | 3 | 1
2,0,0 | 0 | 4 | 2
1,1,0 | 1 | 5 | 2
0,2,0 | 2 | 6 | 2
1,0,1 | 3 | 7 | 2
0,1,1 | 4 | 8 | 2 
0,0,2 | 5 | 9 | 2
3,0,0 | 0 |10 | 3 
2,1,0 | 1 |11 | 3
1,2,0 | 2 |12 | 3
0,3,0 | 3 |13 | 3
2,0,1 | 4 |14 | 3
1,1,1 | 5 |15 | 3
0,2,1 | 6 |16 | 3
1,0,2 | 7 |17 | 3
0,1,2 | 8 |18 | 3
0,0,3 | 9 |19 | 3
4,0,0 | 0 |20 | 4
3,1,0 | 1 |21 | 4
2,2,0 | 2 |22 | 4
1,3,0 | 3 |23 | 4
0,4,0 | 4 |24 | 4
3,0,1 | 5 |25 | 4
2,1,1 | 6 |26 | 4
1,2,1 | 7 |27 | 4
0,3,1 | 8 |28 | 4
2,0,2 | 9 |29 | 4
1,1,2 |10 |30 | 4
0,2,2 |11 |31 | 4
1,0,3 |12 |32 | 4
0,1,3 |13 |33 | 4
0,0,4 |14 |34 | 4
...
where 0 <= i <= p is an index for a p-order symmetric tensor, and j is a cumulative index, useful if we
merge a set of tensors of increasing order in a single array. With this notation there is a specific
relationship between x,y,z and i, which is:
i(x, z) = p * (p + 1) / 2 - (p - z) * (p - z + 1) / 2 + p - x
When converting from i to j or viceversa one has to take into account an offset that depends on p:
j = i + p*(p+1)*(p+2)/6
Note that p=0 (j=0) are the monopoles, p=1 (j=1,2,3) are dipoles, p=2 (j=4,5,6,7,8,9) quadrupoles etc...
Note also that, once p, i are fixed, one can uniquely find x, y and z from:
z(i) = k(i) = floor((2*p+3 - sqrt((2*p+3)^2 - 8*i)) / 2)
where k(i) is the smallest integer such that k(i)(2*p+3 - k(i)) > 2*i.
x(i) = p * (p + 1) / 2 - (p - z(i)) * (p - z(i) + 1) / 2 + p - i
y(i) = p - x(i) - z(i)

Traceless symmetric tensors indices:
x,y,z | i | j | p
0,0,0 | 0 | 0 | 0
1,0,0 | 0 | 1 | 1
0,1,0 | 1 | 2 | 1
0,0,1 | 2 | 3 | 1
2,0,0 | 0 | 4 | 2
1,1,0 | 1 | 5 | 2
0,2,0 | 2 | 6 | 2
1,0,1 | 3 | 7 | 2
0,1,1 | 4 | 8 | 2 
3,0,0 | 0 | 9 | 3 
2,1,0 | 1 |10 | 3
1,2,0 | 2 |11 | 3
0,3,0 | 3 |12 | 3
2,0,1 | 4 |13 | 3
1,1,1 | 5 |14 | 3
0,2,1 | 6 |15 | 3
4,0,0 | 0 |16 | 4
3,1,0 | 1 |17 | 4
2,2,0 | 2 |18 | 4
1,3,0 | 3 |19 | 4
0,4,0 | 4 |20 | 4
3,0,1 | 5 |21 | 4
2,1,1 | 6 |22 | 4
1,2,1 | 7 |23 | 4
0,3,1 | 8 |24 | 4

j = i + p^2
i(x, 0) = p - x
i(x, 1) = 2 * p - x
Thus i(x, z) can be simplified for z in {0, 1}:
i(x, z) = (z + 1) * p - x = z * p + p - x
Inverse relations:
z(i) = i div (p + 1) = 1_{j | j >= p + 1} (i)
x(i) = (z(i) + 1) * p - i
y(i) = p - x(i) - z(i)

Difference between symmetric and traceless symmetric tensor in the number of independent components
p | (p+1)*(p+2)/2 | 2*p+1
0 |       1       |   1
1 |       3       |   3
2 |       6       |   5
3 |      10       |   7
4 |      15       |   9
5 |      21       |  11
6 |      28       |  13
7 |      36       |  15
8 |      45       |  17
9 |      55       |  19
10|      66       |  21

Using totally traceless symmetric tensors drastically decreases the number of indipendent elements, and
in 3D we would have just 2*p + 1 elements. In this case we will have:
A_(x+2),y,z + A_x,(y+2),z + A_x,y,(z+2) = 0
where A is a tensor of order x + y + z + 2. This can also be written as a recurrence relation:
A_x,y,z = -A_(x+2),y,(z-2) - A_x,(y+2),(z-2)
where A is a tensor of order x + y + z. This can be written with the index notation as:
A_i = -A_(i-2*p+2*k(i)-5) - A_(i-2*p+2*k(i)-3)

An important property is that if we contract a symmetric tensor with a traceless symmetric tensor, the
result will be a symmetric traceless tensor. So we can save computation time by using the previous relation.

A tuple of 3D symmetric tensors with order from 0 to P (inclusive) will have (P+1)*(P+2)*(P+3)/6 elements.
A tuple of 3D traceless symmetric tensors with order from 0 to P (inclusive) will have (P+1)^2 elements.
In the following calculations we will use cartesian coordinates.

*/

__host__ __device__
constexpr int symmetricelems3(int n)
{
	return (n + 1) * (n + 2) / 2;
}

__host__ __device__
constexpr int tracelesselems3(int n)
{
	return 2 * n + 1;
}

__host__ __device__
constexpr int symmetricoffset3(int p)
{
	return p * (p + 1) * (p + 2) / 6;
}

__host__ __device__
constexpr int tracelessoffset3(int p)
{
	return p * p;
}

__host__ __device__
inline int k_coeff(int index, int n)
{
	int b = 2*n+3;
	return (b - sqrt(SCAL(b*b - 8*index))) / 2;
}

__host__ __device__
inline int symmetric_z_i(int i, int n)
{
	return k_coeff(i, n);
}

__host__ __device__
constexpr int symmetric_x_i(int i, int n, int z)
{
	return (n * (n + 1) - (n - z) * (n - z + 1)) / 2 + n - i;
}
__host__ __device__
inline int symmetric_x_i(int i, int n)
{
	return symmetric_x_i(i, n, symmetric_z_i(i, n));
}

__host__ __device__
constexpr int symmetric_i_x_z(int x, int z, int n)
{
	return (n * (n + 1) - (n - z) * (n - z + 1)) / 2 + n - x;
}

__host__ __device__
constexpr int traceless_z_i(int i, int n)
{
	return (i >= n+1); // i div (p + 1) = 1_{j | j >= p + 1} (i)
}

__host__ __device__
constexpr int traceless_x_i(int i, int n, int z)
{
	return (z + 1) * n - i;
}
__host__ __device__
constexpr int traceless_x_i(int i, int n)
{
	return traceless_x_i(i, n, traceless_z_i(i, n));
}

__host__ __device__
constexpr int traceless_i_x_z(int x, int z, int n)
{
	return (z + 1) * n - x;
}

__host__ __device__
constexpr SCAL traceless_A_x_z(const SCAL *__restrict__ A, int x, int z, int n)
{
// extract element from traceless tensor A. It's valid also for z >= 2
	if (z >= 2)
		return -traceless_A_x_z(A, x+2, z-2, n) - traceless_A_x_z(A, x, z-2, n);
	else
		return A[traceless_i_x_z(x, z, n)];
}

__host__ __device__
inline void trace3(SCAL *__restrict__ out, const SCAL *__restrict__ in, int n, int m)
{
// contract 2m indices in n-order symmetric tensor "in" giving (n-2m)-order tensor "out"
// assumes n >= 2m
// if n = 2m the result is a scalar
	int elems = symmetricelems3(n);
	int n_out = n - 2 * m;
	int elems_out = symmetricelems3(n_out);
	int i = 0;
	for (int z = 0; z <= n_out; ++z)
		for (int x = n_out-z; x >= 0; --x)
		{
			SCAL t(0);
			int j = symmetric_i_x_z(x, z, n);
			int k_j = k_coeff(j, n);
			const SCAL *inj = in + j;
			const SCAL *__restrict__ injk;
			for (int kz = 0; kz <= m; ++kz)
			{
				injk = inj + 2*kz*(n - k_j - kz) + kz;
				for (int kx = 0; kx <= m-kz; ++kx)
					t += (SCAL)trinomial(m, kx, kz) * injk[-2*kx];
			}
			out[i++] = t;
		}
}

__host__ __device__
inline void contract3(SCAL *__restrict__ C, const SCAL *__restrict__ A, const SCAL *__restrict__ B, int nA, int nB)
{
// contract two symmetric tensors A, B into a symmetric tensor C of order |nA-nB|
// if nA = nB the result is a scalar
// O(|nA-nB|^2 * min(nA,nB)^2)
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // at this point we assume nA >= nB
	int jb = symmetric_i_x_z(0, 0, nB);
	int kjb = k_coeff(jb, nB);
	const SCAL *Bj = B + jb;
	int i = 0;
	for (int z = 0; z <= nC; ++z)
		for (int x = nC-z; x >= 0; --x)
		{
			SCAL t(0);
			int ja = symmetric_i_x_z(x, z, nA);
			int kja = k_coeff(ja, nA);
			const SCAL *__restrict__ Aj = A + ja;
			const SCAL *__restrict__ Ajk, *Bjk;
			for (int kz = 0; kz <= nB; ++kz)
			{
				Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
				Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
				for (int kx = 0; kx <= nB-kz; ++kx)
					t += (SCAL)trinomial(nB, kx, kz) * Ajk[-kx] * Bjk[-kx];
			}
			C[i++] = t;
		}
}

__host__ __device__
inline void contract_acc3(SCAL *__restrict__ C, const SCAL *__restrict__ A, const SCAL *__restrict__ B, int nA, int nB)
{
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// and sum (accumulate) the result to C
// O(|nA-nB|^2 * min(nA,nB)^2)
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	int jb = symmetric_i_x_z(0, 0, nB);
	int kjb = k_coeff(jb, nB);
	const SCAL *Bj = B + jb;
	int i = 0;
	for (int z = 0; z <= nC; ++z)
		for (int x = nC-z; x >= 0; --x)
		{
			SCAL t(0);
			int ja = symmetric_i_x_z(x, z, nA);
			int kja = k_coeff(ja, nA);
			const SCAL *Aj = A + ja;
			const SCAL *__restrict__ Ajk, *__restrict__ Bjk;
			for (int kz = 0; kz <= nB; ++kz)
			{
				Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
				Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
				for (int kx = 0; kx <= nB-kz; ++kx)
					t += (SCAL)trinomial(nB, kx, kz) * Ajk[-kx] * Bjk[-kx];
			}
			C[i++] += t;
		}
}

__host__ __device__
inline void contract_ma3(SCAL *__restrict__ C, const SCAL *__restrict__ A, const SCAL *__restrict__ B, SCAL c, int nA, int nB)
{
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// O(|nA-nB|^2 * min(nA,nB)^2)
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	int jb = symmetric_i_x_z(0, 0, nB);
	int kjb = k_coeff(jb, nB);
	const SCAL *Bj = B + jb;
	int i = 0;
	for (int z = 0; z <= nC; ++z)
		for (int x = nC-z; x >= 0; --x)
		{
			SCAL t(0);
			int ja = symmetric_i_x_z(x, z, nA);
			int kja = k_coeff(ja, nA);
			const SCAL *Aj = A + ja;
			const SCAL *__restrict__ Ajk, *__restrict__ Bjk;
			for (int kz = 0; kz <= nB; ++kz)
			{
				Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
				Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
				for (int kx = 0; kx <= nB-kz; ++kx)
					t += (SCAL)trinomial(nB, kx, kz) * Ajk[-kx] * Bjk[-kx];
			}
			C[i++] += c*t;
		}
}

template <bool b_atomic = false>
__host__ __device__
inline void contract_traceless_ma3(SCAL *__restrict__ C, const SCAL *__restrict__ A, const SCAL *__restrict__ B, SCAL c, int nA, int nB)
{
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result, the other elements
// may be built a posteriori through the function traceless_refine
// O(|nA-nB| * min(nA,nB)^2)
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	int jb = symmetric_i_x_z(0, 0, nB);
	int kjb = k_coeff(jb, nB);
	const SCAL *Bj = B + jb;
#ifdef __CUDA_ARCH__
	const SCAL *coeff = d_contract_coeff + symmetricoffset3(nB);
#else
	const SCAL *coeff = h_contract_coeff + symmetricoffset3(nB);
#endif
	int i = 0, j;
	for (int z = 0; z <= min(1, nC); ++z)
		for (int x = nC-z; x >= 0; --x)
		{
			SCAL t(0);
			int ja = symmetric_i_x_z(x, z, nA);
			int kja = k_coeff(ja, nA);
			const SCAL *Aj = A + ja;
			const SCAL *__restrict__ Ajk, *__restrict__ Bjk;
			j = 0;
			for (int kz = 0; kz <= nB; ++kz)
			{
				Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
				Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
				for (int kx = 0; kx <= nB-kz; ++kx)
					t += coeff[j++] * Ajk[-kx] * Bjk[-kx];
			}
			if constexpr (b_atomic)
			{
#ifdef __CUDA_ARCH__
				myAtomicAdd(C + (i++), c*t);
#else
				std::atomic_ref<SCAL> atomicC(C[i++]);
				atomicC += c*t;
#endif
			}
			else
				C[i++] += c*t;
		}
}

template <bool b_atomic = false>
__device__
inline void contract_traceless_ma_coalesced3(SCAL *__restrict__ C, const SCAL *__restrict__ A, const SCAL *__restrict__ B,
                                             SCAL c, int nA, int nB, int tid, int bdim)
{
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result, the other elements
// may be built a posteriori through the function traceless_refine
// O(|nA-nB| * min(nA,nB)^2)
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	int jb = symmetric_i_x_z(0, 0, nB);
	int kjb = k_coeff(jb, nB);
	const SCAL *Bj = B + jb;
	int nelems = tracelesselems3(nC);
	for (int i = tid; i < nelems; i += bdim)
	{
		int z = traceless_z_i(i, nC);
		int x = traceless_x_i(i, nC, z);
		SCAL t(0);
		int ja = symmetric_i_x_z(x, z, nA);
		int kja = k_coeff(ja, nA);
		const SCAL *Aj = A + ja;
		const SCAL *__restrict__ Ajk, *__restrict__ Bjk;
		for (int kz = 0; kz <= nB; ++kz)
		{
			Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
			Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
			for (int kx = 0; kx <= nB-kz; ++kx)
				t += (SCAL)trinomial(nB, kx, kz) * Ajk[-kx] * Bjk[-kx];
		}
		if constexpr (b_atomic)
		{
#ifdef __CUDA_ARCH__
			myAtomicAdd(C + i, c*t);
#else
			std::atomic_ref<SCAL> atomicC(C[i]);
			atomicC += c*t;
#endif
		}
		else
			C[i] += c*t;
	}
}

template <bool b_atomic = false>
__device__
inline void contract_traceless_ma_coalesced_tuple3(SCAL *__restrict__ Ctuple, const SCAL *__restrict__ Atuple, const SCAL *__restrict__ B,
                                                   SCAL c, int nAmin, int nAmax, int nB, int tid, int bdim)
{
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result, the other elements
// may be built a posteriori through the function traceless_refine
// O(|nA-nB| * min(nA,nB)^2)

	// assuming nAmax >= nAmin >= nB
	int jb = symmetric_i_x_z(0, 0, nB);
	int kjb = k_coeff(jb, nB);
	const SCAL *Bj = B + jb;
	int offmin = tracelessoffset3(nAmin-nB);
	int offmax = tracelessoffset3(nAmax-nB+1);
#ifdef __CUDA_ARCH__
	const SCAL *coeff = d_contract_coeff + symmetricoffset3(nB);
#else
	const SCAL *coeff = h_contract_coeff + symmetricoffset3(nB);
#endif
	int j;
	for (int i = offmin+tid; i < offmax; i += bdim)
	{
		int nC = sqrtf(i);
		int nA = nC + nB;
		int ind = i - tracelessoffset3(nC);
		int z = traceless_z_i(ind, nC);
		int x = traceless_x_i(ind, nC, z);
		SCAL t(0);
		int ja = symmetric_i_x_z(x, z, nA);
		int kja = k_coeff(ja, nA);
		const SCAL *Aj = Atuple + symmetricoffset3(nA) + ja;
		const SCAL *__restrict__ Ajk, *__restrict__ Bjk;
		j = 0;
		for (int kz = 0; kz <= nB; ++kz)
		{
			Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
			Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
			for (int kx = 0; kx <= nB-kz; ++kx)
				t += coeff[j++] * Ajk[-kx] * Bjk[-kx];
		}
		if constexpr (b_atomic)
		{
#ifdef __CUDA_ARCH__
			myAtomicAdd(Ctuple + i, c*t*inv_factorial(nC));
#else
			std::atomic_ref<SCAL> atomicC(Ctuple[i]);
			atomicC += c*t*inv_factorial(nC);
#endif
		}
		else
			Ctuple[i] += c*t*inv_factorial(nC);
	}
}

template <bool b_atomic = false>
__host__ __device__
inline void contract_traceless2_ma3(SCAL *__restrict__ C, const SCAL *__restrict__ A, const SCAL *__restrict__ B, SCAL c, int nA, int nB)
{
// contract two traceless symmetric tensors A, B into a traceless symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result, the other elements
// may be built a posteriori through the function traceless_refine
// O(|nA-nB| * min(nA,nB)^2)
	if (nA < nB)
	{
		swap(A, B);
		swap(nA, nB);
	}
	int nC = nA - nB; // assumes nA >= nB
	int i = 0;
	for (int z = 0; z <= min(1, nC); ++z)
		for (int x = nC-z; x >= 0; --x)
		{
			SCAL t(0);
			for (int kz = 0; kz <= nB; ++kz)
				for (int kx = 0; kx <= nB-kz; ++kx)
					t += (SCAL)trinomial(nB, kx, kz) * traceless_A_x_z(A, x+kx, z+kz, nA) * traceless_A_x_z(B, kx, kz, nB);
			if constexpr (b_atomic)
			{
#ifdef __CUDA_ARCH__
				myAtomicAdd(C + (i++), c*t);
#else
				std::atomic_ref<SCAL> atomicC(C[i++]);
				atomicC += c*t;
#endif
			}
			else
				C[i++] += c*t;
		}
}

template<int nA, int nB, bool b_atomic = false>
__host__ __device__
inline void static_contract_traceless_ma3(SCAL *__restrict__ C, const SCAL *__restrict__ A, const SCAL *__restrict__ B, SCAL c)
{
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result, the other elements
// may be built a posteriori through the function traceless_refine
// O(|nA-nB| * min(nA,nB)^2)
	if constexpr (nA < nB)
		static_contract_traceless_ma3<nB, nA, b_atomic>(C, B, A, c);
	else
	{
		constexpr int nC = nA - nB; // assumes nA >= nB
		constexpr int jb = symmetric_i_x_z(0, 0, nB);
		int kjb = k_coeff(jb, nB);
		const SCAL *Bj = B + jb;
#ifdef __CUDA_ARCH__
		static constexpr const SCAL *coeff = d_contract_coeff + symmetricoffset3(nB);
#else
		static constexpr const SCAL *coeff = h_contract_coeff + symmetricoffset3(nB);
#endif
		int i = 0, j;
#pragma unroll
		for (int z = 0; z <= static_min(1, nC); ++z)
#pragma unroll
			for (int x = nC-z; x >= 0; --x)
			{
				SCAL t(0);
				int ja = symmetric_i_x_z(x, z, nA);
				int kja = k_coeff(ja, nA);
				const SCAL *Aj = A + ja;
				const SCAL *__restrict__ Ajk, *__restrict__ Bjk;
				j = 0;
#pragma unroll
				for (int kz = 0; kz <= nB; ++kz)
				{
					Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
					Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
#pragma unroll
					for (int kx = 0; kx <= nB-kz; ++kx)
						t += coeff[j++] * Ajk[-kx] * Bjk[-kx];
				}
				if constexpr (b_atomic)
				{
#ifdef __CUDA_ARCH__
					myAtomicAdd(C + (i++), c*t);
#else
					std::atomic_ref<SCAL> atomicC(C[i++]);
					atomicC += c*t;
#endif
				}
				else
					C[i++] += c*t;
			}
	}
}

template<int nA, int nB, bool b_atomic = false>
__host__ __device__
inline void static_contract_traceless2_ma3(SCAL *C, const SCAL *A, const SCAL *B, SCAL c)
{
// contract two traceless symmetric tensors A, B into a traceless symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result
// O(|nA-nB| * min(nA,nB)^2)
	if constexpr (nA < nB)
		static_contract_traceless2_ma3<nB, nA, b_atomic>(C, B, A, c);
	else
	{
		constexpr int nC = nA - nB; // assumes nA >= nB
		int i = 0;
#pragma unroll
		for (int z = 0; z <= static_min(1, nC); ++z)
#pragma unroll
			for (int x = nC-z; x >= 0; --x)
			{
				SCAL t(0);
#pragma unroll
				for (int kz = 0; kz <= nB; ++kz)
#pragma unroll
					for (int kx = 0; kx <= nB-kz; ++kx)
						t += (SCAL)trinomial(nB, kx, kz) * traceless_A_x_z(A, x+kx, z+kz, nA) * traceless_A_x_z(B, kx, kz, nB);
				if constexpr (b_atomic)
				{
#ifdef __CUDA_ARCH__
					myAtomicAdd(C + (i++), c*t);
#else
					std::atomic_ref<SCAL> atomicC(C[i++]);
					atomicC += c*t;
#endif
				}
				else
					C[i++] += c*t;
			}
	}
}

template <int nAmin, int nAmax, int nB, int bdim, bool b_atomic = false>
__device__
inline void static_contract_traceless_ma_coalesced_tuple3(SCAL *__restrict__ Ctuple, const SCAL *__restrict__ Atuple, const SCAL *__restrict__ B,
                                                          SCAL c, int tid)
{
// contract two symmetric tensors A, B into a symmetric tensor of order |nA-nB|
// multiply it by a scalar c and sum (accumulate) the result to C
// we can reduce complexity by knowing that the result must be traceless
// C contains only the indipendent elements of the result, the other elements
// may be built a posteriori through the function traceless_refine
// O(|nA-nB| * min(nA,nB)^2)

	// assuming nAmax >= nAmin >= nB
	constexpr int jb = symmetric_i_x_z(0, 0, nB);
	int kjb = k_coeff(jb, nB);
	const SCAL *Bj = B + jb;
	constexpr int offmin = tracelessoffset3(nAmin-nB);
	constexpr int offmax = tracelessoffset3(nAmax-nB+1);
#ifdef __CUDA_ARCH__
	static constexpr const SCAL *coeff = d_contract_coeff + symmetricoffset3(nB);
#else
	static constexpr const SCAL *coeff = h_contract_coeff + symmetricoffset3(nB);
#endif
	for (int i = offmin+tid; i < offmax; i += bdim)
	{
		int nC = sqrtf(i);
		int nA = nC + nB;
		int ind = i - tracelessoffset3(nC);
		int z = traceless_z_i(ind, nC);
		int x = traceless_x_i(ind, nC, z);
		SCAL t(0);
		int ja = symmetric_i_x_z(x, z, nA);
		int kja = k_coeff(ja, nA);
		const SCAL *Aj = Atuple + symmetricoffset3(nA) + ja;
		const SCAL *__restrict__ Ajk, *__restrict__ Bjk;
		int j = 0;
#pragma unroll
		for (int kz = 0; kz <= nB; ++kz)
		{
			Ajk = Aj + kz*(nA - kja) - kz*(kz - 1)/2;
			Bjk = Bj + kz*(nB - kjb) - kz*(kz - 1)/2;
#pragma unroll
			for (int kx = 0; kx <= nB-kz; ++kx)
				t += coeff[j++] * Ajk[-kx] * Bjk[-kx];
		}
		if constexpr (b_atomic)
		{
#ifdef __CUDA_ARCH__
			myAtomicAdd(Ctuple + i, c*t*inv_factorial(nC));
#else
			std::atomic_ref<SCAL> atomicC(Ctuple[i]);
			atomicC += c*t*inv_factorial(nC);
#endif
		}
		else
			Ctuple[i] += c*t*inv_factorial(nC);
	}
}

__host__ __device__
inline void traceless_refine3(SCAL *__restrict__ A, int n)
{
// build a traceless tensor using symmetric index notation
// O(n^2)
	int i = symmetric_i_x_z(n-2, 2, n);
	for (int z = 2; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
		{
			int jsx = symmetric_i_x_z(x+2, z-2, n);
			int jsy = symmetric_i_x_z(x, z-2, n);
			A[i++] = -A[jsx] - A[jsy];
		}
}

__device__
inline void traceless_refine_coalesced3(SCAL *__restrict__ A, int n, int tid, int bdim, int mask)
{
// build a traceless tensor using symmetric index notation
// O(n^2)
	for (int z = 2; z <= n; z += 2)
	{
		for (int xdz = tid; xdz <= 2*(n-z); xdz += bdim)
		{
			int zp = z + xdz / (n-z+1);
			int x = n-zp - xdz % (n-z+1);
			int i = symmetric_i_x_z(x, zp, n);
			int jsx = symmetric_i_x_z(x+2, zp-2, n);
			int jsy = symmetric_i_x_z(x, zp-2, n);
			A[i] = -A[jsx] - A[jsy];
		}
		__syncwarp(mask);
	}
}

template<int n>
__host__ __device__
inline void static_traceless_refine3(SCAL *__restrict__ A)
{
// build a traceless tensor using symmetric index notation
// O(n^2)
	int i = symmetric_i_x_z(n-2, 2, n);
#pragma unroll
	for (int z = 2; z <= n; ++z)
#pragma unroll
		for (int x = n-z; x >= 0; --x)
		{
			int jsx = symmetric_i_x_z(x+2, z-2, n);
			int jsy = symmetric_i_x_z(x, z-2, n);
			A[i++] = -A[jsx] - A[jsy];
		}
}

__host__ __device__
inline void gradient_exact3(SCAL *__restrict__ grad, int n, VEC d, SCAL r)
{
// calculate the gradient of 1/r of order n i.e. nabla^n 1/r, which is a n-order symmetric tensor.
// r is the distance, d is the unit vector
// NOTE that if r2 >> EPS2, it is also totally traceless
// O(n^5)
	if (n == 0)
		grad[0] = 1/r;
	else
	{
		SCAL C = (SCAL)paritysign(n) * binarypow(r, -n-1);
		int i = 0;
		for (int z = 0; z <= n; ++z)
			for (int x = n-z; x >= 0; --x)
			{
				int y = n-x-z;
				SCAL t1(0), t2(0), t3(0);
				for (int k1 = 0; k1 <= x/2; ++k1)
				{
					t2 = 0;
					for (int k2 = 0; k2 <= y/2; ++k2)
					{
						t3 = 0;
						for (int k3 = 0; k3 <= z/2; ++k3)
						{
							int m = k1+k2+k3;
							t3 += coeff13(n, m) * coeff2(z, k3) * binarypow(d.z, z - 2*k3);
						}
						t2 += t3 * coeff2(y, k2) * binarypow(d.y, y - 2*k2);
					}
					t1 += t2 * coeff2(x, k1) * binarypow(d.x, x - 2*k1);
				}
				grad[i++] = C * t1;
			}
	}
}

__host__ __device__
inline void gradient3(SCAL *__restrict__ grad, int n, VEC d, SCAL r, SCAL c = 1)
{
// calculate the gradient of 1/r of order n i.e. nabla^n 1/r, which is a n-order symmetric tensor.
// r is the distance, d is the unit vector
// this version assumes r2 >> EPS2 which ease the computation significantly (from O(n^5) to O(n^3))
// if this assumption does not hold, accuracy will be reduced
// O(n^3)
	if (n == 0)
		grad[0] = c/r;
	else if (n == 1)
	{
		SCAL C = -c/(r*r);
		grad[0] = C*d.x;
		grad[1] = C*d.y;
		grad[2] = C*d.z;
	}
	else
	{
		SCAL x2 = d.x*d.x, y2 = d.y*d.y;
		SCAL C = (SCAL)paritysign(n) * binarypow(r, -n-1) * c;
		SCAL powx, powy, powz = 1;
#ifdef __CUDA_ARCH__
		const SCAL *coeff = d_grad_coeff + d_grad_off[n];
#else
		const SCAL *coeff = h_grad_coeff + h_grad_off[n];
#endif
		int i = 0, j = 0;
		for (int z = 0; z <= 1; ++z)
		{
			for (int x = n-z; x >= 0; --x)
			{
				int y = n-x-z;
				SCAL t1(0), t2;
				powx = (x & 1) ? d.x : SCAL(1);
				for (int k1 = x/2; k1 >= 0; --k1)
				{
					t2 = 0;
					powy = (y & 1) ? d.y : SCAL(1);
					for (int k2 = y/2; k2 >= 0; --k2)
					{
						t2 += coeff[j++] * powy;
						powy *= y2;
					}
					t1 += t2 * powx;
					powx *= x2;
				}
				grad[i++] = C * t1 * powz;
			}
			powz = d.z;
		}
	}
}

__device__
inline void gradient_coalesced3(SCAL *__restrict__ grad, int n, VEC d, SCAL r, int tid, int bdim, SCAL c = 1)
{
// calculate the gradient of 1/r of order n i.e. nabla^n 1/r, which is a n-order symmetric tensor.
// r is the distance, d is the unit vector
// this version assumes r2 >> EPS2 which ease the computation significantly (from O(n^5) to O(n^3))
// if this assumption does not hold, accuracy will be reduced
// O(n^3)
	if (n == 0)
	{
		if (tid == 0)
			grad[0] = c/r;
	}
	else if (n == 1)
	{
		SCAL C = -c/(r*r);
		SCAL powx, powy, powz;
		for (int i = tid; i < 3; i += bdim)
		{
			int z = traceless_z_i(i, 1);
			int x = traceless_x_i(i, 1, z);
			int y = 1-x-z;
			powx = x ? d.x : SCAL(1);
			powy = y ? d.y : SCAL(1);
			powz = z ? d.z : SCAL(1);
			grad[i] = C * powz * powy * powx;
		}
	}
	else
	{
		SCAL x2 = d.x*d.x, y2 = d.y*d.y;
		SCAL C = (SCAL)paritysign(n) * binarypow(r, -n-1) * c;
		SCAL powx, powy, powz;
#ifdef __CUDA_ARCH__
		const SCAL *coeff = d_grad_coeff + d_grad_off[n];
		const int *sum = d_gradsum + tracelessoffset3(n);
#else
		const SCAL *coeff = h_grad_coeff + h_grad_off[n];
		const int *sum = h_gradsum + tracelessoffset3(n);
#endif
		int nelems = tracelesselems3(n), j;
		for (int i = tid; i < nelems; i += bdim)
		{
			int z = traceless_z_i(i, n);
			int x = traceless_x_i(i, n, z);
			int y = n-x-z;
			powz = z ? d.z : SCAL(1);
			powx = (x & 1) ? d.x : SCAL(1);
			SCAL t1(0), t2;
			j = sum[i];
			for (int k1 = x/2; k1 >= 0; --k1)
			{
				t2 = 0;
				powy = (y & 1) ? d.y : SCAL(1);
				for (int k2 = y/2; k2 >= 0; --k2)
				{
					t2 += coeff[j++] * powy;
					powy *= y2;
				}
				t1 += t2 * powx;
				powx *= x2;
			}
			grad[i] = C * t1 * powz;
		}
	}
}

template<int n>
__host__ __device__
inline void static_gradient3(SCAL *__restrict__ grad, VEC d, SCAL r, SCAL c = 1)
{
// calculate the gradient of 1/r of order n i.e. nabla^n log(r), which is a n-order symmetric tensor.
// r is the distance, d is the unit vector
// this version assumes r2 >> EPS2 which ease the computation significantly (from O(n^5) to O(n^3))
// if this assumption does not hold, accuracy will be reduced
// O(n^3)
	if constexpr (n == 0)
		grad[0] = c/r;
	else if constexpr (n == 1)
	{
		SCAL C = -c/(r*r);
		grad[0] = C*d.x;
		grad[1] = C*d.y;
		grad[2] = C*d.z;
	}
	else
	{
		SCAL x2 = d.x*d.x, y2 = d.y*d.y;
		SCAL C = (SCAL)paritysign(n) * binarypow(r, -n-1) * c;
		SCAL powx, powy, powz = 1;
#ifdef __CUDA_ARCH__
		static constexpr const SCAL *coeff = d_grad_coeff + d_grad_off[n];
#else
		static constexpr const SCAL *coeff = h_grad_coeff + h_grad_off[n];
#endif
		int i = 0, j = 0;
#pragma unroll
		for (int z = 0; z <= 1; ++z)
		{
#pragma unroll
			for (int x = n-z; x >= 0; --x)
			{
				int y = n-x-z;
				SCAL t1(0), t2;
				powx = (x & 1) ? d.x : SCAL(1);
#pragma unroll
				for (int k1 = x/2; k1 >= 0; --k1)
				{
					t2 = 0;
					powy = (y & 1) ? d.y : SCAL(1);
#pragma unroll
					for (int k2 = y/2; k2 >= 0; --k2)
					{
						t2 += coeff[j++] * powy;
						powy *= y2;
					}
					t1 += t2 * powx;
					powx *= x2;
				}
				grad[i++] = C * t1 * powz;
			}
			powz = d.z;
		}
	}
}

template <int n, unsigned bdim>
__device__
inline void static_gradient_coalesced3(SCAL *__restrict__ grad, VEC d, SCAL r, unsigned tid, SCAL c = 1)
{
// calculate the gradient of 1/r of order n i.e. nabla^n 1/r, which is a n-order symmetric tensor.
// r is the distance, d is the unit vector
// this version assumes r2 >> EPS2 which ease the computation significantly (from O(n^5) to O(n^3))
// if this assumption does not hold, accuracy will be reduced
// O(n^3)
	if constexpr (n == 0)
	{
		if (tid == 0)
			grad[0] = c/r;
	}
	else if constexpr (n == 1)
	{
		SCAL C = -c/(r*r);
		SCAL powx, powy, powz;
		for (int i = tid; i < 3; i += bdim)
		{
			int z = traceless_z_i(i, 1);
			int x = traceless_x_i(i, 1, z);
			int y = 1-x-z;
			powx = x ? d.x : SCAL(1);
			powy = y ? d.y : SCAL(1);
			powz = z ? d.z : SCAL(1);
			grad[i] = C * powz * powy * powx;
		}
	}
	else
	{
		SCAL x2 = d.x*d.x, y2 = d.y*d.y;
		SCAL C = (SCAL)paritysign(n) * binarypow(r, -n-1) * c;
		SCAL powx, powy, powz;
#ifdef __CUDA_ARCH__
		static constexpr const SCAL *coeff = d_grad_coeff + d_grad_off[n];
		static constexpr const int *sum = d_gradsum + tracelessoffset3(n);
#else
		static constexpr const SCAL *coeff = h_grad_coeff + h_grad_off[n];
		static constexpr const int *sum = h_gradsum + tracelessoffset3(n);
#endif
		constexpr unsigned nelems = tracelesselems3(n);
		unsigned j;
		for (unsigned i = tid; i < nelems; i += bdim)
		{
			int z = traceless_z_i(i, n);
			int x = traceless_x_i(i, n, z);
			int y = n-x-z;
			powz = z ? d.z : SCAL(1);
			powx = (x & 1) ? d.x : SCAL(1);
			SCAL t1(0), t2;
			j = sum[i];
			for (int k1 = x/2; k1 >= 0; --k1)
			{
				t2 = 0;
				powy = (y & 1) ? d.y : SCAL(1);
				for (int k2 = y/2; k2 >= 0; --k2)
				{
					t2 += coeff[j++] * powy;
					powy *= y2;
				}
				t1 += t2 * powx;
				powx *= x2;
			}
			grad[i] = C * t1 * powz;
		}
	}
}

__host__ __device__
inline void tensorpow3(SCAL *__restrict__ power, int n, VEC d)
{
// calculate the powers tensor of d of order n, e.g.:
// r^(0) = 1
// r^(1) = (x, y, z)
// r^(2) = (x^2, xy, xz, y^2, yz, z^2)
// r^(3) = (x^3, x^2 y, x^2 z, xy^2, xyz, xz^2, y^3, y^2 z, yz^2, z^3)
// etc...
// O(n^2)
	int i = 0;
	for (int z = 0; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
			power[i++] = binarypow(d.x, x) * binarypow(d.y, n-x-z) * binarypow(d.z, z);
}

__host__ __device__
inline void tracelesspow3(SCAL *__restrict__ power, int n, VEC d, SCAL r)
{
// O(n^3)
	if (n == 0)
		power[0] = (SCAL)1;
	else
	{
		SCAL C = binarypow(r, n) / static_odfactorial(2*n-1);
		int i = 0;
		for (int z = 0; z <= 1; ++z)
			for (int x = n-z; x >= 0; --x)
			{
				int y = n-x-z;
				SCAL t1(0), t2(0);
				for (int k1 = 0; k1 <= x/2; ++k1)
				{
					t2 = 0;
					for (int k2 = 0; k2 <= y/2; ++k2)
					{
						int m = k1+k2;
						t2 += (SCAL)coeff13(n, m) * coeff2(y, k2) * binarypow(d.y, y - 2*k2);
					}
					t1 += t2 * coeff2(x, k1) * binarypow(d.x, x - 2*k1);
				}
				power[i++] = C * t1 * binarypow(d.z, z);
			}
	}
}

template <int n>
__host__ __device__
inline void static_tensorpow3(SCAL *__restrict__ power, VEC d)
{
// O(n^2)
	int i = 0;
#pragma unroll
	for (int z = 0; z <= n; ++z)
#pragma unroll
		for (int x = n-z; x >= 0; --x)
			power[i++] = binarypow(d.x, x) * binarypow(d.y, n-x-z) * binarypow(d.z, z);
}

template <int n>
__host__ __device__
inline void static_tracelesspow3(SCAL *__restrict__ power, VEC d, SCAL r)
{
// O(n^3)
	if constexpr (n == 0)
		power[0] = (SCAL)1;
	else
	{
		SCAL C = binarypow(r, n) / static_odfactorial(2*n-1);
		int i = 0;
#pragma unroll
		for (int z = 0; z <= 1; ++z)
#pragma unroll
			for (int x = n-z; x >= 0; --x)
			{
				int y = n-x-z;
				SCAL t1(0), t2(0);
#pragma unroll
				for (int k1 = 0; k1 <= x/2; ++k1)
				{
					t2 = 0;
#pragma unroll
					for (int k2 = 0; k2 <= y/2; ++k2)
					{
						int m = k1+k2;
						t2 += (SCAL)coeff13(n, m) * coeff2(y, k2) * binarypow(d.y, y - 2*k2);
					}
					t1 += t2 * coeff2(x, k1) * binarypow(d.x, x - 2*k1);
				}
				power[i++] = C * t1 * binarypow(d.z, z);
			}
	}
}

__host__ __device__
inline void p2m3(SCAL *__restrict__ M, int n, VEC d)
{
// particle to multipole expansion of order n
// d is the coordinate of the particle from the (near) expansion center
// O(n^2)
	SCAL C = (SCAL)paritysign(n) * (SCAL)inv_factorial(n);
	int i = 0;
	for (int z = 0; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
			M[i++] = C * binarypow(d.x, x) * binarypow(d.y, n-x-z) * binarypow(d.z, z);
}

__host__ __device__
inline void p2m_acc3(SCAL *__restrict__ M, int n, VEC d, SCAL q = 1)
{
// particle to multipole expansion of order n + accumulate to M
// d is the coordinate of the particle from the (near) expansion center
// O(n^2)
	SCAL C = (SCAL)paritysign(n) * (SCAL)inv_factorial(n) * q;
	int i = 0;
	for (int z = 0; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
			M[i++] += C * binarypow(d.x, x) * binarypow(d.y, n-x-z) * binarypow(d.z, z);
}

__host__ __device__
inline void p2m_traceless_acc3(SCAL *__restrict__ M, int n, VEC d, SCAL r)
{
// particle to traceless multipole expansion of order n + accumulate to M
// d is the unit vector of the coordinate of the particle from the (near) expansion center
// O(n^3)
	if (n == 0)
		M[0] += (SCAL)1;
	else
	{
		SCAL C = (SCAL)paritysign(n) * (SCAL)inv_factorial(n) * binarypow(r, n) / static_odfactorial(2*n-1);
		int i = 0;
		for (int z = 0; z <= 1; ++z)
			for (int x = n-z; x >= 0; --x)
			{
				int y = n-x-z;
				SCAL t1(0), t2(0);
				for (int k1 = 0; k1 <= x/2; ++k1)
				{
					t2 = 0;
					for (int k2 = 0; k2 <= y/2; ++k2)
					{
						int m = k1+k2;
						t2 += (SCAL)coeff13(n, m) * coeff2(y, k2) * binarypow(d.y, y - 2*k2);
					}
					t1 += t2 * coeff2(x, k1) * binarypow(d.x, x - 2*k1);
				}
				M[i++] += C * t1 * binarypow(d.z, z);
			}
	}
}

template <int n>
__host__ __device__
inline void static_p2m_acc3_(SCAL *__restrict__ M, VEC d, SCAL q)
{
	SCAL C = (SCAL)paritysign(n) * (SCAL)inv_factorial(n) * q;
	int i = 0;
#pragma unroll
	for (int z = 0; z <= n; ++z)
#pragma unroll
		for (int x = n-z; x >= 0; --x)
			M[i++] += C * binarypow(d.x, x) * binarypow(d.y, n-x-z) * binarypow(d.z, z);
}

__host__ __device__
inline void static_p2m_acc3(SCAL *__restrict__ M, int n, VEC d, SCAL q = 1)
{
// particle to multipole expansion of order n + accumulate to M
// d is the coordinate of the particle from the (near) expansion center
// O(n^2)
	switch (n)
	{
		case 0:
			static_p2m_acc3_<0>(M, d, q);
			break;
		case 1:
			static_p2m_acc3_<1>(M, d, q);
			break;
		case 2:
			static_p2m_acc3_<2>(M, d, q);
			break;
		case 3:
			static_p2m_acc3_<3>(M, d, q);
			break;
		case 4:
			static_p2m_acc3_<4>(M, d, q);
			break;
		case 5:
			static_p2m_acc3_<5>(M, d, q);
			break;
		default:
			p2m_acc3(M, n, d, q);
			break;
	}
}

__host__ __device__
inline void p2l3(SCAL *__restrict__ L, int n, VEC d, SCAL r)
{
// particle to local expansion of order n
// d is the unit vector of the coordinate of the particle from the (far) expansion center
// O(n^3)
	SCAL c = (SCAL)paritysign(n) * (SCAL)inv_factorial(n);
	gradient3(L, n, d, r, c);
	traceless_refine3(L, n);
}

__host__ __device__
inline void m2m3(SCAL *__restrict__ Mout, const SCAL *__restrict__ Mtuple, int n, VEC d)
{
// multipole to multipole expansion
// Mtuple is a tuple of multipole tensors of orders from 0 to n (inclusive)
// returns a multipole tensor of order n
// d is the shift from the old position to the new position
// O(n^5)
	int i = 0;
	SCAL C = inv_factorial(n);
	for (int z = 0; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
		{
			int y = n-x-z;
			SCAL t{};
			for (int m = 0; m <= n; ++m)
			{
				const SCAL *__restrict__ Mo = Mtuple + symmetricoffset3(n - m);
				SCAL c(0), c2;
				for (int k1 = 0; k1 <= min(x, m); ++k1)
				{
					c2 = 0;
					for (int k3 = max(0, m-k1-y); k3 <= min(z, m-k1); ++k3)
					{
						int k2 = m-k1-k3; // k2 < y => m-k1-k3 < y => k3 > m-k1-y
						int index = symmetric_i_x_z(x-k1, z-k3, n-m);
						c2 += binomial(y, k2) * binomial(z, k3)
						    * binarypow(d.y, k2) * binarypow(d.z, k3) * Mo[index];
					}
					c += c2 * binomial(x, k1) * binarypow(d.x, k1);
				}
				t += c * static_factorial(n-m);
			}
			Mout[i++] = C * t;
		}
}

__host__ __device__
inline void m2m_acc3(SCAL *__restrict__ Mout, const SCAL *__restrict__ Mtuple, int n, VEC d)
{
// multipole to multipole expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to n (inclusive)
// returns a multipole tensor of order n
// d is the shift from the old position to the new position
// O(n^5)
	int i = 0;
	SCAL C = inv_factorial(n);
	for (int z = 0; z <= n; ++z)
		for (int x = n-z; x >= 0; --x)
		{
			int y = n-x-z;
			SCAL t{};
			for (int m = 0; m <= n; ++m)
			{
				const SCAL *__restrict__ Mo = Mtuple + symmetricoffset3(n - m);
				SCAL c(0), c2;
				for (int k1 = 0; k1 <= min(x, m); ++k1)
				{
					c2 = 0;
					for (int k3 = max(0, m-k1-y); k3 <= min(z, m-k1); ++k3)
					{
						int k2 = m-k1-k3; // k2 < y => m-k1-k3 < y => k3 > m-k1-y
						int index = symmetric_i_x_z(x-k1, z-k3, n-m);
						c2 += (SCAL)binomial(y, k2) * binomial(z, k3)
						    * binarypow(d.y, k2) * binarypow(d.z, k3) * Mo[index];
					}
					c += c2 * binomial(x, k1) * binarypow(d.x, k1);
				}
				t += c * static_factorial(n-m);
			}
			Mout[i++] += C * t;
		}
}

__host__ __device__
inline void m2m_traceless_acc3(SCAL *__restrict__ Mout, SCAL *__restrict__ temp,
                               const SCAL *__restrict__ Mtuple, int n, VEC d, SCAL r)
{
// traceless multipole to multipole expansion + accumulate
// Mtuple is a tuple of traceless multipole tensors of orders from 0 to n (inclusive)
// returns a traceless multipole tensor of order n
// d is the (unit vector) shift from the old position to the new position
// temp is a temporary memory that needs at least 2*n+1 elements (independent for each thread)
// O(n^3)
	if (n == 0)
		Mout[0] += Mtuple[0];
	else
	{
		for (int m = 0; m <= n; ++m)
		{
			int i = 0;
			const SCAL *__restrict__ Mo = Mtuple + tracelessoffset3(n - m);
			SCAL C = inv_factorial(m);
			tracelesspow3(temp, m, d, r);
			for (int z = 0; z <= 1; ++z)
				for (int x = n-z; x >= 0; --x)
				{
					int y = n-x-z;
					SCAL t{};
					for (int k1 = 0; k1 <= min(x, m); ++k1)
						for (int k3 = max(0, m-k1-y); k3 <= min(z, m-k1); ++k3)
							t += traceless_A_x_z(temp, k1, k3, m) * traceless_A_x_z(Mo, x-k1, z-k3, n-m);
					Mout[i++] += C * t;
				}
		}
	}
}

template <int n>
__host__ __device__
inline void static_m2m_acc3_(SCAL *__restrict__ Mout, const SCAL *__restrict__ Mtuple, VEC d)
{
	int i = 0;
	SCAL C = inv_factorial(n);
#pragma unroll
	for (int z = 0; z <= n; ++z)
#pragma unroll
		for (int x = n-z; x >= 0; --x)
		{
			int y = n-x-z;
			SCAL t{};
#pragma unroll
			for (int m = 0; m <= n; ++m)
			{
				const SCAL *__restrict__ Mo = Mtuple + symmetricoffset3(n - m);
				SCAL c(0), c2;
#pragma unroll
				for (int k1 = 0; k1 <= min(x, m); ++k1)
				{
					c2 = 0;
#pragma unroll
					for (int k3 = max(0, m-k1-y); k3 <= min(z, m-k1); ++k3)
					{
						int k2 = m-k1-k3; // k2 < y => m-k1-k3 < y => k3 > m-k1-y
						int index = symmetric_i_x_z(x-k1, z-k3, n-m);
						c2 += binomial(y, k2) * binomial(z, k3)
						    * binarypow(d.y, k2) * binarypow(d.z, k3) * Mo[index];
					}
					c += c2 * binomial(x, k1) * binarypow(d.x, k1);
				}
				t += c * static_factorial(n-m);
			}
			Mout[i++] += C * t;
		}
}

__host__ __device__
inline void static_m2m_acc3(SCAL *__restrict__ Mout, const SCAL *__restrict__ Mtuple, int n, VEC d)
{
// multipole to multipole expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to n (inclusive)
// returns a multipole tensor of order n
// d is the shift from the old position to the new position
// O(n^5)
	switch (n)
	{
		case 0:
			static_m2m_acc3_<0>(Mout, Mtuple, d);
			break;
		case 1:
			static_m2m_acc3_<1>(Mout, Mtuple, d);
			break;
		case 2:
			static_m2m_acc3_<2>(Mout, Mtuple, d);
			break;
		case 3:
			static_m2m_acc3_<3>(Mout, Mtuple, d);
			break;
		case 4:
			static_m2m_acc3_<4>(Mout, Mtuple, d);
			break;
		case 5:
			static_m2m_acc3_<5>(Mout, Mtuple, d);
			break;
		default:
			m2m_acc3(Mout, Mtuple, n, d);
			break;
	}
}

template <bool b_atomic = false, bool no_dipole = false>
__host__ __device__
inline void m2l_acc3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple,
                     int nM, int nL, VEC d, SCAL r, int minm = 0, int maxm = -1, int maxn = -1)
{
// symmetric multipole to traceless local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the unit vector of the position of the local expansion (L) w.r.t. the multipole (M)
// temp is a temporary memory that needs at least (maxm+1)*(maxm+2)/2 elements (independent for each thread)
// maxm = nM+nL by default
// O(maxm*nL^2*nM^2)
	maxm = (maxm == -1) ? nM+nL : maxm;
	maxn = (maxn == -1) ? nL : maxn;
	SCAL scal = binarypow(r, maxm+1) * inv_factorial(maxm); // rescaling to avoid overflowing for single-precision fp
	for (int m = minm; m <= maxm; ++m)
	{
		gradient3(temp, m, d, r, scal);
		traceless_refine3(temp, m);
		for (int n = max(minm, m-nM); n <= min(maxn, m); ++n)
		{
			int mn = m-n; // 0 <= mn <= nM
			if (no_dipole && mn == 1)
				continue;
			SCAL C = inv_factorial(n)/scal;
			contract_traceless_ma3<b_atomic>(Ltuple + tracelessoffset3(n), Mtuple + symmetricoffset3(mn), temp, C, mn, m);
		}
	}
}

template <bool b_atomic = false, bool no_dipole = false>
__device__
inline void m2l_acc_coalesced3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple,
                               int nM, int nL, VEC d, SCAL r, int tid, int bdim, int mask,
                               int minm = 0, int maxm = -1, int maxn = -1)
{
// symmetric multipole to traceless local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the unit vector of the position of the local expansion (L) w.r.t. the multipole (M)
// temp is a temporary memory that needs at least (maxm+1)*(maxm+2)/2 elements (for every `bdim` threads)
// maxm = nM+nL by default
// O(maxm*nL^2*nM^2)
	maxm = (maxm == -1) ? nM+nL : maxm;
	maxn = (maxn == -1) ? nL : maxn;
	SCAL scal = binarypow(r, maxm+1) * inv_factorial(maxm); // rescaling to avoid overflowing for single-precision fp
	for (int m = minm; m <= maxm; ++m)
	{
		gradient_coalesced3(temp, m, d, r, tid, bdim, scal);
		__syncwarp(mask);
		traceless_refine_coalesced3(temp, m, tid, bdim, mask);
		for (int n = max(minm, m-nM); n <= min(maxn, m); ++n)
		{
			int mn = m-n; // 0 <= mn <= nM
			if (no_dipole && mn == 1)
				continue;
			SCAL C = inv_factorial(n)/scal;
			contract_traceless_ma_coalesced3<b_atomic>(Ltuple + tracelessoffset3(n), Mtuple + symmetricoffset3(mn), temp, C, mn, m, tid, bdim);
		}
		__syncwarp(mask);
	}
}

template <bool b_atomic = false, bool no_dipole = false>
__device__
inline void m2l_acc_coalesced_tuple3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple,
                                     int nM, int nL, VEC d, SCAL r, int tid, int bdim, int mask,
                                     int minm = 0, int maxm = -1, int maxn = -1)
{
// symmetric multipole to traceless local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the unit vector of the position of the local expansion (L) w.r.t. the multipole (M)
// temp is a temporary memory that needs at least (maxm+1)*(maxm+2)*(maxm+3)/6 elements (for every `bdim` threads)
// maxm = nM+nL by default
// O(maxm*nL^2*nM^2)
	maxm = (maxm == -1) ? nM+nL : maxm;
	maxn = (maxn == -1) ? nL : maxn;
	SCAL scal = binarypow(r, maxm+1) * inv_factorial(maxm); // rescaling to avoid overflowing for single-precision fp
	for (int m = minm; m <= maxm; ++m)
	{
		gradient_coalesced3(temp + symmetricoffset3(m), m, d, r, tid, bdim, scal);
		__syncwarp(mask);
		traceless_refine_coalesced3(temp + symmetricoffset3(m), m, tid, bdim, mask);
	}
	for (int mn = 0; mn <= nM; ++mn)
	{
		if (no_dipole && mn == 1)
			continue;
		contract_traceless_ma_coalesced_tuple3<b_atomic>
			(Ltuple, temp, Mtuple + symmetricoffset3(mn), 1/scal, mn+minm, min(maxm, mn+maxn), mn, tid, bdim);
		__syncwarp(mask);
	}
}

template <bool b_atomic = false, bool no_dipole = false>
__host__ __device__
inline void m2l_traceless_acc3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple,
                               int nM, int nL, VEC d, SCAL r, int minm = 0, int maxm = -1, int maxn = -1)
{
// traceless multipole to traceless local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the unit vector of the position of the local expansion (L) w.r.t. the multipole (M)
// temp is a temporary memory that needs at least 2*(nM+nL)+1 elements (independent for each thread)
// O((nM+nL)*nL^2*nM^2)
	maxm = (maxm == -1) ? nM+nL : maxm;
	maxn = (maxn == -1) ? nL : maxn;
	for (int m = minm; m <= maxm; ++m)
	{
		gradient3(temp, m, d, r);
		for (int n = max(minm, m-nM); n <= min(maxn, m); ++n)
		{
			int mn = m-n; // 0 <= mn <= nM
			if (no_dipole && mn == 1)
				continue;
			SCAL C = inv_factorial(n);
			contract_traceless2_ma3<b_atomic>(Ltuple + tracelessoffset3(n), Mtuple + tracelessoffset3(mn), temp, C, mn, m);
		}
	}
}

template <int n, int nmax, int m, bool traceless, bool b_atomic, bool no_dipole>
__host__ __device__
inline void static_m2l_inner2_3(SCAL *__restrict__ Ltuple, const SCAL *__restrict__ grad,
                                const SCAL *__restrict__ Mtuple, SCAL scal)
{
// O(n^2 * m * nmax)
	constexpr int mn = m-n; // 0 <= mn <= nM
	if constexpr (!no_dipole || mn != 1)
	{
		SCAL C = inv_factorial(n)/scal;
		if constexpr (traceless)
			static_contract_traceless2_ma3<mn, m, b_atomic>(Ltuple + tracelessoffset3(n), Mtuple + tracelessoffset3(mn), grad, C);
		else
			static_contract_traceless_ma3<mn, m, b_atomic>(Ltuple + tracelessoffset3(n), Mtuple + symmetricoffset3(mn), grad, C);
	}
	if constexpr (n+1 <= nmax)
		static_m2l_inner2_3<n+1, nmax, m, traceless, b_atomic, no_dipole>(Ltuple, grad, Mtuple, scal);
}

template <int m, int N, int minm, int maxm, int maxn, bool traceless, bool b_atomic, bool no_dipole>
__host__ __device__
inline void static_m2l_inner_3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ grad,
                               const SCAL *__restrict__ Mtuple, VEC d, SCAL r)
{
// O((m^3 + N^3 * m) * N)
	SCAL scal = binarypow(r, m+1); // rescaling to avoid overflowing for single-precision fp
	static_gradient3<m>(grad, d, r, scal);
	if constexpr (!traceless)
		static_traceless_refine3<m>(grad);
	static_m2l_inner2_3<static_max(minm, m-N), static_min(maxn, m), m, traceless, b_atomic, no_dipole>(Ltuple, grad, Mtuple, scal);

	if constexpr (m+1 <= maxm)
		static_m2l_inner_3<m+1, N, minm, maxm, maxn, traceless, b_atomic, no_dipole>(Ltuple, grad, Mtuple, d, r);
}

template <int N, int minm, int maxm, int maxn, bool traceless, bool b_atomic, bool no_dipole>
__host__ __device__
inline void static_m2l_acc3_(SCAL *__restrict__ Ltuple, SCAL *__restrict__ grad,
                             const SCAL *__restrict__ Mtuple, VEC d, SCAL r)
{
// O(N^5) - traceless
	constexpr int maxm_ = (maxm == -1) ? 2*N : ((maxm == -2) ? N : maxm);
	constexpr int maxn_ = (maxn == -1) ? N : ((maxn == -2) ? N-2 : maxn);

	static_m2l_inner_3<minm, N, minm, maxm_, maxn_, traceless, b_atomic, no_dipole>(Ltuple, grad, Mtuple, d, r);
}

template <int minm = 0, int maxm = -1, bool traceless = true, bool b_atomic = false, bool no_dipole = false, int maxn = -1>
__host__ __device__
inline void static_m2l_acc3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp,
							const SCAL *__restrict__ Mtuple, int N, VEC d, SCAL r)
{
// multipole to local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the unit vector of the position of the local expansion (L) w.r.t. the multipole (M)
// O(N^5) - traceless
	switch (N)
	{
		case 1:
			static_m2l_acc3_<1, minm, maxm, maxn, traceless, b_atomic, no_dipole>(Ltuple, temp, Mtuple, d, r);
			break;
		case 2:
			static_m2l_acc3_<2, minm, maxm, maxn, traceless, b_atomic, no_dipole>(Ltuple, temp, Mtuple, d, r);
			break;
		case 3:
			static_m2l_acc3_<3, minm, maxm, maxn, traceless, b_atomic, no_dipole>(Ltuple, temp, Mtuple, d, r);
			break;
		case 4:
			static_m2l_acc3_<4, minm, maxm, maxn, traceless, b_atomic, no_dipole>(Ltuple, temp, Mtuple, d, r);
			break;
		case 5:
			static_m2l_acc3_<5, minm, maxm, maxn, traceless, b_atomic, no_dipole>(Ltuple, temp, Mtuple, d, r);
			break;
		default:
			if constexpr (traceless)
				m2l_traceless_acc3<b_atomic, no_dipole>
					(Ltuple, temp, Mtuple, N, N, d, r, minm, (maxm == -2) ? N : maxm, (maxn == -2) ? N-2 : maxn);
			else
				m2l_acc3<b_atomic, no_dipole>(Ltuple, temp, Mtuple, N, N, d, r, minm, (maxm == -2) ? N : maxm, (maxn == -2) ? N-2 : maxn);
			break;
	}
}

template <int m, int maxm, int bdim>
__device__
inline void static_m2l_coalesced_grad3(SCAL *__restrict__ grad, VEC d, SCAL r, SCAL scal, int tid, int mask)
{
	static_gradient_coalesced3<m, bdim>(grad + symmetricoffset3(m), d, r, tid, scal);
	__syncwarp(mask);
	traceless_refine_coalesced3(grad + symmetricoffset3(m), m, tid, bdim, mask);

	if constexpr (m+1 <= maxm)
		static_m2l_coalesced_grad3<m+1, maxm, bdim>(grad, d, r, scal, tid, mask);
}

template <int mn, int N, int minm, int maxm, int maxn, int bdim, bool b_atomic, bool no_dipole>
__device__
inline void static_m2l_coalesced_contract3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ grad, const SCAL *__restrict__ Mtuple,
                                           VEC d, SCAL r, SCAL scal, int tid, int mask)
{
	if constexpr (no_dipole && mn == 1)
	{
		static_contract_traceless_ma_coalesced_tuple3<mn+minm, static_min(maxm, mn+maxn), mn, bdim, b_atomic>
			(Ltuple, grad, Mtuple + symmetricoffset3(mn), 1/scal, tid);
		__syncwarp(mask);
	}

	if constexpr (mn+1 <= N)
		static_m2l_coalesced_contract3<mn+1, N, minm, maxm, maxn, bdim, b_atomic, no_dipole>(Ltuple, grad, Mtuple, d, r, scal, tid, mask);
}

template <int N, int minm, int maxm, int maxn, int bdim, bool b_atomic, bool no_dipole>
__device__
inline void static_m2l_coalesced_inner3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ grad, const SCAL *__restrict__ Mtuple,
                                        VEC d, SCAL r, int tid, int mask)
{
	SCAL scal = binarypow(r, maxm+1) * inv_factorial(maxm); // rescaling to avoid overflowing for single-precision fp
	for (int m = minm; m <= maxm; ++m)
	{
		gradient_coalesced3(grad + symmetricoffset3(m), m, d, r, tid, bdim, scal);
		__syncwarp(mask);
		traceless_refine_coalesced3(grad + symmetricoffset3(m), m, tid, bdim, mask);
	}
	static_m2l_coalesced_contract3<0, N, minm, maxm, maxn, bdim, b_atomic, no_dipole>(Ltuple, grad, Mtuple, d, r, scal, tid, mask);
}

template <int N, int minm, int maxm, int maxn, bool b_atomic, bool no_dipole>
__host__ __device__
inline void static_m2l_acc_coalesced3_(SCAL *__restrict__ Ltuple, SCAL *__restrict__ grad, const SCAL *__restrict__ Mtuple,
                                       VEC d, SCAL r, int tid, int bdim, int mask)
{
	constexpr int maxm_ = (maxm == -1) ? 2*N : ((maxm == -2) ? N : maxm);
	constexpr int maxn_ = (maxn == -1) ? N : ((maxn == -2) ? N-2 : maxn);

	switch (bdim)
	{
		case 2:
			static_m2l_coalesced_inner3<N, minm, maxm_, maxn_, 2, b_atomic, no_dipole>(Ltuple, grad, Mtuple, d, r, tid, mask);
			break;
		case 4:
			static_m2l_coalesced_inner3<N, minm, maxm_, maxn_, 4, b_atomic, no_dipole>(Ltuple, grad, Mtuple, d, r, tid, mask);
			break;
		case 8:
			static_m2l_coalesced_inner3<N, minm, maxm_, maxn_, 8, b_atomic, no_dipole>(Ltuple, grad, Mtuple, d, r, tid, mask);
			break;
		case 16:
			static_m2l_coalesced_inner3<N, minm, maxm_, maxn_, 16, b_atomic, no_dipole>(Ltuple, grad, Mtuple, d, r, tid, mask);
			break;
		case 32:
			static_m2l_coalesced_inner3<N, minm, maxm_, maxn_, 32, b_atomic, no_dipole>(Ltuple, grad, Mtuple, d, r, tid, mask);
			break;
		default:
			m2l_acc_coalesced_tuple3<b_atomic, no_dipole>(Ltuple, grad, Mtuple, N, N, d, r, tid, bdim, mask, minm, maxm, maxn);
	}
}

template <int minm = 0, int maxm = -1, bool b_atomic = false, bool no_dipole = false, int maxn = -1>
__device__
inline void static_m2l_acc_coalesced3(SCAL *__restrict__ Ltuple, SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple,
                                      int N, VEC d, SCAL r, int tid, int bdim, int mask)
{
// multipole to local expansion + accumulate
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// d is the unit vector of the position of the local expansion (L) w.r.t. the multipole (M)
	switch (N)
	{
		case 1:
			static_m2l_acc_coalesced3_<1, minm, maxm, maxn, b_atomic, no_dipole>(Ltuple, temp, Mtuple, d, r, tid, bdim, mask);
			break;
		case 2:
			static_m2l_acc_coalesced3_<2, minm, maxm, maxn, b_atomic, no_dipole>(Ltuple, temp, Mtuple, d, r, tid, bdim, mask);
			break;
		case 3:
			static_m2l_acc_coalesced3_<3, minm, maxm, maxn, b_atomic, no_dipole>(Ltuple, temp, Mtuple, d, r, tid, bdim, mask);
			break;
		case 4:
			static_m2l_acc_coalesced3_<4, minm, maxm, maxn, b_atomic, no_dipole>(Ltuple, temp, Mtuple, d, r, tid, bdim, mask);
			break;
		case 5:
			static_m2l_acc_coalesced3_<5, minm, maxm, maxn, b_atomic, no_dipole>(Ltuple, temp, Mtuple, d, r, tid, bdim, mask);
			break;
		default:
			m2l_acc_coalesced_tuple3<b_atomic, no_dipole>(Ltuple, temp, Mtuple, N, N, d, r, tid, bdim, mask,
				minm, (maxm == -2) ? N : maxm, (maxn == -2) ? N-2 : maxn);
			break;
	}
}

__host__ __device__
inline void l2l_acc3(SCAL *__restrict__ Lout, SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple, int n, int nL, VEC d)
{
// local to local expansion + accumulate
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns a local expansion tensor of order n
// d is the shift from the old position to the new position
// temp is a temporary memory that needs at least (nL-n+1)*(nL-n+2)/2 elements (independent for each thread)
	for (int m = n; m <= nL; ++m)
	{
		int mn = m-n;
		tensorpow3(temp, mn, d);
		SCAL C = binomial(m, mn);
		contract_traceless_ma3(Lout, Ltuple + symmetricoffset3(m), temp, C, m, mn);
	}
}

__host__ __device__
inline void l2l_traceless_acc3(SCAL *__restrict__ Lout, SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple,
                                                   int n, int nL, VEC d, SCAL r)
{
// local to local expansion + accumulate
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns a local expansion tensor of order n
// d is the unit vector from the old position to the new position
// temp is a temporary memory that needs at least 2*(nL-n)+1 elements (independent for each thread)
// O(((nL-n)^3 + n * (nL-n)^2)*nL) ~ O(nL^4) for nL times
	for (int m = n; m <= nL; ++m)
	{
		int mn = m-n;
		tracelesspow3(temp, mn, d, r);
		SCAL C = binomial(m, mn);
		contract_traceless2_ma3(Lout, Ltuple + tracelessoffset3(m), temp, C, m, mn);
	}
}

template <int m, int n, int nL, bool traceless>
__host__ __device__
inline void static_l2l_inner3(SCAL *__restrict__ Lout, SCAL *__restrict__ temp,
                              const SCAL *__restrict__ Ltuple, VEC d, SCAL r)
{
	constexpr int mn = m-n;
	SCAL C = binomial(m, mn);
	if constexpr (traceless)
	{
		static_tracelesspow3<mn>(temp, d, r);
		static_contract_traceless2_ma3<m, mn>(Lout, Ltuple + tracelessoffset3(m), temp, C);
	}
	else
	{
		static_tensorpow3<mn>(temp, d);
		static_contract_traceless_ma3<m, mn>(Lout, Ltuple + symmetricoffset3(m), temp, C);
	}

	if constexpr (m+1 <= nL)
		static_l2l_inner3<m+1, n, nL, traceless>(Lout, temp, Ltuple, d, r);
}

template <int n, int nL, bool traceless>
__host__ __device__
inline void static_l2l_acc_3(SCAL *__restrict__ Ltupleo, SCAL *__restrict__ temp,
                             const SCAL *__restrict__ Ltuplei, VEC d, SCAL r)
{
	static_l2l_inner3<n, n, nL, traceless>(Ltupleo + tracelessoffset3(n), temp, Ltuplei, d, r);

	if constexpr (n+1 <= nL)
		static_l2l_acc_3<n+1, nL, traceless>(Ltupleo, temp, Ltuplei, d, r);
}

template <int minn = 0, bool traceless = true>
__host__ __device__
inline void static_l2l_acc3(SCAL *__restrict__ Ltupleo, SCAL *__restrict__ temp,
                            const SCAL *__restrict__ Ltuplei, int nL, VEC d, SCAL r = 0)
{
// local to local expansion + accumulate
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns a local expansion tensor of order n
// d is direction from the old position to the new position (normalized if traceless = true)
// temp is a temporary memory that needs at least 2*(nL-n)+1 elements (independent for each thread)
// O(((nL-n)^3 + n * (nL-n)^2)*nL) ~ O(nL^4) for nL times
	switch (nL)
	{
		case 1:
			static_l2l_acc_3<minn, 1, traceless>(Ltupleo, temp, Ltuplei, d, r);
			break;
		case 2:
			static_l2l_acc_3<minn, 2, traceless>(Ltupleo, temp, Ltuplei, d, r);
			break;
		case 3:
			static_l2l_acc_3<minn, 3, traceless>(Ltupleo, temp, Ltuplei, d, r);
			break;
		case 4:
			static_l2l_acc_3<minn, 4, traceless>(Ltupleo, temp, Ltuplei, d, r);
			break;
		case 5:
			static_l2l_acc_3<minn, 5, traceless>(Ltupleo, temp, Ltuplei, d, r);
			break;
		default:
			for (int q = minn; q <= nL; ++q)
				if constexpr (traceless)
					l2l_traceless_acc3(Ltupleo + tracelessoffset3(q), temp, Ltuplei, q, nL, d, r);
				else
					l2l_acc3(Ltupleo + tracelessoffset3(q), temp, Ltuplei, q, nL, d);
			break;
	}
}

__host__ __device__
inline SCAL m2p_pot3(SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple, int nM, VEC d, SCAL r2)
{
// multipole to particle expansion (for potential)
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns the potential evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least (nM+1)*(nM+2)/2 + 1 elements (independent for each thread)
// O(nM^4)
	SCAL pot(0);
	for (int n = 0; n <= nM; ++n)
	{
		gradient3(temp+1, n, d, r2);
		contract3(temp, Mtuple + symmetricoffset3(n), temp+1, n, n);
		pot += temp[0];
	}
	return pot;
}

__host__ __device__
inline SCAL l2p_pot3(SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple, int nL, VEC d)
{
// local to particle expansion (for potential)
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns the potential evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least (nL+1)*(nL+2)/2 + 1 elements (independent for each thread)
// O(nM^4)
	SCAL pot(0);
	for (int n = 0; n <= nL; ++n)
	{
		tensorpow3(temp+1, n, d);
		contract3(temp, Ltuple + symmetricoffset3(n), temp+1, n, n);
		pot += temp[0];
	}
	return pot;
}

__host__ __device__
inline VEC m2p_field3(SCAL *__restrict__ temp, const SCAL *__restrict__ Mtuple, int nM, VEC d, SCAL r2)
{
// multipole to particle expansion (for field)
// Mtuple is a tuple of multipole tensors of orders from 0 to nM (inclusive)
// returns the field evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least (nM+2)*(nM+3)/2 + 3 elements (independent for each thread)
	VEC field{};
	for (int n = 0; n <= nM; ++n)
	{
		gradient3(temp+3, n+1, d, r2);
		contract3(temp, Mtuple + symmetricoffset3(n), temp+3, n, n+1);
		field.x -= temp[0];
		field.y -= temp[1];
		field.z -= temp[2];
	}
	return field;
}

__host__ __device__
inline VEC l2p_field3(SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple, int nL, VEC d)
{
// local to particle expansion (for field)
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns the field evaluated at distance d from the expansion center
// temp is a temporary memory that needs at least nL*(nL+1)/2 + 3 elements (independent for each thread)
	VEC field{};
	for (int n = 1; n <= nL; ++n)
	{
		tensorpow3(temp+3, n-1, d);
		contract3(temp, Ltuple + symmetricoffset3(n), temp+3, n, n-1);
		SCAL C = (SCAL)n;
		field.x -= C*temp[0];
		field.y -= C*temp[1];
		field.z -= C*temp[2];
	}
	return field;
}

__host__ __device__
inline VEC l2p_traceless_field3(SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple, int nL, VEC d, SCAL r)
{
// local to particle expansion (for field)
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns the field evaluated at distance d from the expansion center
// d is a unit vector
// temp is a temporary memory that needs at least 2*nL+2 elements (independent for each thread)
// O(nL^4)
	for (int i = 0; i < 3; ++i)
		temp[i] = (SCAL)0;
	for (int n = 1; n <= nL; ++n)
	{
		tracelesspow3(temp+3, n-1, d, r);
		SCAL C = (SCAL)n;
		contract_traceless2_ma3(temp, Ltuple + tracelessoffset3(n), temp+3, C, n, n-1);
	}
	return VEC{-temp[0], -temp[1], -temp[2]};
}

template <int n, int nL, bool traceless>
__host__ __device__
inline void static_l2p_field_inner3(SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple, VEC d, SCAL r)
{
	constexpr SCAL C = (SCAL)n;
	if constexpr (traceless)
	{
		static_tracelesspow3<n-1>(temp+3, d, r);
		static_contract_traceless2_ma3<n, n-1>(temp, Ltuple + tracelessoffset3(n), temp+3, C);
	}
	else
	{
		static_tensorpow3<n-1>(temp+3, d);
		static_contract_traceless_ma3<n, n-1>(temp, Ltuple + symmetricoffset3(n), temp+3, C);
	}

	if constexpr (n+1 <= nL)
		static_l2p_field_inner3<n+1, nL, traceless>(temp, Ltuple, d, r);
}

template <int nL, bool traceless>
__host__ __device__
inline VEC static_l2p_field_3(SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple, VEC d, SCAL r)
{
	for (int i = 0; i < 3; ++i)
		temp[i] = (SCAL)0;

	static_l2p_field_inner3<1, nL, traceless>(temp, Ltuple, d, r);

	return VEC{-temp[0], -temp[1], -temp[2]};
}

template <bool traceless = true>
__host__ __device__
inline VEC static_l2p_field3(SCAL *__restrict__ temp, const SCAL *__restrict__ Ltuple, int nL, VEC d, SCAL r = 0)
{
// local to particle expansion (for field)
// Ltuple is a tuple of local expansion tensors of orders from 0 to nL (inclusive)
// returns the field evaluated at distance d from the expansion center
// d is direction from the old position to the new position (normalized if traceless = true)
// temp is a temporary memory that needs at least 2*nL+2 elements (independent for each thread)
// O(nL^4)
	switch (nL)
	{
		case 1:
			return static_l2p_field_3<1, traceless>(temp, Ltuple, d, r);
		case 2:
			return static_l2p_field_3<2, traceless>(temp, Ltuple, d, r);
		case 3:
			return static_l2p_field_3<3, traceless>(temp, Ltuple, d, r);
		case 4:
			return static_l2p_field_3<4, traceless>(temp, Ltuple, d, r);
		case 5:
			return static_l2p_field_3<5, traceless>(temp, Ltuple, d, r);
		default:
			if constexpr (traceless)
				return l2p_traceless_field3(temp, Ltuple, nL, d, r);
			else
				return l2p_field3(temp, Ltuple, nL, d);
	}
}

#endif // !FMM_CART_BASE3_CUDA_H