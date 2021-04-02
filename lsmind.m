%LSMIND Least Squares Minimum Norm Solution
%  X = LSMIND(A,B) computes the minimum norm solution to a real linear
%  least squares problem:
%
%  Minimize 2-norm(| B - A*X |).
%
%  using the singular value decomposition (SVD) of A. A is an M-by-N matrix
%  which may be rank-deficient.
%
%  The problem is solved in three steps:
%   (1) Reduce the coefficient matrix A to bidiagonal form with
%       Householder transformations, reducing the original problem
%       into a "bidiagonal least squares problem" (BLS)
%   (2) Solve the BLS using a divide and conquer approach.
%   (3) Apply back all the Householder tranformations to solve
%       the original least squares problem.
%
%  X = LSMIND(A,B,TOL) where the effective rank of A is determined by
%  treating as zero those singular values which are less than TOL (<1)
%  times the largest singular value.