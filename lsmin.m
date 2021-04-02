%LSMIN Least Squares Minimum Norm Solution
%  X = LSMIN(A,B)computes the minimum norm solution to a real linear least
%  squares problem:
%
%  Minimize 2-norm(| B - A*X |).
%
%  using the singular value decomposition (SVD) of A. A is an M-by-N matrix
%  which may be rank-deficient.
%
%  X = LSMIN(A,B,TOL) where the effective rank of A is determined by
%  treating as zero those singular values which are less than TOL times the
%  largest singular value.