A = randn(21,21);
B = randn(21,5);

tic
C = lsmin(A,B)
toc
tic
C = pinv(A)*B
toc

A = randn(21,31);
B = randn(21,5);

C = lsmin(A,B);
C = pinv(A)*B;

A = randn(21,11);
B = randn(21,5);

C = lsmin(A,B);
C = pinv(A)*B;
