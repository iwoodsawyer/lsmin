/*
 * Least Squares (divide-and-conquer SVD)
 *
 * X = lsmind(A,B)
 *
 * compile command:
 * mex -O lsmind.c libmwlapack.lib
 * or
 * mex -O lsmind.c libmwblas.lib libmwlapack.lib (>= R2007B)
 *
 * calls the SGELSD/CGELSD/DGELSD/ZGELSD named LAPACK function
 *
 * Ivo Houtzager
 */

#include "mex.h"
#include "matrix.h"
#include "math.h"
#include "lsmin.h"

void ls_double(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ptrdiff_t rank, *iwork, isize, swork, lwork, liwork, lrwork, nlvl, smlsiz, ipspec, N, info = 1;
    size_t cplx = 0, cplxa = 0, cplxb = 0, dc = 1;
    double *Ap, *Apr, *Api, *Bp, *Bpr, *Bpi, *Xpr ,*Xpi, *S;
    double *work, *rwork, *size, rsize, rcond, dminmn, dsmlsizp1, tmp;
    size_t ma, na, mb, nb, mina, ldb, element_size = sizeof(double);
    mwIndex i, j;
    mxClassID classid = mxDOUBLE_CLASS;
    mxComplexity cplxflag = mxREAL;

    /* check complex */
    if (mxIsComplex(prhs[0]) || mxIsComplex(prhs[1])) {
        cplxflag = mxCOMPLEX;
        cplx = 1;
        if (mxIsComplex(prhs[0])) {
            cplxa = 1;
        }
        if (mxIsComplex(prhs[1])) {
            cplxb = 1;
        }
        dc = 2;
    }

    /* check for proper dimensions*/
    ma = mxGetM(prhs[0]);
    na = mxGetN(prhs[0]);
    if (!mxIsNumeric(prhs[1]) || mxIsSparse(prhs[1])) {
        mexErrMsgTxt("Input B must be a full matrix.");
    }
    mb = mxGetM(prhs[1]);
    nb = mxGetN(prhs[1]);
    if (ma != mb) {
        mexErrMsgTxt("Number of rows of A and B must be equal." );
    }
    if (ma == 0 || na == 0 || nb == 0) {
        plhs[0] = mxCreateNumericMatrix(na,nb,classid,cplxflag);
        return;
    }

    /* load input and output matrices */
    Ap = mxMalloc(dc*ma*na*element_size);
    Apr = mxGetData(prhs[0]);
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    if (cplxa) {
        Api = mxGetImagData(prhs[0]);
        for (j=0; j<na; j++) {
            for (i=0; i<ma; i++) {
                Ap[j*2*ma+2*i] = Apr[j*ma+i];
                Ap[j*2*ma+2*i+1] = Api[j*ma+i];
            }
        }
    }
    else {
    #endif
        memcpy(Ap,Apr,dc*ma*na*element_size);
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    }
    #endif

    /* leading dimensions of B */
    ldb = max(ma,na);

    Bp = mxMalloc(dc*ldb*nb*element_size);
    Bpr = mxGetData(prhs[1]);
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    if (cplxb) {
        Bpi = mxGetImagData(prhs[1]);
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                Bp[j*2*ldb+2*i] = Bpr[j*mb+i];
                Bp[j*2*ldb+2*i+1] = Bpi[j*mb+i];
            }
        }
    }
    else {
    #endif
        if (ldb == mb) {
            memcpy(Bp,Bpr,dc*nb*mb*element_size);
        }
        else {
            for (j=0; j<nb; j++) {
                for (i=0; i<dc*mb; i++) {
                    Bp[j*dc*ldb+i] = Bpr[j*dc*mb+i];
                }
            }
        }
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    }
    #endif

    if (nrhs == 3) {
        rcond = mxGetScalar(prhs[2]);
    }
    else {
        /* use machine precision */
        rcond = -1.0;
    }

    /* allocate rank and matrix S */
    mina = min(ma,na);
    S = mxMalloc(mina*element_size);

    /* determine blocksize */
    size = mxMalloc(dc*element_size);
    lwork = -1;
    if (cplx) {
        zgelsd(&ma, &na, &nb, Ap, &ma, Bp, &ldb, S, &rcond, &rank, size, &lwork, &rsize, &isize, &info);
    }
    else {
        dgelsd(&ma, &na, &nb, Ap, &ma, Bp, &ldb, S, &rcond, &rank, size, &lwork, &isize, &info);
    }
    if (info != 0) {
        mxFree(size);
        mxFree(S);
        mxFree(Ap);
        mxFree(Bp);
        if (cplx) {
            mexErrMsgTxt("ZEGLSD not successful.");
        }
        else {
            mexErrMsgTxt("DEGLSD not successful.");
        }
    }
    lwork = size[0];

    /* allocate workspace */
    ipspec = 9;
    N = 0;
    if (cplx) {
        char *name = "ZGELSD";
        char *opts = "";
        smlsiz = ilaenv(&ipspec, name, opts, &N, &N, &N, &N, 6, 1);
    }
    else
    {
        char *name = "DGELSD";
        char *opts = "";
        smlsiz = ilaenv(&ipspec, name, opts, &N, &N, &N, &N, 6, 1);
    }
    dminmn = mina;
    dsmlsizp1 = smlsiz + 1;
    tmp = log(dminmn)/dsmlsizp1/log(2.0) + 1;
    nlvl = ceil(tmp);
    if (nlvl < 0) {
        nlvl = 0;
    }
    if (cplx) {
        swork = 2*ldb + ldb*nb;
    }
    else {
        swork = 12*ldb + 2*ldb*smlsiz + 8*ldb*nlvl + ldb*nb + (smlsiz+1)*2;
    }
    lwork = max(lwork,swork);
    work = mxMalloc(dc*lwork*element_size);

    /* We compute the size of iwork because DGELSD in older versions
    of LAPACK does not return it on a query call. */
    liwork = 3*mina*nlvl + 11*mina;
    if (liwork < 1) {
        liwork = 1;
    }
    iwork = mxMalloc(liwork*sizeof(ptrdiff_t));
    if (cplx) {
        lrwork = 10*ldb + 2*ldb*smlsiz + 8*nlvl + 3*smlsiz*nb + (smlsiz+1)*2;
        if (lrwork < 1) {
            lrwork = 1;
        }
        rwork = mxMalloc(lrwork*element_size);
    }

    /* calls the DGELSD function */
    if (cplx) {
        zgelsd(&ma, &na, &nb, Ap, &ma, Bp, &ldb, S, &rcond, &rank, work, &lwork, rwork, iwork, &info);
    }
    else {
        dgelsd(&ma, &na, &nb, Ap, &ma, Bp, &ldb, S, &rcond, &rank, work, &lwork, iwork, &info);
    }
    mxFree(work);
    if (cplx) {
        mxFree(rwork);
    }
    mxFree(iwork);
    mxFree(S);
    mxFree(Ap);
    if (info != 0) {
        mxFree(Bp);
        if (cplx) {
            mexErrMsgTxt("ZEGLSD not successful.");
        }
        else {
            mexErrMsgTxt("DEGLSD not successful.");
        }
    }

    /* copy output B to X matrix */
    plhs[0] = mxCreateNumericMatrix(na,nb,classid,cplxflag);
    Xpr = mxGetData(plhs[0]);
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    if (cplx) {
        Xpi = mxGetImagData(plhs[0]);
        for (j=0; j<nb; j++) {
            for (i=0; i<na; i++) {
                Xpr[j*na+i] = Bp[j*2*ldb+2*i];
                Xpi[j*na+i] = Bp[j*2*ldb+2*i+1];
            }
        }
    }
    else {
    #endif
        if (ldb == na) {
            memcpy(Xpr,Bp,dc*na*nb*element_size);
        }
        else {
            for (j=0; j<nb; j++) {
                for (i=0; i<dc*na; i++) {
                    Xpr[j*dc*na+i] = Bp[j*dc*ldb+i];
                }
            }
        }
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    }
    #endif
    mxFree(Bp);
}


void ls_single(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ptrdiff_t rank, *iwork, isize, swork, lwork, liwork, lrwork, nlvl, smlsiz, ipspec, N, info = 1;
    size_t cplx = 0, cplxa = 0, cplxb = 0, dc = 1;
    float *Ap, *Apr, *Api, *Bp, *Bpr, *Bpi, *Xpr ,*Xpi, *S;
    float *work, *rwork, *size, rsize, rcond, dminmn, dsmlsizp1, tmp;
    size_t ma, na, mb, nb, mina, ldb, element_size = sizeof(float);
    mwIndex i, j;
    mxClassID classid = mxSINGLE_CLASS;
    mxComplexity cplxflag = mxREAL;
    
    /* check complex */
    if (mxIsComplex(prhs[0]) || mxIsComplex(prhs[1])) {
        cplxflag = mxCOMPLEX;
        cplx = 1;
        if (mxIsComplex(prhs[0])) {
            cplxa = 1;
        }
        if (mxIsComplex(prhs[1])) {
            cplxb = 1;
        }
        dc = 2;
    }

    /* check for proper dimensions*/
    ma = mxGetM(prhs[0]);
    na = mxGetN(prhs[0]);
    if (!mxIsNumeric(prhs[1]) || mxIsSparse(prhs[1])) {
        mexErrMsgTxt("Input B must be a full matrix.");
    }
    mb = mxGetM(prhs[1]);
    nb = mxGetN(prhs[1]);
    if (ma != mb) {
        mexErrMsgTxt("Number of rows of A and B must be equal." );
    }
    if (ma == 0 || na == 0 || nb == 0) {
        plhs[0] = mxCreateNumericMatrix(na,nb,classid,cplxflag);
        return;
    }

    /* load input and output matrices */
    Ap = mxMalloc(dc*ma*na*element_size);
    Apr = mxGetData(prhs[0]);
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    if (cplxa) {
        Api = mxGetImagData(prhs[0]);
        for (j=0; j<na; j++) {
            for (i=0; i<ma; i++) {
                Ap[j*2*ma+2*i] = Apr[j*ma+i];
                Ap[j*2*ma+2*i+1] = Api[j*ma+i];
            }
        }
    }
    else {
    #endif
        memcpy(Ap,Apr,dc*ma*na*element_size);
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    }
    #endif

    /* leading dimensions of B */
    ldb = max(ma,na);

    Bp = mxMalloc(dc*ldb*nb*element_size);
    Bpr = mxGetData(prhs[1]);
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    if (cplxb) {
        Bpi = mxGetImagData(prhs[1]);
        for (j=0; j<nb; j++) {
            for (i=0; i<mb; i++) {
                Bp[j*2*ldb+2*i] = Bpr[j*mb+i];
                Bp[j*2*ldb+2*i+1] = Bpi[j*mb+i];
            }
        }
    }
    else {
    #endif
        if (ldb == mb) {
            memcpy(Bp,Bpr,dc*nb*mb*element_size);
        }
        else {
            for (j=0; j<nb; j++) {
                for (i=0; i<dc*mb; i++) {
                    Bp[j*dc*ldb+i] = Bpr[j*dc*mb+i];
                }
            }
        }
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    }
    #endif

    if (nrhs == 3) {
        rcond = mxGetScalar(prhs[2]);
    }
    else {
        /* use machine precision */
        rcond = -1.0;
    }

    /* allocate rank and matrix S */
    mina = min(ma,na);
    S = mxMalloc(mina*element_size);

    /* determine blocksize */
    size = mxMalloc(dc*element_size);
    lwork = -1;
    if (cplx) {
        cgelsd(&ma, &na, &nb, Ap, &ma, Bp, &ldb, S, &rcond, &rank, size, &lwork, &rsize, &isize, &info);
    }
    else {
        sgelsd(&ma, &na, &nb, Ap, &ma, Bp, &ldb, S, &rcond, &rank, size, &lwork, &isize, &info);
    }
    if (info != 0) {
        mxFree(size);
        mxFree(S);
        mxFree(Ap);
        mxFree(Bp);
        if (cplx) {
            mexErrMsgTxt("CEGLSD not successful.");
        }
        else {
            mexErrMsgTxt("SEGLSD not successful.");
        }
    }
    lwork = size[0];

    /* allocate workspace */
    ipspec = 9;
    N = 0;
    if (cplx) {
        char *name = "CGELSD";
        char *opts = "";
        smlsiz = ilaenv(&ipspec, name, opts, &N, &N, &N, &N, 6, 1);
    }
    else
    {
        char *name = "SGELSD";
        char *opts = "";
        smlsiz = ilaenv(&ipspec, name, opts, &N, &N, &N, &N, 6, 1);
    }
    dminmn = mina;
    dsmlsizp1 = smlsiz + 1;
    tmp = log(dminmn)/dsmlsizp1/log(2.0) + 1;
    nlvl = ceil(tmp);
    if (nlvl < 0) {
        nlvl = 0;
    }
    if (cplx) {
        swork = 2*ldb + ldb*nb;
    }
    else {
        swork = 12*ldb + 2*ldb*smlsiz + 8*ldb*nlvl + ldb*nb + (smlsiz+1)*2;
    }
    lwork = max(lwork,swork);
    work = mxMalloc(dc*lwork*element_size);

    /* We compute the size of iwork because DGELSD in older versions
    of LAPACK does not return it on a query call. */
    liwork = 3*mina*nlvl + 11*mina;
    if (liwork < 1) {
        liwork = 1;
    }
    iwork = mxMalloc(liwork*sizeof(ptrdiff_t));
    if (cplx) {
        lrwork = 10*ldb + 2*ldb*smlsiz + 8*nlvl + 3*smlsiz*nb + (smlsiz+1)*2;
        if (lrwork < 1) {
            lrwork = 1;
        }
        rwork = mxMalloc(lrwork*element_size);
    }

    /* calls the DGELSD function */
    if (cplx) {
        cgelsd(&ma, &na, &nb, Ap, &ma, Bp, &ldb, S, &rcond, &rank, work, &lwork, rwork, iwork, &info);
    }
    else {
        sgelsd(&ma, &na, &nb, Ap, &ma, Bp, &ldb, S, &rcond, &rank, work, &lwork, iwork, &info);
    }
    mxFree(work);
    if (cplx) {
        mxFree(rwork);
    }
    mxFree(iwork);
    mxFree(S);
    mxFree(Ap);
    if (info != 0) {
        mxFree(Bp);
        if (cplx) {
            mexErrMsgTxt("CEGLSD not successful.");
        }
        else {
            mexErrMsgTxt("SEGLSD not successful.");
        }
    }

    /* copy output B to X matrix */
    plhs[0] = mxCreateNumericMatrix(na,nb,classid,cplxflag);
    Xpr = mxGetData(plhs[0]);
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    if (cplx) {
        Xpi = mxGetImagData(plhs[0]);
        for (j=0; j<nb; j++) {
            for (i=0; i<na; i++) {
                Xpr[j*na+i] = Bp[j*2*ldb+2*i];
                Xpi[j*na+i] = Bp[j*2*ldb+2*i+1];
            }
        }
    }
    else {
    #endif
        if (ldb == na) {
            memcpy(Xpr,Bp,dc*na*nb*element_size);
        }
        else {
            for (j=0; j<nb; j++) {
                for (i=0; i<dc*na; i++) {
                    Xpr[j*dc*na+i] = Bp[j*dc*ldb+i];
                }
            }
        }
    #if !(MX_HAS_INTERLEAVED_COMPLEX)
    }
    #endif
    mxFree(Bp);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* check for proper number of arguments */
    if (nrhs < 2) {
        mexErrMsgTxt("LSMIND requires two or three input arguments.");
    }
    if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    if (!mxIsNumeric(prhs[0]) || mxIsSparse(prhs[0])) {
        mexErrMsgTxt( "Input must be a full matrix." );
    }
    if (mxIsDouble(prhs[0]) && mxIsDouble(prhs[1])) {
        ls_double(nlhs, plhs, nrhs, prhs);
    }
    else if (mxIsSingle(prhs[0]) && mxIsSingle(prhs[1])) {
        ls_single(nlhs, plhs, nrhs, prhs);
    }
    else {
        mexErrMsgTxt("Class is not supported or not similar for both inputs.");
    }
}
