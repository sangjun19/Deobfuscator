// dear emacs, please treat this as -*- C++ -*-

#include <R.h>
#include <Rmath.h>
#include <Rdefines.h>


#include "internal.h"

static R_INLINE SEXP add_args (SEXP args, SEXP Snames, SEXP Pnames, SEXP Cnames)
{

  SEXP var;
  int v;

  PROTECT(args = VectorToPairList(args));

  // we construct the call from end to beginning
  // covariates, parameter, states, then time

  // Covariates
  for (v = LENGTH(Cnames)-1; v >= 0; v--) {
    var = NEW_NUMERIC(1);
    args = LCONS(var,args);
    UNPROTECT(1);
    PROTECT(args);
    SET_TAG(args,installChar(STRING_ELT(Cnames,v)));
  }

  // Parameters
  for (v = LENGTH(Pnames)-1; v >= 0; v--) {
    var = NEW_NUMERIC(1);
    args = LCONS(var,args);
    UNPROTECT(1);
    PROTECT(args);
    SET_TAG(args,installChar(STRING_ELT(Pnames,v)));
  }

  // Latent state variables
  for (v = LENGTH(Snames)-1; v >= 0; v--) {
    var = NEW_NUMERIC(1);
    args = LCONS(var,args);
    UNPROTECT(1);
    PROTECT(args);
    SET_TAG(args,installChar(STRING_ELT(Snames,v)));
  }

  // Time
  var = NEW_NUMERIC(1);
  args = LCONS(var,args);
  UNPROTECT(1);
  PROTECT(args);
  SET_TAG(args,install("t"));

  UNPROTECT(1);
  return args;

}

static R_INLINE SEXP eval_call (
                                SEXP fn, SEXP args,
                                double *t,
                                double *x, int nvar,
                                double *p, int npar,
                                double *c, int ncov)
{

  SEXP var = args, ans, ob;
  int v;

  *(REAL(CAR(var))) = *t; var = CDR(var);
  for (v = 0; v < nvar; v++, x++, var=CDR(var)) *(REAL(CAR(var))) = *x;
  for (v = 0; v < npar; v++, p++, var=CDR(var)) *(REAL(CAR(var))) = *p;
  for (v = 0; v < ncov; v++, c++, var=CDR(var)) *(REAL(CAR(var))) = *c;

  PROTECT(ob = LCONS(fn,args));
  PROTECT(ans = eval(ob,R_ClosureEnv(fn)));

  UNPROTECT(2);
  return ans;

}

static R_INLINE SEXP ret_array (int n, int nreps, int ntimes, SEXP names) {
  int dim[3] = {n, nreps, ntimes};
  const char *dimnm[3] = {"name", ".id", "time"};
  SEXP Y;

  PROTECT(Y = makearray(3,dim));
  setrownames(Y,names,3);
  fixdimnames(Y,dimnm,3);

  UNPROTECT(1);
  return Y;

}

SEXP do_emeasure (SEXP object, SEXP x, SEXP times, SEXP params, SEXP gnsi)
{
  pompfunmode mode = undef;
  int ntimes, nvars, npars, ncovars, nreps, nrepsx, nrepsp;
  int nobs = 0;
  SEXP Snames, Pnames, Cnames, Onames = R_NilValue;
  SEXP fn, args;
  SEXP pompfun;
  SEXP Y = R_NilValue;
  int *dim;
  lookup_table_t covariate_table;
  SEXP cvec;
  double *cov;

  PROTECT(times = AS_NUMERIC(times));
  ntimes = length(times);
  if (ntimes < 1)
    err("length('times') = 0, no work to do.");

  PROTECT(x = as_state_array(x));
  dim = INTEGER(GET_DIM(x));
  nvars = dim[0]; nrepsx = dim[1];

  if (ntimes != dim[2])
    err("length of 'times' and 3rd dimension of 'x' do not agree.");

  PROTECT(params = as_matrix(params));
  dim = INTEGER(GET_DIM(params));
  npars = dim[0]; nrepsp = dim[1];

  nreps = (nrepsp > nrepsx) ? nrepsp : nrepsx;

  if ((nreps % nrepsp != 0) || (nreps % nrepsx != 0))
    err("larger number of replicates is not a multiple of smaller.");

  PROTECT(pompfun = GET_SLOT(object,install("emeasure")));

  PROTECT(Snames = GET_ROWNAMES(GET_DIMNAMES(x)));
  PROTECT(Pnames = GET_ROWNAMES(GET_DIMNAMES(params)));
  PROTECT(Cnames = get_covariate_names(GET_SLOT(object,install("covar"))));
  PROTECT(Onames = GET_SLOT(pompfun,install("obsnames")));

  // set up the covariate table
  covariate_table = make_covariate_table(GET_SLOT(object,install("covar")),&ncovars);
  PROTECT(cvec = NEW_NUMERIC(ncovars));
  cov = REAL(cvec);

  // extract the user-defined function
  PROTECT(fn = pomp_fun_handler(pompfun,gnsi,&mode,Snames,Pnames,Onames,Cnames));

  // extract 'userdata' as pairlist
  PROTECT(args = GET_SLOT(object,install("userdata")));

  int nprotect = 11;
  int first = 1;

  // first do setup
  switch (mode) {

  case Rfun: {
    double *ys, *yt = 0;
    double *time = REAL(times), *xs = REAL(x), *ps = REAL(params);
    SEXP ans;
    int j, k;

    PROTECT(args = add_args(args,Snames,Pnames,Cnames)); nprotect++;

    for (k = 0; k < ntimes; k++, time++) { // loop over times

      R_CheckUserInterrupt();   // check for user interrupt

      table_lookup(&covariate_table,*time,cov); // interpolate the covariates

      for (j = 0; j < nreps; j++) { // loop over replicates

        if (first) {

          PROTECT(
                  ans = eval_call(
                                  fn,args,
                                  time,
                                  xs+nvars*((j%nrepsx)+nrepsx*k),nvars,
                                  ps+npars*(j%nrepsp),npars,
                                  cov,ncovars
                                  )
                  );

          nobs = LENGTH(ans);

          PROTECT(Onames = GET_NAMES(ans));
          if (invalid_names(Onames))
            err("'emeasure' must return a named numeric vector.");

          PROTECT(Y = ret_array(nobs,nreps,ntimes,Onames));

          nprotect += 3;

          yt = REAL(Y);
          ys = REAL(AS_NUMERIC(ans));

          memcpy(yt,ys,nobs*sizeof(double));
          yt += nobs;

          first = 0;

        } else {

          PROTECT(
                  ans = eval_call(
                                  fn,args,
                                  time,
                                  xs+nvars*((j%nrepsx)+nrepsx*k),nvars,
                                  ps+npars*(j%nrepsp),npars,
                                  cov,ncovars
                                  )
                  );

          if (LENGTH(ans) != nobs)
            err("'emeasure' returns variable-length results.");

          ys = REAL(AS_NUMERIC(ans));

          memcpy(yt,ys,nobs*sizeof(double));
          yt += nobs;

          UNPROTECT(1);

        }

      }
    }

  }

    break;

  case native: case regNative: {
    double *yt = 0, *xp, *pp;
    double *time = REAL(times), *xs = REAL(x), *ps = REAL(params);
    int *oidx, *sidx, *pidx, *cidx;
    pomp_emeasure *ff = NULL;
    int j, k;

    nobs = LENGTH(Onames);
    // extract observable, state, parameter covariate indices
    sidx = INTEGER(GET_SLOT(pompfun,install("stateindex")));
    pidx = INTEGER(GET_SLOT(pompfun,install("paramindex")));
    oidx = INTEGER(GET_SLOT(pompfun,install("obsindex")));
    cidx = INTEGER(GET_SLOT(pompfun,install("covarindex")));

    // address of native routine
    *((void **) (&ff)) = R_ExternalPtrAddr(fn);

    PROTECT(Y = ret_array(nobs,nreps,ntimes,Onames)); nprotect++;
    yt = REAL(Y);

    for (k = 0; k < ntimes; k++, time++) { // loop over times

      R_CheckUserInterrupt();   // check for user interrupt

      // interpolate the covar functions for the covariates
      table_lookup(&covariate_table,*time,cov);

      for (j = 0; j < nreps; j++, yt += nobs) { // loop over replicates

        xp = &xs[nvars*((j%nrepsx)+nrepsx*k)];
        pp = &ps[npars*(j%nrepsp)];

        (*ff)(yt,xp,pp,oidx,sidx,pidx,cidx,cov,*time);

      }
    }

  }

    break;

  default: {
    nobs = LENGTH(Onames);
    int dim[3] = {nobs, nreps, ntimes};
    const char *dimnm[3] = {"name",".id","time"};
    double *yt = 0;
    int i, n = nobs*nreps*ntimes;

    PROTECT(Y = makearray(3,dim)); nprotect++;
    setrownames(Y,Onames,3);
    fixdimnames(Y,dimnm,3);

    for (i = 0, yt = REAL(Y); i < n; i++, yt++) *yt = R_NaReal;

    warn("'emeasure' unspecified: NAs generated.");
  }

  }

  UNPROTECT(nprotect);
  return Y;
}
