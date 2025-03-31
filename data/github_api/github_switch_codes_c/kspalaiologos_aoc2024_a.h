
#ifndef _A_H
#define _A_H

// Headers
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <regex.h>

// Types
typedef int I;
typedef unsigned UI;
typedef short T;
typedef long L;
typedef FILE * Fp;
typedef void V;
typedef char C;
typedef double D;
typedef regex_t Re;
typedef regmatch_t Rm;
#define S static

// Vector
#define _VEC_DECL_TYPE(T, N) \
  typedef T v##N##T __attribute__ ((vector_size (N * sizeof(T))))
#define _VT(T, N) T __attribute__ ((vector_size (N * sizeof(T))))
_VEC_DECL_TYPE(I, 2); // v2I
_VEC_DECL_TYPE(C, 2); // v2C
#define V2I(v) v[0]][v[1]
#define VC(t, v) __builtin_convertvector(v, t)

// Reduction operators
#define _V_OP(v, op, id) \
  ({ typeof((v)[0]) _v = id; Fx((sizeof(v) / sizeof((v)[0])), _v = _v op (v)[x]); _v; })
#define Vor(v) _V_OP(v, |, 0)
#define Vand(v) _V_OP(v, &, ~0)

// Language
#define R return
#define B break
#define CO continue
#define _(a...) {return({a;});}
#define Fi(n,a...) for(typeof(n) i=0;i<n;i++){a;}
#define Fid(n,a...) for(typeof(n) i=n-1;i>=0;i--){a;}
#define Fj(n,a...) for(typeof(n) j=0;j<n;j++){a;}
#define Fx(n,a...) for(typeof(n) x=0;x<n;x++){a;}
#define Fy(n,a...) for(typeof(n) y=0;y<n;y++){a;}
#define F2d(n,a...) Fi(n, Fj(n, _VT(typeof(n), 2) ij = {i,j}; a))
#define F_(z,i,n,a...) for(typeof(n) z=i;z<n;z++){a;}
#define F$(x,y,z,a...) for(x;y;z){a;}
#define I(x,a...) if(x){a;}
#define J(a...) else I(a)
#define E(a...) else{a;}
#define W(c,a...) while(c){a;}
#define SW(e,a...) switch(e){a;}
#define SC(c, a...) case c: {a; break;}
#define SD(a...) default: {a; break;}
#define C3(a,c1,b,c2,c) (a c1 b && b c2 c)
#define CR0(b,c) (0 <= b && b < c)
#define ZM(a) bzero(a, sizeof(a))
#define CM(a,b) memcpy(a,b,sizeof(a))

// Hygienic heapsort w. comparator.
#define _UNIQUE() __COUNTER__
#define _VAR(name) _JOIN(name, _UNIQUE())
#define _JOIN(a, b) _JOIN_2(a, b)
#define _JOIN_2(a, b) a ## b
#define hsort_internal(x, n, cmp, _a, _b, _n, _i, _p, _c) { \
  L _n = n, _i = _n / 2, _p, _c; \
  F$ (typeof(x[0]) t,,x[_p] = t, \
    I (_i > 0, t = x[--_i]) \
    J (--_n == 0, break) \
    E (t = x[_n], x[_n] = x[0]) \
    _p = _i;  _c = _i * 2 + 1; \
    W (_c < _n, \
      I (_c + 1 < _n && \
        ({ typeof(x[0]) _a = x[_c + 1], _b = x[_c]; cmp; }), _c++) \
      I (({ typeof(x[0]) _a = x[_c], _b = t; cmp; }), \
        x[_p] = x[_c]; _p = _c; _c = _p * 2 + 1; \
      ) E(break)))}
#define hsort(x, n, _a, _b, cmp) \
  hsort_internal(x, n, cmp, _a, _b, _VAR(__n), _VAR(__i), _VAR(__p), _VAR(__c))

// Fall: 1 if predicate holds for all elements, 0 otherwise. Short-circuits.
// Fany: 1 if predicate holds for any element, 0 otherwise. Short-circuits.
#define Fall_internal(_v,z,i,n,a...) \
  ({ I _v = 1; F_(z, i, n, I(!({a;}), _v = 0; B)) _v; })
#define Fall(z,i,n,a...) Fall_internal(_VAR(__v),z,i,n,a)
#define Fany_internal(_v,z,i,n,a...) \
  ({ I _v = 0; F_(z, i, n, I(({a;}), _v = 1; B)) _v; })
#define Fany(z,i,n,a...) Fany_internal(_VAR(__v),z,i,n,a)

// Summing fold-like LTR construct.
#define Fisum(n,i,a...) ({ typeof(i) _v = i; Fi(n, _v += ({a;})); _v; })
#define Fjsum(n,j,a...) ({ typeof(j) _v = j; Fj(n, _v += ({a;})); _v; })
#define F2dsum(n,i,a...) ({ typeof(i) _v = i; F2d(n, _v += ({a;})); _v; })
#define Fiprod(n,i,a...) ({ typeof(i) _v = i; Fi(n, _v *= ({a;})); _v; })
#define Fjprod(n,j,a...) ({ typeof(j) _v = j; Fj(n, _v *= ({a;})); _v; })
#define F2dprod(n,i,a...) ({ typeof(i) _v = i; F2d(n, _v *= ({a;})); _v; })

#define Fimin(n,i,a...) ({ typeof(i) _v = i; Fi(n, _v = MIN(_v, ({a;}))); _v; })
#define Fimax(n,i,a...) ({ typeof(i) _v = i; Fi(n, _v = MAX(_v, ({a;}))); _v; })
#define Fimini(n,id,a...) ({ typeof(id) _m = id, _t; I _mi = -1; Fi(n, _t = ({a;}); I(_t < _m, _m = _t, _mi = i)); _mi; })
#define Fimaxi(n,id,a...) ({ typeof(id) _m = id, _t; I _mi = -1; Fi(n, _t = ({a;}); I(_t > _m, _m = _t, _mi = i)); _mi; })

// Swap
#define SWAP_internal(a,b,_t) { typeof(a) _t = a; a = b; b = _t; }
#define SWAP(a,b) SWAP_internal(a,b,_VAR(__t))

// High precision timer:
#include <time.h>
S V barrier() { asm volatile("mfence" ::: "memory"); }
S L rdtsc() { L a, d; asm volatile("rdtsc" : "=a" (a), "=d" (d)); R a | (d << 32); }
#define TIC() struct timespec __tic; L __tsc_tic; barrier(); clock_gettime(CLOCK_MONOTONIC, &__tic); __tsc_tic = rdtsc(); barrier()
#define TOC() ({ struct timespec __toc; barrier(); clock_gettime(CLOCK_MONOTONIC, &__toc); barrier(); \
  (__toc.tv_sec - __tic.tv_sec) + (__toc.tv_nsec - __tic.tv_nsec) / 1e9; })
#define TSC_TOC() ({ L __tsc_toc; barrier(); __tsc_toc = rdtsc(); barrier(); \
  __tsc_toc - __tsc_tic; })

// Main
#define M(a...) I main(V){TIC();a;printf("Elapsed: %f s, %ld cycles.\n",TOC(),TSC_TOC());}

// Queue
#define QPUSH2(x,y) q[qt][0] = x; q[qt++][1] = y
#define QPOP2(x,y) x = q[qh][0]; y = q[qh++][1]
#define QPUSH(x) q[qt++] = x
#define QPOP(x) x = q[qh++]
#define QINIT() I qh = 0, qt = 0
#define WQ(a...) W(qh < qt, a)

// Grid
#define CHARGRIDF(P,f,g,a...) \
  Fi(P, Fj(P, g[i][j] = getc(f); a) getc(f))
#define LINESF(P,f,a...) C buf[P], * p; W(p = fgets(buf, P, f), a)

// Min/max/signum/remainder
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define SGN(x) ((x > 0) - (x < 0))
#define REM(x,y) ((x % y + y) % y)

// Regex stuff
#define regcomp(r, e) regcomp(&r, e, REG_EXTENDED)

// ASAN stuff
const char * __asan_default_options() { return "detect_leaks=0"; }

// Static helpers
S Fp fget(C * n) {
  Fp f = fopen(n, "r"); I(!f, perror("fopen"); exit(1)); R f; }
S I scani(FILE * f, I * l)_(fscanf(f, "%d", l) != EOF)
S I scanl(FILE * f, L * l)_(fscanf(f, "%ld", l) != EOF)
S C * slurp(C * n) {
  Fp f = fget(n);
  fseek(f, 0, SEEK_END);
  L s = ftell(f);
  rewind(f);
  C * d = malloc(s + 1);
  fread(d, 1, s, f);
  d[s] = 0; R d; }
S V resi(I t1, I t2) { printf("T1: %d, T2: %d\n", t1, t2); }
S V resl(L t1, L t2) { printf("T1: %ld, T2: %ld\n", t1, t2); }

S L ilog10l(L n)_(L r = 0; W (n > 0, n /= 10; r++) r)

S v2I dir8[8] = {
  {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}
};
S v2I dir4cw[4] = {
  {-1, 0}, {0, 1}, {1, 0}, {0, -1}
};

S I mod_inv(I a, I m)_(a %= m; Fx(m, I((a * x) % m == 1, R x)) 1)
S I isnum(C c)_(isdigit(c) || c == '-')

#endif

