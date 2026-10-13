/* FATROP adapter for CasADi-generated stage functions (see stage_fatrop.py).
 *
 * Knows nothing about the vehicle or the event. A generated table
 * (fl_types[], fl_n_types) provides, per stage type, the CasADi C functions
 *   BAbt(x_kp1, u, x, p)                 -> (nu+nx+1) x nx_kp1
 *   b(x_kp1, u, x, p)                    -> nx_kp1
 *   RSQrqt(s, u, x, lam_dyn, lam_g, lam_gi, p) -> (nu+nx+1) x (nu+nx)
 *   Ggt(u, x, p) / g(u, x, p)            -> (nu+nx+1) x ng / ng
 *   Ggti(u, x, p) / gi(u, x, p)          -> (nu+nx+1) x ngi / ngi
 *   rq(s, u, x, p) / L(s, u, x, p)       -> nu+nx / 1
 * in FATROP's (u, x) ordering. Absent functions are NULL (no dynamics on the
 * last stage). Per-stage parameters, bounds and the initial guess are passed
 * in at fl_create, so one library serves every track.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <blasfeo.h>
#include "fatrop/ocp/OCPCInterface.h"

#ifndef casadi_int
#define casadi_int long long int
#endif
typedef int (*cas_fn)(const double **arg, double **res, casadi_int *iw, double *w, int mem);
typedef int (*cas_work)(casadi_int *sz_arg, casadi_int *sz_res, casadi_int *sz_iw, casadi_int *sz_w);

typedef struct {
    cas_fn f;
    cas_work work;
} fl_fun;

enum { F_BABT, F_B, F_RSQ, F_GGT, F_G, F_GGTI, F_GI, F_RQ, F_L, F_COUNT };

typedef struct {
    int nu, nx, nx_next, ng, ngi, np;
    fl_fun fn[F_COUNT];
} fl_type;

extern const fl_type fl_types[];
extern const int fl_n_types;

typedef struct {
    int K;
    int *type, *np_off, *bnd_off;
    double *params, *lb, *ub, *x0, *u0;
    int *x_off, *u_off;
    const double **arg;
    double **res;
    casadi_int *iw;
    double *w, *buf;
    struct FatropOcpCSolver *solver;
    struct FatropOcpCInterface iface;
} fl_problem;

static const fl_type *T(fl_problem *P, int k) { return &fl_types[P->type[k]]; }

static int call(fl_problem *P, const fl_fun *f, const double **args, int nargs, double *out) {
    for (int i = 0; i < nargs; ++i) P->arg[i] = args[i];
    P->res[0] = out;
    return f->f(P->arg, P->res, P->iw, P->w, 0);
}

/* pack a column-major m x n block into a blasfeo matrix */
static void pack(int m, int n, double *A, struct blasfeo_dmat *M) {
    if (m > 0 && n > 0) blasfeo_pack_dmat(m, n, A, m, M, 0, 0);
}

/* ---- dimensions ---- */
static fatrop_int get_nx(fatrop_int k, void *u) { return T(u, k)->nx; }
static fatrop_int get_nu(fatrop_int k, void *u) { return T(u, k)->nu; }
static fatrop_int get_ng(fatrop_int k, void *u) { return T(u, k)->ng; }
static fatrop_int get_ngi(fatrop_int k, void *u) { return T(u, k)->ngi; }
static fatrop_int get_np(fatrop_int k, void *u) { return T(u, k)->np; }
static fatrop_int get_npg(void *u) { (void)u; return 0; }
static fatrop_int get_K(void *u) { return ((fl_problem *)u)->K; }
static fatrop_int get_sp(double *p, fatrop_int k, void *u) {
    fl_problem *P = u;
    memcpy(p, P->params + P->np_off[k], sizeof(double) * T(P, k)->np);
    return 0;
}
static fatrop_int get_gp(double *p, void *u) { (void)p; (void)u; return 0; }
static fatrop_int get_bounds(double *lo, double *hi, fatrop_int k, void *u) {
    fl_problem *P = u;
    int n = T(P, k)->ngi;
    memcpy(lo, P->lb + P->bnd_off[k], sizeof(double) * n);
    memcpy(hi, P->ub + P->bnd_off[k], sizeof(double) * n);
    return 0;
}
static fatrop_int get_x_init(double *x, fatrop_int k, void *u) {
    fl_problem *P = u;
    memcpy(x, P->x0 + P->x_off[k], sizeof(double) * T(P, k)->nx);
    return 0;
}
static fatrop_int get_u_init(double *x, fatrop_int k, void *u) {
    fl_problem *P = u;
    memcpy(x, P->u0 + P->u_off[k], sizeof(double) * T(P, k)->nu);
    return 0;
}

/* ---- evaluations ---- */
static fatrop_int eval_BAbt(const double *xkp1, const double *uk, const double *xk, const double *sp,
                            const double *gp, struct blasfeo_dmat *res, fatrop_int k, void *u) {
    fl_problem *P = u;
    const fl_type *t = T(P, k);
    const double *a[] = {xkp1, uk, xk, sp};
    call(P, &t->fn[F_BABT], a, 4, P->buf);
    pack(t->nu + t->nx + 1, t->nx_next, P->buf, res);
    (void)gp;
    return 0;
}
static fatrop_int eval_b(const double *xkp1, const double *uk, const double *xk, const double *sp,
                         const double *gp, double *res, fatrop_int k, void *u) {
    fl_problem *P = u;
    const double *a[] = {xkp1, uk, xk, sp};
    (void)gp;
    return call(P, &T(P, k)->fn[F_B], a, 4, res);
}
static fatrop_int eval_RSQrqt(const double *s, const double *uk, const double *xk, const double *lam_dyn,
                              const double *lam_g, const double *lam_gi, const double *sp,
                              const double *gp, struct blasfeo_dmat *res, fatrop_int k, void *u) {
    fl_problem *P = u;
    const fl_type *t = T(P, k);
    const double *a[] = {s, uk, xk, lam_dyn, lam_g, lam_gi, sp};
    call(P, &t->fn[F_RSQ], a, 7, P->buf);
    pack(t->nu + t->nx + 1, t->nu + t->nx, P->buf, res);
    (void)gp;
    return 0;
}
static fatrop_int eval_Ggt(const double *uk, const double *xk, const double *sp, const double *gp,
                           struct blasfeo_dmat *res, fatrop_int k, void *u) {
    fl_problem *P = u;
    const fl_type *t = T(P, k);
    if (t->ng == 0) return 0;
    const double *a[] = {uk, xk, sp};
    call(P, &t->fn[F_GGT], a, 3, P->buf);
    pack(t->nu + t->nx + 1, t->ng, P->buf, res);
    (void)gp;
    return 0;
}
static fatrop_int eval_g(const double *uk, const double *xk, const double *sp, const double *gp,
                         double *res, fatrop_int k, void *u) {
    fl_problem *P = u;
    if (T(P, k)->ng == 0) return 0;
    const double *a[] = {uk, xk, sp};
    (void)gp;
    return call(P, &T(P, k)->fn[F_G], a, 3, res);
}
static fatrop_int eval_Ggti(const double *uk, const double *xk, const double *sp, const double *gp,
                            struct blasfeo_dmat *res, fatrop_int k, void *u) {
    fl_problem *P = u;
    const fl_type *t = T(P, k);
    if (t->ngi == 0) return 0;
    const double *a[] = {uk, xk, sp};
    call(P, &t->fn[F_GGTI], a, 3, P->buf);
    pack(t->nu + t->nx + 1, t->ngi, P->buf, res);
    (void)gp;
    return 0;
}
static fatrop_int eval_gi(const double *uk, const double *xk, const double *sp, const double *gp,
                          double *res, fatrop_int k, void *u) {
    fl_problem *P = u;
    if (T(P, k)->ngi == 0) return 0;
    const double *a[] = {uk, xk, sp};
    (void)gp;
    return call(P, &T(P, k)->fn[F_GI], a, 3, res);
}
static fatrop_int eval_rq(const double *s, const double *uk, const double *xk, const double *sp,
                          const double *gp, double *res, fatrop_int k, void *u) {
    fl_problem *P = u;
    const double *a[] = {s, uk, xk, sp};
    (void)gp;
    return call(P, &T(P, k)->fn[F_RQ], a, 4, res);
}
static fatrop_int eval_L(const double *s, const double *uk, const double *xk, const double *sp,
                         const double *gp, double *res, fatrop_int k, void *u) {
    fl_problem *P = u;
    const double *a[] = {s, uk, xk, sp};
    (void)gp;
    return call(P, &T(P, k)->fn[F_L], a, 4, res);
}

static int fl_verbose = 0;
static void fl_write(const char *msg, int num) { if (fl_verbose) fwrite(msg, 1, num, stdout); }
static void fl_flush(void) { if (fl_verbose) fflush(stdout); }

/* ---- exported API (ctypes) ---- */
void *fl_create(int K, const int *type, const double *params, const double *lb, const double *ub,
                const double *x0, const double *u0, int verbose) {
    fl_problem *P = calloc(1, sizeof(fl_problem));
    P->K = K;
    P->type = malloc(sizeof(int) * K);
    memcpy(P->type, type, sizeof(int) * K);
    P->np_off = malloc(sizeof(int) * K);
    P->bnd_off = malloc(sizeof(int) * K);
    P->x_off = malloc(sizeof(int) * K);
    P->u_off = malloc(sizeof(int) * K);
    int np = 0, nb = 0, nxs = 0, nus = 0;
    casadi_int sz_arg = 0, sz_res = 0, sz_iw = 0, sz_w = 0, buf = 0;
    for (int k = 0; k < K; ++k) {
        const fl_type *t = &fl_types[type[k]];
        P->np_off[k] = np; np += t->np;
        P->bnd_off[k] = nb; nb += t->ngi;
        P->x_off[k] = nxs; nxs += t->nx;
        P->u_off[k] = nus; nus += t->nu;
    }
    for (int i = 0; i < fl_n_types; ++i) {
        const fl_type *t = &fl_types[i];
        casadi_int m = (casadi_int)(t->nu + t->nx + 1) * (t->nu + t->nx + t->nx_next + t->ng + t->ngi + 1);
        if (m > buf) buf = m;
        for (int j = 0; j < F_COUNT; ++j) {
            if (!t->fn[j].work) continue;
            casadi_int a, r, iw, w;
            t->fn[j].work(&a, &r, &iw, &w);
            if (a > sz_arg) sz_arg = a;
            if (r > sz_res) sz_res = r;
            if (iw > sz_iw) sz_iw = iw;
            if (w > sz_w) sz_w = w;
        }
    }
    P->params = malloc(sizeof(double) * (np + 1));
    memcpy(P->params, params, sizeof(double) * np);
    P->lb = malloc(sizeof(double) * (nb + 1));
    P->ub = malloc(sizeof(double) * (nb + 1));
    memcpy(P->lb, lb, sizeof(double) * nb);
    memcpy(P->ub, ub, sizeof(double) * nb);
    P->x0 = malloc(sizeof(double) * (nxs + 1));
    P->u0 = malloc(sizeof(double) * (nus + 1));
    memcpy(P->x0, x0, sizeof(double) * nxs);
    memcpy(P->u0, u0, sizeof(double) * nus);
    P->arg = malloc(sizeof(double *) * (sz_arg + 8));
    P->res = malloc(sizeof(double *) * (sz_res + 2));
    P->iw = malloc(sizeof(casadi_int) * (sz_iw + 1));
    P->w = malloc(sizeof(double) * (sz_w + 1));
    P->buf = malloc(sizeof(double) * (buf + 1));

    struct FatropOcpCInterface *I = &P->iface;
    I->get_nx = get_nx; I->get_nu = get_nu; I->get_ng = get_ng; I->get_ng_ineq = get_ngi;
    I->get_n_stage_params = get_np; I->get_n_global_params = get_npg;
    I->get_default_stage_params = get_sp; I->get_default_global_params = get_gp;
    I->get_horizon_length = get_K;
    I->eval_BAbt = eval_BAbt; I->eval_RSQrqt = eval_RSQrqt; I->eval_Ggt = eval_Ggt;
    I->eval_Ggt_ineq = eval_Ggti; I->eval_b = eval_b; I->eval_g = eval_g; I->eval_gineq = eval_gi;
    I->eval_rq = eval_rq; I->eval_L = eval_L; I->get_bounds = get_bounds;
    I->get_initial_xk = get_x_init; I->get_initial_uk = get_u_init;
    I->user_data = P;
    fl_verbose = verbose;
    P->solver = fatrop_ocp_c_create(I, fl_write, fl_flush);
    return P;
}

int fl_set_option_double(void *h, const char *name, double v) {
    return fatrop_ocp_c_set_option_double(((fl_problem *)h)->solver, name, v);
}
int fl_set_option_int(void *h, const char *name, int v) {
    return fatrop_ocp_c_set_option_int(((fl_problem *)h)->solver, name, v);
}
int fl_set_option_bool(void *h, const char *name, int v) {
    return fatrop_ocp_c_set_option_bool(((fl_problem *)h)->solver, name, v);
}

void fl_set_initial(void *h, const double *x0, const double *u0) {
    fl_problem *P = h;
    int nxs = 0, nus = 0;
    for (int k = 0; k < P->K; ++k) { nxs += T(P, k)->nx; nus += T(P, k)->nu; }
    memcpy(P->x0, x0, sizeof(double) * nxs);
    memcpy(P->u0, u0, sizeof(double) * nus);
}

int fl_solve(void *h) { return fatrop_ocp_c_solve(((fl_problem *)h)->solver); }

/* primal layout per stage: (u_k, x_k); copied out as separate x and u arrays */
void fl_get_solution(void *h, double *x, double *u) {
    fl_problem *P = h;
    const struct blasfeo_dvec *v = fatrop_ocp_c_get_primal(P->solver);
    const struct FatropOcpCDims *d = fatrop_ocp_c_get_dims(P->solver);
    for (int k = 0; k < P->K; ++k) {
        const fl_type *t = T(P, k);
        int off = d->ux_offs[k];
        if (t->nu) blasfeo_unpack_dvec(t->nu, (struct blasfeo_dvec *)v, off, u + P->u_off[k], 1);
        blasfeo_unpack_dvec(t->nx, (struct blasfeo_dvec *)v, off + t->nu, x + P->x_off[k], 1);
    }
}

void fl_get_stats(void *h, double *out) {
    const struct FatropOcpCStats *s = fatrop_ocp_c_get_stats(((fl_problem *)h)->solver);
    double v[] = {s->compute_sd_time, s->duinf_time, s->eval_hess_time, s->eval_jac_time,
                  s->eval_cv_time, s->eval_grad_time, s->eval_obj_time, s->initialization_time,
                  s->time_total, s->iterations_count, s->return_flag};
    memcpy(out, v, sizeof(v));
}

void fl_destroy(void *h) {
    fl_problem *P = h;
    fatrop_ocp_c_destroy(P->solver);
    free(P->type); free(P->np_off); free(P->bnd_off); free(P->x_off); free(P->u_off);
    free(P->params); free(P->lb); free(P->ub); free(P->x0); free(P->u0);
    free(P->arg); free(P->res); free(P->iw); free(P->w); free(P->buf);
    free(P);
}
