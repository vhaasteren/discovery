"""Class-quantized white-noise Gram (`discovery.classgram`).

Synthetic pulsar: 900 TOAs, three backends, two of them subbanded (epochs of
4-8 TOAs with a x4 sigma spread inside every epoch), one singleton-only. The
reference is a dense Sherman-Morrison Gram `W^T N^-1 W` with `N` materialized.
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import discovery as ds  # noqa: E402
from discovery import _kernels  # noqa: E402
from discovery import classgram as cg  # noqa: E402
from discovery import metamath  # noqa: E402
from discovery import metamatrix  # noqa: E402


@pytest.fixture
def metamath_backend():
    ds.config(kernels="metamath")
    yield
    ds.config(kernels="matrix")


# --------------------------------------------------------------------------
# synthetic pulsar
# --------------------------------------------------------------------------

class _Synthetic:
    """Structure + callables + dense reference for the pure-array tests."""

    def __init__(self, seed=1, k=14):
        rng = np.random.default_rng(seed)
        n_b = [480, 320, 100]
        n = sum(n_b)
        backend = np.repeat(np.arange(3), n_b)
        toaerr = np.zeros(n)
        epoch = np.full(n, -1)
        e = i = 0
        for b, nb in enumerate(n_b):
            base = [1e-6, 3e-7, 2e-6][b]
            if b < 2:
                while i < sum(n_b[:b + 1]):
                    m = min(int(rng.integers(4, 9)), sum(n_b[:b + 1]) - i)
                    toaerr[i:i + m] = base * np.exp(rng.uniform(-0.7, 0.7, m))
                    epoch[i:i + m] = e
                    e += 1
                    i += m
            else:
                toaerr[i:i + nb] = base * np.exp(rng.uniform(-0.7, 0.7, nb))
                i += nb
        self.n, self.k, self.n_epoch = n, k, e
        self.backend, self.toaerr, self.epoch = backend, toaerr, epoch
        self.names_e = [f"b{b}_efac" for b in range(3)]
        self.names_q = [f"b{b}_log10_t2equad" for b in range(3)]
        self.names_l = [f"b{b}_log10_ecorr" for b in range(2)]
        self.names = self.names_e + self.names_q + self.names_l
        bk, te2 = jnp.asarray(backend), jnp.asarray(toaerr ** 2)
        eb = np.array([backend[np.flatnonzero(epoch == j)[0]] for j in range(e)])
        ebj = jnp.asarray(eb)
        names_e, names_q, names_l = self.names_e, self.names_q, self.names_l

        def diag(params):
            E = jnp.asarray([params[kk] for kk in names_e])[bk]
            Q = 10.0 ** jnp.asarray([params[kk] for kk in names_q])[bk]
            return E ** 2 * (te2 + Q ** 2)
        diag.params = names_e + names_q

        def ecorr(params):
            return 10.0 ** (2 * jnp.asarray([params[kk] for kk in names_l])[ebj])
        ecorr.params = names_l

        self.struct = cg.MeasurementStructure(diag, epoch, (epoch >= 0).astype(float), e, ecorr)
        self.theta0 = {**{kk: 1.1 for kk in names_e}, **{kk: -7.0 for kk in names_q},
                       **{kk: -6.5 for kk in names_l}}
        self.W = rng.standard_normal((n, k))
        self.W[:, 0] = 1.0
        self.r0 = rng.standard_normal(n) * 1e-6
        self.rng = rng
        U = np.zeros((n, e))
        U[np.flatnonzero(epoch >= 0), epoch[epoch >= 0]] = 1.0
        self.U = U

    def dense_N(self, theta):
        d = np.asarray(self.struct.diag(theta))
        lam = np.asarray(self.struct.ecorr(theta))
        return np.diag(d) + (self.U * lam[None, :]) @ self.U.T

    def exact(self, theta, W=None, r0=None):
        W = self.W if W is None else W
        r0 = self.r0 if r0 is None else r0
        N = self.dense_N(theta)
        NmW = np.linalg.solve(N, W)
        return W.T @ NmW, NmW.T @ r0

    def solve(self, rhs, theta):
        N = self.dense_N(theta)
        return np.linalg.solve(N, np.asarray(rhs)), float(np.linalg.slogdet(N)[1])

    def draws(self, n_draw, *, efac=0.15, dex=0.8, seed=7):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n_draw):
            th = dict(self.theta0)
            for kk in self.names_e:
                th[kk] = self.theta0[kk] * float(np.exp(rng.normal(0, efac)))
            for kk in self.names_q + self.names_l:
                th[kk] = self.theta0[kk] + float(rng.normal(0, dex))
            out.append(th)
        return out

    def layout(self, **kw):
        kw.setdefault("sigma_bin_dex", 0.2)
        kw.setdefault("dense_threshold", self.k)
        return cg.build_layout(self.struct, self.theta0, self.toaerr, **kw)

    def gram(self, **kw):
        return cg.ClassGram(self.W, self.r0, self.struct, self.theta0, self.layout(**kw))


@pytest.fixture(scope="module")
def syn():
    return _Synthetic()


def _cond(G, H):
    L = np.linalg.cholesky(G)
    Hh = np.linalg.solve(L, np.linalg.solve(L, H).T)
    ev = np.linalg.eigvalsh(0.5 * (Hh + Hh.T))
    return float(ev.max() / ev.min())


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


# --------------------------------------------------------------------------
# 1. kernel structure
# --------------------------------------------------------------------------

def _duck(n=60, seed=0, n_backend=3):
    rng = np.random.default_rng(seed)

    class Duck:
        pass
    d = Duck()
    d.name = "J0000+0000"
    # clustered epochs (3 TOAs within an hour) so makegp_ecorr has columns
    d.toas = np.sort(np.repeat(np.linspace(53000.0, 53100.0, n // 3 + 1)[: (n + 2) // 3], 3)[:n]
                     + rng.uniform(0.0, 0.03, n))
    d.toaerrs = rng.uniform(5e-7, 2e-6, n)
    d.residuals = rng.standard_normal(n) * 1e-6
    d.freqs = np.full(n, 1400.0)
    d.backend_flags = np.array([f"be{(i * n_backend) // n}" for i in range(n)], dtype="U8")
    d.flags = {"group": d.backend_flags}
    d.pos = np.array([1.0, 0.0, 0.0])
    return d


def _params(psr, efac=1.2, lgq=-7.0, lge=-6.5):
    out = {}
    for b in sorted(set(psr.backend_flags)):
        out[f"{psr.name}_{b}_efac"] = efac
        out[f"{psr.name}_{b}_log10_t2equad"] = lgq
        out[f"{psr.name}_{b}_log10_ecorr"] = lge
    return out


def test_measurement_structure_kernel_shapes(metamath_backend):
    psr = _duck()
    p0 = _params(psr)
    sm = ds.makenoise_measurement(psr, {}, ecorr=True, enterprise=True)
    assert isinstance(sm, metamath.NoiseMatrixSM)
    s_sm = cg.measurement_structure(sm, p0)
    assert s_sm.n_epoch == np.asarray(sm.F).shape[1]
    assert set(s_sm.params) == set(p0)
    d1 = ds.makenoise_measurement(psr, {})
    s_d = cg.measurement_structure(d1, p0)
    assert s_d.n_epoch == 0 and s_d.ecorr is None
    assert s_d.epoch.shape == (len(psr.toas),) and (s_d.epoch == -1).all()
    # a Woodbury stack is not white noise
    egp = ds.makegp_ecorr(psr, p0, enterprise=True)
    wk = metamath.WoodburyKernel(d1, egp.F, egp.Phi)
    with pytest.raises(TypeError, match="NoiseMatrixSM"):
        cg.measurement_structure(wk, p0)
    # non-disjoint exposure
    F = np.asarray(sm.F).copy()
    F[0, :] = 1.0
    with pytest.raises(ValueError, match="column-disjoint"):
        cg.measurement_structure(_kernels.NoiseMatrixSM(sm.N, F, sm.P), p0)


def test_white_noise_kernel_duality_on_real_pulsar(psr, metamath_backend):
    """ECORR as a fixed GP and ECORR as an SM kernel canonicalize to the same
    NoiseMatrixSM; the SM kernel with free ECORR TRACKS a bump exactly."""
    nd = dict(psr.noisedict)
    nd_wn = {k: v for k, v in nd.items() if k.endswith(("_efac", "_log10_t2equad"))}
    nd_ec = {k: v for k, v in nd.items() if k.endswith("_log10_ecorr")}
    like_gp = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, nd),
                                   ds.makegp_ecorr(psr, nd)])
    like_sm = ds.PulsarLikelihood([psr.residuals,
                                   ds.makenoise_measurement(psr, nd, ecorr=True),
                                   ds.makegp_timing(psr, svd=True)])
    k_gp, k_sm = cg.white_noise_kernel(like_gp), cg.white_noise_kernel(like_sm)
    assert isinstance(k_gp, metamath.NoiseMatrixSM) and isinstance(k_sm, metamath.NoiseMatrixSM)
    y = np.asarray(psr.residuals)
    a, la = metamatrix.func(k_gp.make_solve)(y, params={})
    b, lb = metamatrix.func(k_sm.make_solve)(y, params={})
    assert _rel(a, b) < 1e-12 and abs(float(la) - float(lb)) < 1e-6

    rng = np.random.default_rng(0)
    W = rng.standard_normal((len(y), 6))
    r0 = y
    # free ECORR in the SM kernel (efac/equad pinned): the tracker follows a bump
    like_free = ds.PulsarLikelihood([psr.residuals,
                                     ds.makenoise_measurement(psr, nd_wn, ecorr=True)])
    k_free = cg.white_noise_kernel(like_free)
    st_free = cg.measurement_structure(k_free, nd_ec)
    assert set(st_free.params) == set(nd_ec)
    lay = cg.build_layout(st_free, nd_ec, psr.toaerrs)
    tracker = cg.ClassGram(W, r0, st_free, nd_ec, lay)
    bumped = {k: v + 0.5 for k, v in nd_ec.items()}
    G_tr, b_tr = tracker.gram(bumped)
    # the same model with the bumped ECORR baked as a FIXED GP
    like_bumped = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, nd),
                                       ds.makegp_ecorr(psr, {**nd, **bumped})])
    st_b = cg.measurement_structure(cg.white_noise_kernel(like_bumped), {})
    G_ref, b_ref = cg.ClassGram(W, r0, st_b, {}, cg.build_layout(st_b, {}, psr.toaerrs)).gram({})
    assert _rel(G_tr, G_ref) < 1e-11 and _rel(b_tr, b_ref) < 1e-11

    # a VariableGP ECORR is a sampled block, not white noise: excluded, not folded
    like_var = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, nd),
                                    ds.makegp_ecorr(psr)])
    assert isinstance(cg.white_noise_kernel(like_var), metamath.NoiseMatrix)
    # SM kernel plus a fixed ECORR GP is a double ECORR
    with pytest.raises(ValueError, match="both"):
        cg.white_noise_kernel(ds.PulsarLikelihood([
            psr.residuals, ds.makenoise_measurement(psr, nd, ecorr=True), ds.makegp_ecorr(psr, nd)]))


def test_white_noise_kernel_empty_ecorr_degrades_to_diagonal(metamath_backend):
    psr = _duck(n=20)
    psr.toas = np.arange(20) * 10.0 + 53000.0            # no repeated epochs
    p0 = _params(psr)
    egp = ds.makegp_ecorr(psr, p0, enterprise=True)
    assert np.asarray(egp.F).shape[1] == 0
    like = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr, p0), egp])
    assert isinstance(cg.white_noise_kernel(like), metamath.NoiseMatrix)


# --------------------------------------------------------------------------
# 2. response partition
# --------------------------------------------------------------------------

def test_response_partition_recovers_backends(syn):
    group, ng = cg.response_partition(syn.struct.diag, syn.theta0, syn.n)
    assert ng == 3
    # same partition as the backend labels (up to relabelling)
    for g in range(ng):
        assert len(set(syn.backend[group == g])) == 1
    n_out = 8
    names = [f"p{i}" for i in range(n_out)]

    def per_toa(params):
        return jnp.asarray([params[nm] for nm in names]) ** 2
    per_toa.params = names
    with pytest.raises(ValueError, match="max_groups"):
        cg.response_partition(per_toa, {nm: 1.0 for nm in names}, n_out, max_groups=4)


def _staggered_selection(psr):
    """Emulate MetaPulsar's staggered ('group','g','f') fallback on a duck:
    per-TOA label = first available of the three flags."""
    n = len(psr.toas)
    out = np.full(n, "", dtype="U16")
    for key in ("group", "g", "f"):
        vals = psr.flags.get(key)
        if vals is None:
            continue
        take = (out == "") & (np.asarray(vals) != "")
        out[take] = np.asarray(vals)[take]
    return out


def test_response_partition_matches_staggered_selection(metamath_backend):
    psr = _duck(n=80, n_backend=2)
    n = len(psr.toas)
    group = np.array(["G_A"] * 30 + [""] * 50, dtype="U8")
    g = np.array([""] * 30 + ["g_B"] * 20 + [""] * 30, dtype="U8")
    f = np.array([""] * 50 + ["f_C"] * 15 + ["f_D"] * 15, dtype="U8")
    psr.flags = {"group": group, "g": g, "f": f}
    labels = _staggered_selection(psr)
    assert sorted(set(labels)) == ["G_A", "f_C", "f_D", "g_B"]
    p0 = {}
    for lab in sorted(set(labels)):
        p0[f"{psr.name}_{lab}_efac"] = 1.1
        p0[f"{psr.name}_{lab}_log10_t2equad"] = -6.8
        p0[f"{psr.name}_{lab}_log10_tnequad"] = -6.8
    for tnequad in (False, True):
        kern = ds.makenoise_measurement(psr, {}, selection=_staggered_selection, tnequad=tnequad)
        st = cg.measurement_structure(kern, p0)
        grp, ng = cg.response_partition(st.diag, p0, n)
        assert ng == 4
        for gg in range(ng):
            assert len(set(labels[grp == gg])) == 1


# --------------------------------------------------------------------------
# 3. layout and empty structures
# --------------------------------------------------------------------------

def test_build_layout_partitions_every_toa(syn):
    lay_all_dense = syn.layout(dense_threshold=10 ** 6)
    assert lay_all_dense.n_class == 0 and lay_all_dense.n_dense == syn.n
    G, b = cg.ClassGram(syn.W, syn.r0, syn.struct, syn.theta0, lay_all_dense).gram(syn.theta0)
    G_ref, b_ref = syn.exact(syn.theta0)
    assert _rel(G, G_ref) < 1e-12 and _rel(b, b_ref) < 1e-12
    lay_one = syn.layout(sigma_bin_dex=100.0, dense_threshold=2)
    assert lay_one.n_class == 3 and lay_one.n_dense == 0
    lay = syn.layout()
    dense_mask = np.zeros(syn.n, bool)
    dense_mask[lay.dense] = True
    assert np.array_equal(dense_mask, lay.toa_class < 0)
    with pytest.raises(ValueError, match="positive"):
        cg.build_layout(syn.struct, syn.theta0, np.where(np.arange(syn.n) == 3, 0.0, syn.toaerr))
    with pytest.raises(ValueError, match="dense_threshold"):
        cg.build_layout(syn.struct, syn.theta0, syn.toaerr, dense_threshold=1)


def test_all_dense_and_no_ecorr_evaluate_in_numpy_and_jax(syn):
    st = cg.MeasurementStructure(syn.struct.diag, np.full(syn.n, -1), np.zeros(syn.n), 0, None)
    th = {k: v for k, v in syn.theta0.items() if k not in syn.names_l}
    lay = cg.build_layout(st, th, syn.toaerr)
    g = cg.ClassGram(syn.W, syn.r0, st, th, lay)
    G_np, b_np = g.gram(th, np)
    G_j, b_j = g.gram({k: jnp.asarray(v) for k, v in th.items()}, jnp)
    d = np.asarray(st.diag(th))
    G_ref = syn.W.T @ (syn.W / d[:, None])
    assert _rel(G_np, G_ref) < 1e-12 and _rel(G_j, G_ref) < 1e-12
    assert _rel(b_np, syn.W.T @ (syn.r0 / d)) < 1e-12 and _rel(b_j, b_np) < 1e-13


# --------------------------------------------------------------------------
# 4-7. exactness, EFAC, limits, PSD
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dex,thr", [(0.1, 2), (0.3, 16), (1.0, 14)])
def test_exact_at_bake_point(syn, dex, thr):
    g = syn.gram(sigma_bin_dex=dex, dense_threshold=thr)
    report = cg.validate_class_gram(g, syn.W, syn.r0, syn.solve, syn.theta0, rtol=1e-12)
    assert report["n_classes"] + 0 >= 0 and report["n_epoch"] == syn.n_epoch
    G, b = g.gram(syn.theta0)
    G_ref, b_ref = syn.exact(syn.theta0)
    assert _rel(G, G_ref) < 1e-12 and _rel(b, b_ref) < 1e-12
    assert np.array_equal(G, G.T)


@pytest.mark.parametrize("dex", [0.5, 0.2])
def test_efac_moves_are_exact(syn, dex):
    g = syn.gram(sigma_bin_dex=dex)
    th = dict(syn.theta0)
    th["b0_efac"], th["b1_efac"], th["b2_efac"] = 0.8, 1.5, 2.2
    G, b = g.gram(th)
    G_ref, b_ref = syn.exact(th)
    assert _rel(G, G_ref) < 1e-12 and _rel(b, b_ref) < 1e-12


def test_exact_limits(syn):
    g_dense = syn.gram(dense_threshold=10 ** 6)
    gap = np.min(np.diff(np.unique(np.log10(syn.toaerr ** 2))))
    g_fine = syn.gram(sigma_bin_dex=0.5 * gap, dense_threshold=2)
    for th in syn.draws(5):
        G_ref, b_ref = syn.exact(th)
        for g in (g_dense, g_fine):
            G, b = g.gram(th)
            assert _rel(G, G_ref) < 1e-10 and _rel(b, b_ref) < 1e-10


def test_psd_over_prior_box(syn):
    g = syn.gram()
    rng = np.random.default_rng(11)
    for _ in range(200):
        th = {}
        for kk in syn.names_e:
            th[kk] = float(rng.uniform(0.3, 10.0))
        for kk in syn.names_q + syn.names_l:
            th[kk] = float(rng.uniform(-9.0, -5.0))
        ev = np.linalg.eigvalsh(g.gram(th)[0])
        assert ev.min() >= -1e-12 * ev.max()


# --------------------------------------------------------------------------
# 8-9. geometry over a box, consistent centering
# --------------------------------------------------------------------------

def test_refinement_in_bin_width(syn):
    draws = syn.draws(40)
    stats = {}
    for dex in (0.5, 0.3, 0.2, 0.1):
        g = syn.gram(sigma_bin_dex=dex)
        conds = []
        for th in draws:
            G_ref, _ = syn.exact(th)
            conds.append(_cond(g.gram(th)[0] + np.eye(syn.k), G_ref + np.eye(syn.k)))
        stats[dex] = (float(np.median(conds)), float(np.max(conds)))
    for a, b in zip((0.5, 0.3, 0.2), (0.3, 0.2, 0.1)):
        assert stats[b][0] <= stats[a][0] + 1e-9 and stats[b][1] <= stats[a][1] + 1e-9
    assert stats[0.2][1] < 1.1


def test_consistent_centering_beats_exact_b(syn):
    # plant a large signal: |mu| >> 100 whitened units
    rng = np.random.default_rng(5)
    c = rng.standard_normal(syn.k) * 1e-4
    r_big = syn.W @ c + syn.r0
    g = cg.ClassGram(syn.W, r_big, syn.struct, syn.theta0, syn.layout())
    worst_consistent, worst_exact_b = 0.0, 0.0
    for th in syn.draws(20):
        G_ref, b_ref = syn.exact(th, r0=r_big)
        H = G_ref + np.eye(syn.k)
        G, b = g.gram(th)
        A = G + np.eye(syn.k)
        L = np.linalg.cholesky(A)
        mu_ref = np.linalg.solve(H, b_ref)
        assert np.abs(L.T @ mu_ref).max() > 100.0
        off_consistent = np.abs(L.T @ (mu_ref - np.linalg.solve(A, b))).max()
        off_exact_b = np.abs(L.T @ (mu_ref - np.linalg.solve(A, b_ref))).max()
        worst_consistent = max(worst_consistent, off_consistent)
        worst_exact_b = max(worst_exact_b, off_exact_b)
    assert worst_consistent < 1.2
    assert worst_exact_b > 5.0


# --------------------------------------------------------------------------
# 10. jax parity, gradient, single compilation, symmetry
# --------------------------------------------------------------------------

def test_jax_matches_numpy_and_grad_matches_fd(syn):
    g = syn.gram()
    names = syn.names
    v0 = jnp.asarray([syn.theta0[nm] for nm in names])

    def as_dict(v):
        return {nm: v[i] for i, nm in enumerate(names)}

    n_trace = 0

    def fn(v):
        nonlocal n_trace
        n_trace += 1
        return g.gram(as_dict(v), jnp)
    f = jax.jit(fn)
    G_j, b_j = f(v0)
    f(v0 + 0.01)
    assert n_trace == 1
    G_n, b_n = g.gram(syn.theta0, np)
    assert _rel(G_j, G_n) < 1e-12 and _rel(b_j, b_n) < 1e-12
    assert np.array_equal(np.asarray(G_j), np.asarray(G_j).T)

    def loss(v):
        G, b = g.gram(as_dict(v), jnp)
        return jnp.sum(G ** 2) + 1e12 * jnp.sum(b ** 2)
    grad = np.asarray(jax.jit(jax.grad(loss))(v0))
    fd = np.zeros_like(grad)
    for i in range(len(names)):
        h = 1e-5
        fd[i] = (float(loss(v0.at[i].add(h))) - float(loss(v0.at[i].add(-h)))) / (2 * h)
    assert np.abs(grad - fd).max() / np.abs(fd).max() < 1e-6
