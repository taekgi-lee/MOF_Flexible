"""rev2, agent `ls`: label spreading recomputed with the corrected labels (label set v2: gas seeds 1,833), the revised
material groups and the 191 standardized descriptors; pooled gate grid (25 fold seeds, all used for the selection);
cross-check against rf_final (No P <= 0.10, Yes P > 0.85). The No gate and the Yes gate may come from different (k, alpha),
i.e. from two propagations. Ten further fold seeds are evaluated only for the selected settings (independent of the selection).

Arithmetic: sklearn LabelSpreading (kernel='knn', max_iter=1000, tol=1e-3), re-implemented on the sklearn graph matrix
(checked here against sklearn itself). Support = number of distinct material groups among the same-label seeds in the
k nearest neighbours of a structure (the structure itself excluded).

Inputs : data/descriptors.csv, data/labels/labels_final.csv, data/master_groups.csv, results/final_rf/,
         results/label_spreading/ls_rows_7188.csv (row order of the 7,188 structures and the labels as extracted).
Outputs: results/label_spreading/*.csv and compute.json.
Run    : cd 06_label_extension && python ls_compute_rev2.py
"""
import itertools, json, os, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..') + '/'
OUT = REPO + 'results/label_spreading/'
os.makedirs(OUT, exist_ok=True)
KS, AL, SUP = [15, 20, 30, 40, 50, 75], [0.1, 0.2], [1, 2, 3, 4, 5]
SEEDS = list(range(20260724, 20260749))          # 25 fold-assignment seeds; the gates are selected on the passes pooled over all of them
SEEDS5 = SEEDS[:5]                               # the five seeds of the audited numbers (labels before correction; report only)
assert len(SEEDS) == 25 and SEEDS[0] == 20260724 and SEEDS[-1] == 20260748
T_NO, T_YES = 0.995, 0.885          # purity thresholds (fixed)
RF_NO, RF_YES = 0.10, 0.85          # rf_final confident intervals: No P <= 0.10, Yes P > 0.85
SEL_SUP = 3                         # support (fixed)
SEEDS_CHECK = list(range(20260749, 20260759))    # 10 new fold seeds, evaluated only for the selected settings (not used for the selection)
assert not set(SEEDS) & set(SEEDS_CHECK)
J = {}


# ----------------------------------------------------------------------------------------------- data
def short(n):
    return n.replace('_ASR_pacman', '').replace('_FSR_pacman', ' (FSR)').replace('_ION_pacman', ' (ION)')


order = pd.read_csv(OUT + 'ls_rows_7188.csv')          # row order of the 7,188 structures
cif = order.cif_name.values
d = pd.read_csv(REPO + 'data/descriptors.csv').set_index('cif_name')
rac = [c for c in d.columns if c.startswith('rac_')]
cols = ['LCD', 'PLD', 'LFPD', 'density', 'ASA', 'GSA', 'VSA', 'PV', 'VF', 'GPV'] + rac + \
       ['K_vrh', 'G_vrh', 'anisotropy', 'lambda_min', 'born_stable']
assert len(cols) == 191 and len(cif) == 7188 and len(set(cif)) == 7188
Xraw = d.loc[cif, cols].values.astype(float)
assert np.isfinite(Xraw).all()
X = StandardScaler().fit_transform(Xraw)

lab = pd.read_csv(REPO + 'data/labels/labels_final.csv', keep_default_na=False, na_values=['']).set_index('cif')
g = lab.loc[cif, 'gas'].str.lower().values
y = np.full(len(cif), -1)
y[g == 'yes'] = 1
y[g == 'no'] = 0
y_old = np.full(len(cif), -1)                                                # labels as extracted (before the five gas corrections)
y_old[order.label_in.values == 'Yes'] = 1
y_old[order.label_in.values == 'No'] = 0
chg = np.where(y != y_old)[0]
mg = pd.read_csv(REPO + 'data/master_groups.csv', keep_default_na=False,
                 na_values=['']).set_index('cif_name').loc[cif]
grp = mg.material_id.values
assert not pd.isna(grp).any()
codes = pd.factorize(grp)[0]
U = y == -1
S = np.where(~U)[0]
J['setup'] = dict(n_structures=int(len(cif)), n_features=len(cols), seeds=int((~U).sum()), seeds_yes=int((y == 1).sum()),
                  seeds_no=int((y == 0).sum()), yes_fraction_of_seeds=float((y == 1).sum() / (~U).sum()),
                  targets=int(U.sum()), seed_materials=int(len(set(grp[~U]))),
                  label_changes=[(short(cif[i]), int(y_old[i]), int(y[i])) for i in chg],
                  gas_label_counts=pd.Series(g).value_counts().to_dict())
print('setup', J['setup'])
assert J['setup']['seeds'] == 1833 and J['setup']['targets'] == 5355          # label set v2
assert J['setup']['seeds_yes'] == 650 and J['setup']['seeds_no'] == 1183 and len(chg) == 5 and all(y_old[i] == 1 and y[i] == -1 for i in chg)

rf = pd.read_csv(REPO + 'results/final_rf/probabilities_gas_final.csv', keep_default_na=False, na_values=['']).set_index('cif')
assert set(rf.index) == set(cif[U]), 'rf_final targets differ from the label-spreading targets'
p = np.full(len(cif), np.nan)
p[U] = rf.loc[cif[U], 'P_mean'].values
cat = np.full(len(cif), '', object)
cat[U] = rf.loc[cif[U], 'category'].values
assert (rf.loc[cif[U], 'material_id'].values == grp[U]).all()
# own derivation of the categories (seeds of the CIF's material group)
sl = pd.DataFrame({'g': grp[~U], 'y': y[~U]}).groupby('g').y.mean()
mycat = np.array(['' if not u else ('noseed' if gg not in sl.index else ('yes' if sl[gg] == 1 else ('no' if sl[gg] == 0 else 'conflict')))
                  for u, gg in zip(U, grp)], object)
assert (mycat == cat).all()
new = U & (cat == 'noseed')
rfNo, rfYes = U & (p <= RF_NO), U & (p > RF_YES)
rfUns = U & ~rfNo & ~rfYes


def nm(mask):
    return dict(cifs=int(mask.sum()), materials=int(len(set(grp[mask]))))


J['rf'] = dict(thresholds=dict(no=RF_NO, yes=RF_YES), categories=pd.Series(cat[U]).value_counts().to_dict(),
               noseed=nm(new), all=nm(U), noseed_no=nm(new & rfNo), noseed_yes=nm(new & rfYes), noseed_unsure=nm(new & rfUns),
               all_no=nm(rfNo), all_yes=nm(rfYes), all_unsure=nm(rfUns))
print('rf', J['rf'])
# must equal headline_extension.json
H = json.load(open(REPO + 'results/final_rf/headline_extension.json'))
assert J['rf']['noseed'] == H['noseed'] and J['rf']['noseed_no'] == H['noseed_no'] and J['rf']['noseed_yes'] == H['noseed_yes']
assert J['rf']['all_no'] == H['all_no'] and J['rf']['all_yes'] == H['all_yes'] and J['rf']['all'] == H['targets']


# ----------------------------------------------------------------------------------------------- label spreading
class Graph:
    def __init__(self, X, kmax):
        self.X = X
        self.idx = NearestNeighbors(n_neighbors=kmax).fit(X).kneighbors(None, n_neighbors=kmax, return_distance=False)
        self._gm = {}

    def gm(self, k):
        if k not in self._gm:
            m = LabelSpreading(kernel='knn', n_neighbors=k)
            m.X_ = self.X
            self._gm[k] = m._build_graph().tocsr()
        return self._gm[k]

    def support(self, y_vis, k, label):
        nb = self.idx[:, :k]
        gg = np.where(y_vis[nb] == label, codes[nb], -1)
        gg.sort(axis=1)
        return ((np.diff(gg, axis=1) != 0) & (gg[:, 1:] >= 0)).sum(axis=1) + (gg[:, 0] >= 0)

    def spread(self, y_vis, k, alpha, max_iter=1000, tol=1e-3):
        Gm = self.gm(k)
        L = np.zeros((len(y_vis), 2))
        L[y_vis == 0, 0] = 1
        L[y_vis == 1, 1] = 1
        ys = L * (1 - alpha)
        prev = np.zeros_like(L)
        for _ in range(max_iter):
            if np.abs(L - prev).sum() < tol:
                break
            prev = L
            L = alpha * (Gm @ L) + ys
        s = L.sum(1, keepdims=True)
        zero = int((s == 0).sum())
        s[s == 0] = 1
        L = L / s
        return L[:, 1], L[:, 0], zero


G = Graph(X, 80)
# check against sklearn itself (one setting, full seeds)
m = LabelSpreading(kernel='knn', n_neighbors=50, alpha=0.1, max_iter=1000, tol=1e-3, n_jobs=1).fit(X, y)
py_, pn_, _ = G.spread(y, 50, 0.1)
J['check_sklearn_max_abs_diff_k50_a01'] = float(np.abs(m.label_distributions_[:, 1] - py_).max())
print('max |sklearn - own| P(yes), k=50, alpha=0.1:', J['check_sklearn_max_abs_diff_k50_a01'])
assert J['check_sklearn_max_abs_diff_k50_a01'] < 1e-12
# support by brute force for a few structures
rng = np.random.default_rng(0)
sup50 = G.support(y, 50, 0)
for i in rng.choice(len(cif), 200, replace=False):
    nb = G.idx[i, :50]
    assert len(set(grp[nb[y[nb] == 0]])) == sup50[i]


# ----------------------------------------------------------------------------------------------- validation grid
def grid(yy, tag, keep_passes=True, seeds=SEEDS, settings=None):
    """Material-masked five-fold validation, pooled over the fold seeds. Returns per-seed rows and the passes."""
    Sb = np.where(yy >= 0)[0]
    rows, passes, yes995 = [], [], []
    for seed in seeds:
        fold = np.zeros(len(Sb), int)
        for i, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Sb, yy[Sb], groups=grp[Sb])):
            fold[te] = i
        # no material group in two folds
        assert pd.DataFrame({'g': grp[Sb], 'f': fold}).groupby('g').f.nunique().max() == 1
        for k, a in (settings or itertools.product(KS, AL)):
            o = np.zeros((len(Sb), 4))
            for f in range(5):
                hid = Sb[fold == f]
                yv = yy.copy()
                yv[hid] = -1
                py, pn, _ = G.spread(yv, k, a)
                o[fold == f, 0] = py[hid]
                o[fold == f, 1] = pn[hid]
                o[fold == f, 2] = G.support(yv, k, 1)[hid]
                o[fold == f, 3] = G.support(yv, k, 0)[hid]
            for s in SUP:
                for gate, mk, true in (('no', (o[:, 1] > T_NO) & (o[:, 3] >= s), 0), ('yes', (o[:, 0] > T_YES) & (o[:, 2] >= s), 1)):
                    rows.append(dict(labels=tag, seed=seed, k=k, alpha=a, support=s, gate=gate, n_pass=int(mk.sum()),
                                     n_correct=int((yy[Sb][mk] == true).sum())))
                    if keep_passes:
                        for j in np.where(mk)[0]:
                            passes.append(dict(labels=tag, gate=gate, k=k, alpha=a, support=s, seed=seed, cif=cif[Sb[j]], mat=grp[Sb[j]],
                                               correct=int(yy[Sb][j] == true)))
            mk = (o[:, 0] > T_NO) & (o[:, 2] >= 3)                       # yes gate at the no threshold (0.995), support >= 3
            yes995.append(dict(labels=tag, seed=seed, k=k, alpha=a, n_pass=int(mk.sum()), n_correct=int((yy[Sb][mk] == 1).sum())))
        print(tag, seed, 'done', flush=True)
    R_ = pd.DataFrame(rows)
    R_['acc'] = R_.n_correct / R_.n_pass
    return R_, pd.DataFrame(passes), pd.DataFrame(yes995)


def pooled(R_):
    P = R_.groupby(['gate', 'support', 'k', 'alpha'], as_index=False).agg(
        n=('n_pass', 'sum'), c=('n_correct', 'sum'), acc_min=('acc', 'min'), acc_max=('acc', 'max'),
        n_seeds_ge95=('acc', lambda v: int((v >= 0.95).sum())), n_min=('n_pass', 'min'), n_max=('n_pass', 'max'))
    P['acc'] = P.c / P.n
    return P


def wilson(c, n, z=1.959963984540054):
    """95% Wilson score interval of c/n. It treats the n passes as independent; pooled passes are repeated counts of the same seeds."""
    ph = c / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(mid - half), float(mid + half)]


def pick(dd):
    dd = dd[dd.acc >= 0.95]
    if not len(dd):
        return None
    r = dd.sort_values(['n', 'acc'], ascending=False).iloc[0]
    return dict(k=int(r.k), alpha=float(r.alpha), support=int(r.support), n=int(r.n), correct=int(r.c), acc=float(r.acc))


Rn, Qn, Y995 = grid(y, 'corrected')
Pn = pooled(Rn)
Rn.to_csv(OUT + 'grid_per_seed.csv', index=False)
Pn.to_csv(OUT + 'grid_pooled.csv', index=False)
Qn.to_csv(OUT + 'grid_passes.csv', index=False)
Y995.to_csv(OUT + 'grid_yes_at_0995.csv', index=False)
Ro, _, _ = grid(y_old, 'before_correction', keep_passes=False, seeds=SEEDS5)       # reproduces the audited numbers (179/187, 243/257)
Po = pooled(Ro)
Po.to_csv(OUT + 'grid_pooled_before_correction.csv', index=False)

sel = {}
for tag, P in (('corrected', Pn), ('before_correction', Po)):
    sel[tag] = {}
    for gate in ('no', 'yes'):
        dd = P[P.gate == gate]
        sel[tag][gate] = dict(support3=pick(dd[dd.support == 3]), support_any=pick(dd), support_ge3=pick(dd[dd.support >= 3]),
                              n_settings_ge95_support3=int((dd[dd.support == 3].acc >= 0.95).sum()),
                              settings_ge95_support3=[(int(r.k), float(r.alpha), int(r.n), int(r.c), round(float(r.acc), 4))
                                                      for r in dd[(dd.support == 3) & (dd.acc >= 0.95)].itertuples()],
                              most_accurate_support3=(lambda r: dict(k=int(r.k), alpha=float(r.alpha), n=int(r.n), correct=int(r.c), acc=float(r.acc)))(
                                  dd[dd.support == 3].sort_values('acc', ascending=False).iloc[0]),
                              settings_ge95_any_support=[(int(r.k), float(r.alpha), int(r.support), int(r.n), int(r.c), round(float(r.acc), 4))
                                                         for r in dd[dd.acc >= 0.95].itertuples()])
J['selection'] = sel
print(json.dumps(sel, indent=1))
# rule: support 3 and thresholds fixed; among (k, alpha) with pooled accuracy >= 0.95, the largest pooled number of passes.
# If no setting meets the criterion, the most accurate setting at support 3 is used for description only.
_n, _y = sel['corrected']['no'], sel['corrected']['yes']
assert _n['support3'] is not None
SEL_NO = (_n['support3']['k'], _n['support3']['alpha'])
SEL_YES = (_y['support3']['k'], _y['support3']['alpha']) if _y['support3'] else (_y['most_accurate_support3']['k'], _y['most_accurate_support3']['alpha'])
J['selected'] = dict(no=dict(k=SEL_NO[0], alpha=SEL_NO[1], threshold=T_NO, support=SEL_SUP, meets_criterion=True),
                     yes=dict(k=SEL_YES[0], alpha=SEL_YES[1], threshold=T_YES, support=SEL_SUP, meets_criterion=_y['support3'] is not None),
                     no_before_correction=sel['before_correction']['no']['support3'])
PREV_NO = (50, 0.1)                 # the No setting with the most passes that the rule selected before the label correction
print('SELECTED', J['selected'])

# Table S13a (pooled, support 3) and Table S13b (selected gates, support 1-5, per fold seed)
J['tableS13a'] = [dict(k=int(k), alpha=float(a),
                       **{f'{gt}_{fld}': (int(v) if fld != 'acc' else float(v))
                          for gt in ('no', 'yes') for fld, v in zip(('n', 'c', 'acc'), Pn[(Pn.gate == gt) & (Pn.support == 3) & (Pn.k == k) & (Pn.alpha == a)][['n', 'c', 'acc']].values[0])},
                       **{f'{gt}_{fld}': (lambda r: float(r[fld]) if fld != 'n_seeds_ge95' else int(r[fld]))(Pn[(Pn.gate == gt) & (Pn.support == 3) & (Pn.k == k) & (Pn.alpha == a)].iloc[0])
                          for gt in ('no', 'yes') for fld in ('acc_min', 'acc_max', 'n_seeds_ge95')})
                  for k, a in itertools.product(KS, AL)]
for r_ in J['tableS13a']:
    for gt in ('no', 'yes'):
        r_[f'{gt}_wilson'] = wilson(r_[f'{gt}_c'], r_[f'{gt}_n'])
J['fold_seeds'] = dict(selection=dict(first=SEEDS[0], last=SEEDS[-1], n=len(SEEDS)), check=dict(first=SEEDS_CHECK[0], last=SEEDS_CHECK[-1], n=len(SEEDS_CHECK)))
J['n_grid_settings'] = len(KS) * len(AL)
tb = []
for gate, (k, a) in (('no', SEL_NO), ('yes', SEL_YES)):
    for s in SUP:
        dd = Rn[(Rn.gate == gate) & (Rn.k == k) & (Rn.alpha == a) & (Rn.support == s)].sort_values('seed')
        pr = Pn[(Pn.gate == gate) & (Pn.k == k) & (Pn.alpha == a) & (Pn.support == s)].iloc[0]
        q = Qn[(Qn.gate == gate) & (Qn.k == k) & (Qn.alpha == a) & (Qn.support == s)]
        u = q.groupby('cif').agg(correct=('correct', 'first'), mat=('mat', 'first'), times=('seed', 'size'))
        wrong = u[u.correct == 0]
        assert len(dd) == len(SEEDS)
        tb.append(dict(gate=gate, k=k, alpha=a, support=s, n=int(pr.n), correct=int(pr.c), acc=float(pr.acc), wilson=wilson(int(pr.c), int(pr.n)),
                       per_seed_n=dd.n_pass.astype(int).tolist(), per_seed_acc=[None if np.isnan(v) else float(v) for v in dd.acc],
                       acc_min=float(dd.acc.min()), acc_max=float(dd.acc.max()), n_seeds_ge95=int((dd.acc >= 0.95).sum()),
                       distinct_seeds=int(len(u)), distinct_materials=int(u.mat.nunique()),
                       distinct_wrong_seeds=int(len(wrong)), distinct_wrong_materials=int(wrong.mat.nunique()),
                       wrong=[f'{short(c)} x{t}' for c, t in zip(wrong.index, wrong.times)]))
J['tableS13b'] = tb
# other Yes settings of interest (alpha 0.2; support 4)
J['yes_alt'] = {f'k{k}_a{a}_s{s}': (lambda r: dict(n=int(r.n), correct=int(r.c), acc=float(r.acc), acc_min=float(r.acc_min), acc_max=float(r.acc_max)))(
    Pn[(Pn.gate == 'yes') & (Pn.k == k) & (Pn.alpha == a) & (Pn.support == s)].iloc[0]) for k, a, s in ((75, 0.1, 4), (75, 0.2, 3), (75, 0.2, 4), (75, 0.1, 5))}
J['yes_at_0995_support3_pooled'] = {f'k{int(k)}_a{a}': dict(n=int(dd.n_pass.sum()), correct=int(dd.n_correct.sum()), per_seed=dd.n_pass.astype(int).tolist())
                                    for (k, a), dd in Y995.groupby(['k', 'alpha'])}
J['before_correction_check'] = dict(
    no_k50_a01_s3=(lambda r: (int(r.c), int(r.n)))(Po[(Po.gate == 'no') & (Po.k == 50) & (Po.alpha == 0.1) & (Po.support == 3)].iloc[0]),
    yes_k75_a01_s3=(lambda r: (int(r.c), int(r.n)))(Po[(Po.gate == 'yes') & (Po.k == 75) & (Po.alpha == 0.1) & (Po.support == 3)].iloc[0]),
    yes_k75_a01_s4=(lambda r: (int(r.c), int(r.n)))(Po[(Po.gate == 'yes') & (Po.k == 75) & (Po.alpha == 0.1) & (Po.support == 4)].iloc[0]))
print('before correction (audited: 179/187, 243/257, 214/224):', J['before_correction_check'])

# every No setting that meets the criterion at support 3 (pooled over the 25 fold seeds)
NO_OK = [(int(k_), float(a_)) for k_, a_, *_ in _n['settings_ge95_support3']]
assert SEL_NO in NO_OK
ALT = sorted(set(NO_OK) | {SEL_NO, SEL_YES, PREV_NO, (75, 0.1), (75, 0.2)})
# independent check: 10 new fold seeds, evaluated only for the selected settings (they played no part in the selection)
Rc, _, _ = grid(y, 'corrected_check_seeds', keep_passes=False, seeds=SEEDS_CHECK, settings=sorted({SEL_NO, SEL_YES}))
Rc.to_csv(OUT + 'grid_check_seeds.csv', index=False)
ck = {}
for (gate_, k, a, s_), dd in Rc.groupby(['gate', 'k', 'alpha', 'support']):
    if s_ in (3, 4) and (k, a) == (SEL_NO if gate_ == 'no' else SEL_YES):
        n_, c_ = int(dd.n_pass.sum()), int(dd.n_correct.sum())
        ck[f'{gate_}_k{k}_a{a}_s{s_}'] = dict(n=n_, correct=c_, acc=c_ / n_, wilson=wilson(c_, n_), acc_min=float(dd.acc.min()), acc_max=float(dd.acc.max()),
                                              n_seeds=int(len(dd)), n_seeds_ge95=int((dd.acc >= 0.95).sum()), n_min=int(dd.n_pass.min()), n_max=int(dd.n_pass.max()))
J['check_seeds'] = dict(seeds=[SEEDS_CHECK[0], SEEDS_CHECK[-1]], results=ck)
print('check seeds', json.dumps(ck, indent=1))

# ----------------------------------------------------------------------------------------------- targets
runs = {}
for k, a in sorted(set(ALT) | {(15, 0.2), (50, 0.2), (75, 0.1), (50, 0.1)}):
    py, pn, zero = G.spread(y, k, a)
    runs[(k, a)] = dict(py=py, pn=pn, sy=G.support(y, k, 1), sn=G.support(y, k, 0), zero=zero)
J['zero_mass_rows'] = {f'k{k}_a{a}': int(r['zero']) for (k, a), r in runs.items()}


def gate(cls, ka, s=SEL_SUP, thr=None):
    r = runs[ka]
    if cls == 'no':
        return U & (r['pn'] > (T_NO if thr is None else thr)) & (r['sn'] >= s)
    return U & (r['py'] > (T_YES if thr is None else thr)) & (r['sy'] >= s)


lsNo, lsYes = gate('no', SEL_NO), gate('yes', SEL_YES)


def cross(base, gn, gy):
    rr = dict(n=int(base.sum()), materials=int(len(set(grp[base]))), rf_no=nm(base & rfNo), rf_yes=nm(base & rfYes), rf_unsure=nm(base & rfUns),
              ls_no=nm(base & gn), ls_yes=nm(base & gy), ls_both_gates=int((base & gn & gy).sum()),
              both_no=nm(base & gn & rfNo), rf_yes_confirmed=int((base & gy & rfYes).sum()),
              opposite=int((base & ((gn & rfYes) | (gy & rfNo))).sum()),
              ls_no_rf_unsure=nm(base & gn & rfUns), ls_yes_rf_unsure=nm(base & gy & rfUns),
              ls_only_confident=int((base & (gn | gy) & rfUns).sum()),
              rf_p_of_ls_no=[float(p[base & gn].min()), float(p[base & gn].max())] if (base & gn).any() else None,
              rf_p_of_ls_yes=[float(p[base & gy].min()), float(p[base & gy].max())] if (base & gy).any() else None,
              ls_no_rf_p_le_020=int((base & gn & (p <= 0.2)).sum()))
    return rr


J['cross'] = {b: cross(mk, lsNo, lsYes) for b, mk in (('noseed', new), ('all', U))}
J['cross_alt'] = {
    'yes_support4': {b: cross(mk, lsNo, gate('yes', SEL_YES, 4)) for b, mk in (('noseed', new), ('all', U))},
    'yes_alpha02': {b: cross(mk, lsNo, gate('yes', (75, 0.2))) for b, mk in (('noseed', new), ('all', U))},
    'yes_k75_a01': {b: cross(mk, lsNo, gate('yes', (75, 0.1))) for b, mk in (('noseed', new), ('all', U))},
    'no_k50_a01': {b: cross(mk, gate('no', PREV_NO), lsYes) for b, mk in (('noseed', new), ('all', U))},
    'no_k50_a02': {b: cross(mk, gate('no', (50, 0.2)), lsYes) for b, mk in (('noseed', new), ('all', U))},
    'no_k75_a01': {b: cross(mk, gate('no', (75, 0.1)), lsYes) for b, mk in (('noseed', new), ('all', U))},
    'no_k75_a02': {b: cross(mk, gate('no', (75, 0.2)), lsYes) for b, mk in (('noseed', new), ('all', U))},
    'no_k15_a02': {b: cross(mk, gate('no', (15, 0.2)), lsYes) for b, mk in (('noseed', new), ('all', U))}}
for ka in NO_OK:
    J['cross_alt'].setdefault(f'no_k{ka[0]}_a{str(ka[1]).replace(".", "")}', {b: cross(mk, gate('no', ka), lsYes) for b, mk in (('noseed', new), ('all', U))})
J['no_settings_meeting_criterion'] = [list(ka) for ka in NO_OK]
J['ls_no_overlap_selected_k50'] = dict(selected=int(lsNo.sum()), k50=int(gate('no', PREV_NO).sum()), both=int((lsNo & gate('no', PREV_NO)).sum()))
# opposite calls with the other Yes setting of interest (k = 75, alpha = 0.1), same No gate
_ya = (75, 0.1)
_oa = U & ((lsNo & rfYes) | (gate('yes', _ya) & rfNo))
J['opposite_detail_yes_k75_a01'] = [dict(cif=short(cif[i]), material=grp[i], category=cat[i], ls_call='No' if lsNo[i] else 'Yes',
                                         ls_P_yes=float(runs[_ya]['py'][i]), support_yes=int(runs[_ya]['sy'][i]),
                                         ls_P_yes_selected_yes_run=float(runs[SEL_YES]['py'][i]), support_yes_selected_yes_run=int(runs[SEL_YES]['sy'][i]),
                                         rf_P=float(p[i])) for i in np.where(_oa)[0]]
print('opposite calls with the yes setting k = 75, alpha = 0.1:', J['opposite_detail_yes_k75_a01'])
opp = U & ((lsNo & rfYes) | (lsYes & rfNo))
_rff = REPO + 'results/final_rf/probabilities_gas.csv'                                    # same model, with the 10 single forests (optional)
RFF = pd.read_csv(_rff).set_index('cif') if os.path.exists(_rff) else None
J['opposite_detail'] = [dict(cif=short(cif[i]), material=grp[i], category=cat[i], ls_call='No' if lsNo[i] else 'Yes',
                             ls_P_yes=float(runs[SEL_YES]['py'][i]), support_yes=int(runs[SEL_YES]['sy'][i]), support_no=int(runs[SEL_NO]['sn'][i]),
                             rf_P=float(p[i]), rf_P_sd=float(rf.loc[cif[i], 'P_sd']),
                             rf_P_forests=None if RFF is None else [float(v) for v in RFF.loc[cif[i], [c for c in RFF.columns if c.startswith('P_2026')]].values],
                             n_yes_seeds_among_neighbours=int((y[G.idx[i, :SEL_YES[0]]] == 1).sum()),
                             n_no_seeds_among_neighbours=int((y[G.idx[i, :SEL_YES[0]]] == 0).sum())) for i in np.where(opp)[0]]
print('opposite calls:', J['opposite_detail'])
J['spearman'] = {f'k{k}_a{a}': dict(noseed=float(spearmanr(p[new], runs[(k, a)]['py'][new])[0]), all=float(spearmanr(p[U], runs[(k, a)]['py'][U])[0]))
                 for (k, a) in runs}
# targets that receive no label mass (P(yes) = P(no) = 0; they pass no gate) and the rank correlation without them
J['zero_mass_targets'], J['spearman_without_zero_mass'] = {}, {}
for (k, a), r_ in runs.items():
    zm = (r_['py'] + r_['pn']) == 0
    assert not (zm & ~U).any() and not (zm & (gate('no', (k, a), 1) | gate('yes', (k, a), 1))).any()
    J['zero_mass_targets'][f'k{k}_a{a}'] = dict(all=int((zm & U).sum()), noseed=int((zm & new).sum()), rf_no=int((zm & rfNo).sum()), rf_yes=int((zm & rfYes).sum()))
    J['spearman_without_zero_mass'][f'k{k}_a{a}'] = dict(noseed=float(spearmanr(p[new & ~zm], r_['py'][new & ~zm])[0]), all=float(spearmanr(p[U & ~zm], r_['py'][U & ~zm])[0]))
print('zero-mass targets', J['zero_mass_targets']); print('spearman without them', J['spearman_without_zero_mass'])
print(json.dumps(J['cross'], indent=1)); print(J['spearman'])

# by category, and the in-sample result (confident calls opposite to the literature label of the CIF's own material)
J['by_category'] = {c: dict(n=int((U & (cat == c)).sum()), ls_no=int((U & (cat == c) & lsNo).sum()), ls_yes=int((U & (cat == c) & lsYes).sum()),
                            rf_no=int((U & (cat == c) & rfNo).sum()), rf_yes=int((U & (cat == c) & rfYes).sum()))
                    for c in ('yes', 'no', 'conflict', 'noseed')}
un = U & np.isin(cat, ['yes', 'no'])
ls_con = (U & (cat == 'yes') & lsNo) | (U & (cat == 'no') & lsYes)
rf_con = (U & (cat == 'yes') & rfNo) | (U & (cat == 'no') & rfYes)
oof = pd.read_csv(REPO + 'results/final_rf/oof_gas.csv')
oof = oof[oof.split == 'Material'].set_index('cif_name').P_mean
det = []
for i in np.where(ls_con)[0]:
    own = np.where((grp == grp[i]) & ~U)[0]
    k = SEL_NO[0] if lsNo[i] else SEL_YES[0]
    nb = G.idx[i, :k]
    det.append(dict(cif=short(cif[i]), material=grp[i], ls_call='No' if lsNo[i] else 'Yes', rf_P=float(p[i]),
                    rf_call='No' if rfNo[i] else ('Yes' if rfYes[i] else 'unsure'),
                    own_seeds=[(short(cif[j]), 'Yes' if y[j] == 1 else 'No', float(oof[cif[j]])) for j in own],
                    own_seed_among_neighbours=[short(cif[j]) for j in own if j in nb],
                    neighbour_seeds_yes=int((y[nb] == 1).sum()), neighbour_seeds_no=int((y[nb] == 0).sum())))
J['in_sample'] = dict(unanimous_cifs=int(un.sum()), unanimous_yes_cifs=int((U & (cat == 'yes')).sum()), unanimous_no_cifs=int((U & (cat == 'no')).sum()),
                      ls_confident_in_unanimous=int((un & (lsNo | lsYes)).sum()), ls_opposite=int(ls_con.sum()),
                      rf_confident_in_unanimous=int((un & (rfNo | rfYes)).sum()), rf_opposite=int(rf_con.sum()),
                      ls_opposite_detail=det)
print(json.dumps(J['in_sample'], indent=1))

# Yes gate: purity only versus support (Observation 2) and concentration of the neighbouring Yes seeds
r = runs[SEL_YES]
pur = U & (r['py'] > T_YES)
one = pur & (r['sy'] == 1)
seedsets = {}
for i in np.where(one)[0]:
    nb = G.idx[i, :SEL_YES[0]]
    key = (','.join(sorted(short(c) for c in cif[nb[y[nb] == 1]])), int((y[nb] == 0).sum()))
    seedsets[key] = seedsets.get(key, 0) + 1
J['yes_purity_only'] = dict(**nm(pur), rf_unsure=int((pur & rfUns).sum()), rf_no=int((pur & rfNo).sum()), rf_yes=int((pur & rfYes).sum()),
                            rf_p_range=[float(p[pur].min()), float(p[pur].max())],
                            by_support={int(s): nm(pur & (r['sy'] == s)) for s in sorted(set(r['sy'][pur]))},
                            support_ge3=nm(pur & (r['sy'] >= 3)), support_ge4=nm(pur & (r['sy'] >= 4)),
                            support1_seed_sets=[dict(yes_seeds=k_[0], n_no_seeds=k_[1], n_cifs=v) for k_, v in seedsets.items()],
                            support1_materials=pd.Series(grp[one]).value_counts().to_dict(),
                            support1_noseed=int((one & new).sum()))
# relation of the single-seed group to the seed UMEDIB
um = np.where(cif == 'UMEDIB_ASR_pacman')[0]
if len(um):
    um = um[0]
    chem = mg.chemistry_id.values
    same_mat = one & (grp == grp[um])
    same_chem = one & (chem == chem[um]) & ~same_mat
    other = one & ~same_mat & ~same_chem
    key = chem[um]                                            # 'Zn||<linker key>'
    metal, linker = key.split('||')[0], key.split('||')[1]
    same_ml = one & ~same_mat & np.array([(c.split('||')[0] == metal) and (linker in c.split('||')[1:]) for c in chem])
    rest = one & ~same_mat & ~same_ml
    J['yes_purity_only']['umedib_by_metal_linker'] = dict(same_material=int(same_mat.sum()), same_metal_and_linker_other_material=nm(same_ml),
                                                          same_metal_and_linker_materials=pd.Series(grp[same_ml]).value_counts().to_dict(),
                                                          other=nm(rest), other_materials=pd.Series(grp[rest]).value_counts().to_dict(),
                                                          other_metals=sorted(set(c.split('||')[0] for c in chem[rest])))
    J['yes_purity_only']['umedib'] = dict(material=grp[um], chemistry=chem[um], label=int(y[um]),
                                          same_material=int(same_mat.sum()), same_chemistry_other_material=nm(same_chem),
                                          same_chemistry_materials=pd.Series(grp[same_chem]).value_counts().to_dict(),
                                          unrelated=nm(other), unrelated_materials=pd.Series(grp[other]).value_counts().to_dict(),
                                          largest_other_material=pd.Series(grp[one & ~same_mat]).value_counts().head(1).to_dict())
conc = []
doi = mg.doi.values
for i in np.where(lsYes)[0]:
    nb = G.idx[i, :SEL_YES[0]]
    ysd = nb[y[nb] == 1]
    vc = pd.Series(grp[ysd]).value_counts()
    top2 = list(vc.index[:2])
    conc.append(dict(cif=short(cif[i]), material=grp[i], category=cat[i], ls_P=float(r['py'][i]), rf_P=float(p[i]), support=int(r['sy'][i]),
                     n_yes=int(len(ysd)), n_no=int((y[nb] == 0).sum()), n_mats=int(len(vc)), top1=vc.index[0], top1_frac=float(vc.iloc[0] / len(ysd)),
                     top2=' | '.join(top2), top2_frac=float(vc.iloc[:2].sum() / len(ysd)),
                     top2_dois=' | '.join(sorted(set(doi[ysd[np.isin(grp[ysd], top2)]])))))
C = pd.DataFrame(conc)
PAIR = ['M::Cu||JYMQBKIOEMKLHD||nbo', 'M::Cu||GXHVGMYXKICGKR||nbo']
pf, pd_ = [], set()
for i in np.where(lsYes)[0]:
    nb = G.idx[i, :SEL_YES[0]]
    ysd = nb[y[nb] == 1]
    inpair = np.isin(grp[ysd], PAIR)
    pf.append(float(inpair.mean()))
    pd_ |= set(doi[ysd[inpair]])
C['cu_nbo_pair_frac'] = pf
pair_seeds = np.where(~U & np.isin(grp, PAIR))[0]
J['cu_nbo_pair'] = dict(materials=PAIR, dois_of_neighbouring_seeds=sorted(pd_), n_ge_088=int((C.cu_nbo_pair_frac >= 0.88).sum()),
                        n_ge_090=int((C.cu_nbo_pair_frac >= 0.90).sum()), n_ge_085=int((C.cu_nbo_pair_frac >= 0.85).sum()),
                        frac_range_ge_088=[float(C.cu_nbo_pair_frac[C.cu_nbo_pair_frac >= 0.88].min()), float(C.cu_nbo_pair_frac[C.cu_nbo_pair_frac >= 0.88].max())],
                        frac_range_ge_085=[float(C.cu_nbo_pair_frac[C.cu_nbo_pair_frac >= 0.85].min()), float(C.cu_nbo_pair_frac[C.cu_nbo_pair_frac >= 0.85].max())],
                        seeds_of_pair=dict(n=int(len(pair_seeds)), yes=int((y[pair_seeds] == 1).sum()), no=int((y[pair_seeds] == 0).sum()),
                                           by_material=pd.Series(grp[pair_seeds]).value_counts().to_dict(), dois=sorted(set(doi[pair_seeds]))),
                        support3_only=dict(n=int((C.support == 3).sum()), n_pair_ge_088=int(((C.support == 3) & (C.cu_nbo_pair_frac >= 0.88)).sum())))
print('cu nbo pair', J['cu_nbo_pair'])
C.to_csv(OUT + 'ls_yes_targets.csv', index=False)
J['ls_yes_concentration'] = dict(n=int(len(C)), top2_ge_088=int((C.top2_frac >= 0.88).sum()), top2_ge_085=int((C.top2_frac >= 0.85).sum()),
                                 top2_frac_range_of_those=[float(C.top2_frac[C.top2_frac >= 0.88].min()), float(C.top2_frac[C.top2_frac >= 0.88].max())] if (C.top2_frac >= 0.88).any() else None,
                                 top2_sets=C[C.top2_frac >= 0.88].groupby('top2').agg(n=('cif', 'size'), dois=('top2_dois', 'first')).reset_index().to_dict('records'),
                                 top2_frac_min=float(C.top2_frac.min()), top2_frac_max=float(C.top2_frac.max()))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 60)
print(C.drop(columns=['top2_dois']).round(3).to_string(index=False))
print(json.dumps(J['yes_purity_only'], indent=1)); print(json.dumps(J['ls_yes_concentration'], indent=1))

# Observation 1: Yes at the No threshold on the targets, and the neighbour-label metric relative to the base rate
J['targets_yes_at_0995_support3'] = {f'k{k}_a{a}': int(gate('yes', (k, a), 3, T_NO).sum()) for (k, a) in runs}
J['targets_yes_at_0995_purity_only'] = {f'k{k}_a{a}': int((U & (runs[(k, a)]['py'] > T_NO)).sum()) for (k, a) in runs}
# does P > 0.995 in practice require a neighbourhood without seeds of the other class? (the run of each gate)
def purity995(ka):
    r_ = runs[ka]
    nbk = G.idx[:, :ka[0]]
    n_yes_nb, n_no_nb = (y[nbk] == 1).sum(axis=1), (y[nbk] == 0).sum(axis=1)
    pn995, py995 = U & (r_['pn'] > T_NO), U & (r_['py'] > T_NO)
    return dict(
        k=ka[0], alpha=ka[1],
        no=dict(n=int(pn995.sum()), with_zero_yes_seeds=int((pn995 & (n_yes_nb == 0)).sum()), max_yes_seeds=int(n_yes_nb[pn995].max()) if pn995.any() else 0),
        yes=dict(n=int(py995.sum()), with_zero_no_seeds=int((py995 & (n_no_nb == 0)).sum()), max_no_seeds=int(n_no_nb[py995].max()) if py995.any() else 0,
                 with_support_ge3=int((py995 & (r_['sy'] >= 3)).sum()), with_zero_seeds_of_any_class=int((py995 & (n_no_nb == 0) & (n_yes_nb == 0)).sum())),
        targets_with_zero_no_seeds_among_neighbours=int((U & (n_no_nb == 0)).sum()), targets_with_zero_yes_seeds_among_neighbours=int((U & (n_yes_nb == 0)).sum()))


J['purity_0995_neighbourhoods'] = dict(no_run=purity995(SEL_NO), yes_run=purity995(SEL_YES))
print('purity 0.995 neighbourhoods', J['purity_0995_neighbourhoods'])
# Yes at the No threshold with support >= 3 on the targets, whole (k, alpha) grid
J['targets_yes_at_0995_support3_grid'] = {}
for k_, a_ in itertools.product(KS, AL):
    py_g, _, _ = G.spread(y, k_, a_)
    J['targets_yes_at_0995_support3_grid'][f'k{k_}_a{a_}'] = int((U & (py_g > T_NO) & (G.support(y, k_, 1) >= 3)).sum())
print('targets passing as yes at 0.995, support >= 3:', J['targets_yes_at_0995_support3_grid'])
Si = S
_, I = NearestNeighbors(n_neighbors=400).fit(X[Si]).kneighbors(X[Si])
gs, ys = grp[Si], y[Si]
fy, fn = [], []
for i in range(len(Si)):
    nb = [j for j in I[i, 1:] if gs[j] != gs[i]][:15]
    assert len(nb) == 15
    (fy if ys[i] == 1 else fn).append((ys[nb] == ys[i]).mean())
J['neighbour_label_metric_15'] = dict(yes_same=float(np.mean(fy)), yes_base=float((ys == 1).mean()), no_same=float(np.mean(fn)), no_base=float((ys == 0).mean()),
                                      yes_lift=float(np.mean(fy) / (ys == 1).mean()), no_lift=float(np.mean(fn) / (ys == 0).mean()))
print('neighbour-label metric', J['neighbour_label_metric_15'])

# the "conservative symmetric gate" of the old text: k = 15, alpha = 0.2, support >= 5
J['k15_a02_support5_targets'] = dict(no=nm(gate('no', (15, 0.2), 5)), yes_at_0885=nm(gate('yes', (15, 0.2), 5)), yes_at_0995=nm(gate('yes', (15, 0.2), 5, T_NO)))
v = Pn[(Pn.k == 15) & (Pn.alpha == 0.2) & (Pn.support == 5)]
J['k15_a02_support5_validation'] = {r_.gate: dict(n=int(r_.n), correct=int(r_.c), acc=float(r_.acc)) for r_ in v.itertuples()}

# the released file ls_gas_predictions_k50.csv: which run is it? (labels before correction, as released)
rel = {}
_OLD = os.environ.get('OLD_LS_DIR')            # folder with the previously released LS files (not in this repository); block skipped when unset
for fn_ in (('ls_gas_predictions_k50.csv', 'ls_gas_predictions_k75.csv', 'ls_gas_prediction.csv') if _OLD else ()):
    t = pd.read_csv(os.path.join(_OLD, fn_))
    col = [c for c in t.columns if c.lower() in ('p_yes', 'p_yes_ls', 'ls_p_yes')]
    t = t.set_index('cif_name').loc[cif]
    best = {}
    for k, a in itertools.product(KS, AL):
        py, _, _ = G.spread(y_old, k, a)
        best[f'k{k}_a{a}'] = float(np.abs(py - t[col[0]].values).max())
    kk = min(best, key=best.get)
    rel[fn_] = dict(columns=list(t.columns), closest_run=kk, max_abs_diff=best[kk], max_abs_diff_k50_a01=best['k50_a0.1'], max_abs_diff_k75_a02=best['k75_a0.2'])
J['released_files'] = rel
print(json.dumps(rel, indent=1))

# per-target file (for Figure S13 and for release)
T = pd.DataFrame({'cif': cif[U], 'material_id': grp[U], 'gas_label': g[U], 'category': cat[U], 'rf_P_yes': p[U],
                  'rf_call': np.where(rfNo[U], 'No', np.where(rfYes[U], 'Yes', 'unsure'))})
for ka in dict.fromkeys([SEL_NO, SEL_YES]):                      # P(yes) and P(no) of the propagation of each gate (one or two runs)
    T[f'ls_P_yes_k{ka[0]}_a{ka[1]}'] = runs[ka]['py'][U]
    T[f'ls_P_no_k{ka[0]}_a{ka[1]}'] = runs[ka]['pn'][U]
T['n_no_seed_materials'] = runs[SEL_NO]['sn'][U]                 # among the k neighbours of the No gate
T['n_yes_seed_materials'] = runs[SEL_YES]['sy'][U]               # among the k neighbours of the Yes gate
T['ls_confident_no'], T['ls_confident_yes'] = lsNo[U], lsYes[U]
for ka in sorted(set(NO_OK) - {SEL_NO}):                         # the other No settings that meet the criterion
    T[f'ls_confident_no_k{ka[0]}_a{ka[1]}'] = gate('no', ka)[U]
T['ls_confident_yes_k75_a0.1'] = gate('yes', (75, 0.1))[U]
# the per-target file carries the columns of the propagation of each gate (they differ when the two gates use different settings)
assert all(f'ls_P_{c_}_k{ka[0]}_a{ka[1]}' in T.columns for ka in (SEL_NO, SEL_YES) for c_ in ('yes', 'no'))
assert not (T.ls_confident_no & T.ls_confident_yes).any()
T.to_csv(OUT + 'ls_gas_targets_rev2.csv', index=False)
json.dump(J, open(OUT + 'compute.json', 'w'), indent=1, default=str)
print('written', OUT)
