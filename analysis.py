"""Figure 3 and the earlier fixed-threshold results.

Trains the TF-IDF + logistic-regression classifier on the full training set and
evaluates the three routing policies on the test set at fixed thresholds
(tau = 0.8, theta = 0.10), plus a sweep over tau for Figure 3.

The validation-selected results reported in Tables 4-7 are produced by analysis2.py.

Reads:  data/train.csv, data/test.csv, intent_coding.csv
Writes: results_main.csv, results_sensitivity.csv, tier_confusion.csv,
        figure3_threshold_sweep.png
Seed fixed at 42.
"""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 42
TAU, THETA = 0.80, 0.10
B = 1000

# ------------------------------------------------------------------ data and coding
tr = pd.read_csv('data/train.csv')
te = pd.read_csv('data/test.csv')
ic = pd.read_csv('intent_coding.csv')
tier_of = {r.intent: max(int(r.C), int(r.U), int(r.I)) for r in ic.itertuples()}
assert set(tr.category) == set(tier_of)

# ------------------------------------------------------------------ classifier
w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True, min_df=2)
X_tr = hstack([w.fit_transform(tr.text), c.fit_transform(tr.text)]).tocsr()
X_te = hstack([w.transform(te.text), c.transform(te.text)]).tocsr()
clf = LogisticRegression(C=20, max_iter=2000, random_state=SEED).fit(X_tr, tr.category)

P = clf.predict_proba(X_te)
cls = clf.classes_
tcls = np.array([tier_of[k] for k in cls])
pred = cls[P.argmax(1)]
conf = P.max(1)
y = te.category.values
ok = pred == y
t_true = np.array([tier_of[k] for k in y])
t_pred = np.array([tier_of[k] for k in pred])
print(f"Test accuracy: {ok.mean():.4f}")


def tmax_at(theta):
    """Highest tier among all intents with predicted probability >= theta."""
    return np.array([tcls[row >= theta].max() if (row >= theta).any() else 3 for row in P])


def route(policy, tau, theta=THETA):
    if policy == 'P0':
        return conf >= tau
    if policy == 'P1':
        return (t_pred == 1) & (conf >= tau)
    if policy == 'P2':
        return (tmax_at(theta) == 1) & (conf >= tau)
    raise ValueError(policy)


def under_tier(policy, theta=THETA):
    if policy == 'P1':
        return (t_pred[t_true == 3] < 3).mean()
    if policy == 'P2':
        return (tmax_at(theta)[t_true == 3] < 3).mean()
    return np.nan


def metrics(auto, idx=None):
    a, o, t = (auto, ok, t_true) if idx is None else (auto[idx], ok[idx], t_true[idx])
    return {'automation': a.mean(),
            'uar_t3': a[t == 3].mean(),
            'uar_t23': a[t >= 2].mean(),
            'harm': ((~o) & (t >= 2) & a).sum() / max(a.sum(), 1),
            'over_esc_t1': (~a[t == 1]).mean()}


NAMES = {'P0': 'P0 confidence-only',
         'P1': 'P1 predicted-intent tier',
         'P2': 'P2 risk-conservative tier'}

# ------------------------------------------------------------------ results_main.csv
rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, len(te), len(te)) for _ in range(B)]
tmax_main = tmax_at(THETA)
rows = []
for p in ['P0', 'P1', 'P2']:
    auto = route(p, TAU)
    point = metrics(auto)
    boots = [metrics(auto, i) for i in boot_idx]
    row = {'policy': NAMES[p]}
    for k in point:
        lo, hi = np.percentile([b[k] for b in boots], [2.5, 97.5])
        row[k] = f"{100*point[k]:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
    if p == 'P0':
        row['under_tier_t3'] = '–'
    else:
        tp = t_pred if p == 'P1' else tmax_main
        ut = [(tp[i][t_true[i] == 3] < 3).mean() for i in boot_idx]
        lo, hi = np.percentile(ut, [2.5, 97.5])
        row['under_tier_t3'] = f"{100*under_tier(p):.1f} [{100*lo:.1f}, {100*hi:.1f}]"
    rows.append(row)
pd.DataFrame(rows).to_csv('results_main.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------------ results_sensitivity.csv
sens = []
for tau in [0.6, 0.7, 0.8, 0.9]:
    for p, thetas in [('P0', [None]), ('P1', [None]), ('P2', [0.05, 0.10, 0.20])]:
        for th in thetas:
            m = metrics(route(p, tau, th if th is not None else THETA))
            sens.append({'tau': tau, 'theta': th if th is not None else '', 'policy': NAMES[p],
                         **{k: round(100*v, 1) for k, v in m.items()},
                         'under_tier_t3': '' if p == 'P0' else
                         round(100*under_tier(p, th if th is not None else THETA), 1)})
pd.DataFrame(sens).to_csv('results_sensitivity.csv', index=False)

# ------------------------------------------------------------------ tier_confusion.csv
conf_mat = pd.crosstab(pd.Series(t_true, name='true tier'), pd.Series(t_pred, name='predicted tier'))
conf_mat.to_csv('tier_confusion.csv')
print("\nTier confusion (rows = true, columns = predicted):")
print(conf_mat)

# ------------------------------------------------------------------ Figure 3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

taus = np.round(np.arange(0.30, 0.951, 0.05), 2)
curves = {p: [metrics(route(p, t)) for t in taus] for p in ['P0', 'P1', 'P2']}
style = {'P0': dict(color='#8c3a2e', marker='o', ls='-', label='P0 confidence-only'),
         'P1': dict(color='#1f4e79', marker='s', ls='--', label='P1 predicted-intent tier'),
         'P2': dict(color='#2e7d4f', marker='^', ls='-', label='P2 risk-conservative tier')}

plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True, dpi=300)
for p in ['P0', 'P1', 'P2']:
    ax1.plot(taus, [100*m['uar_t23'] for m in curves[p]], ms=4, lw=1.2, **style[p])
    ax2.plot(taus, [100*m['over_esc_t1'] for m in curves[p]], ms=4, lw=1.2, **style[p])
ax1.set_ylabel('Tier 2–3 requests\nunder AI authority (%)')
ax2.set_ylabel('Tier 1 requests not\nresolved by AI (%)')
ax2.set_xlabel('Confidence threshold τ')
for ax, lab in [(ax1, '(a)'), (ax2, '(b)')]:
    ax.text(0.03, 0.92, lab, transform=ax.transAxes, fontweight='bold', va='top')
    ax.grid(alpha=0.3, lw=0.5)
ax1.legend(frameon=False, loc='center left', fontsize=7)
fig.tight_layout()
fig.savefig('figure3_threshold_sweep.png', dpi=300)
print("\nWrote results_main.csv, results_sensitivity.csv, tier_confusion.csv, figure3_threshold_sweep.png")
