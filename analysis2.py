"""Reproduces all results in Section V (Tables 4–7): validation-based thresholds,
paired bootstrap, calibration, robustness checks, baselines and cost model.
Reads data/train.csv, data/test.csv and intent_coding.csv. Seed fixed at 42."""
import numpy as np, pandas as pd, json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
SEED = 42
rng = np.random.default_rng(SEED)

tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
_ic = pd.read_csv('intent_coding.csv')
CODE = {r.intent: (int(r.C), int(r.U), int(r.I)) for r in _ic.itertuples()}
tier_of = {k: max(v) for k, v in CODE.items()}
labels = sorted(tr.category.unique())
assert set(labels) == set(CODE)

# ---------------------------------------------------------------- 1. SPLIT
tr_fit, val = train_test_split(tr, test_size=0.20, random_state=SEED, stratify=tr.category)
print(f"train_fit={len(tr_fit)}  val={len(val)}  test={len(te)} (test untouched until step 4)")

def fit_vectorizers_and_model(train_df):
    w = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=1)
    c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), sublinear_tf=True, min_df=2)
    Xw = w.fit_transform(train_df.text); Xc = c.fit_transform(train_df.text)
    X = hstack([Xw, Xc]).tocsr()
    clf = LogisticRegression(C=20, max_iter=2000, random_state=SEED).fit(X, train_df.category)
    return w, c, clf

def transform(w, c, df):
    return hstack([w.transform(df.text), c.transform(df.text)]).tocsr()

w, c, clf = fit_vectorizers_and_model(tr_fit)
cls = clf.classes_; tcls = np.array([tier_of[k] for k in cls])

Pv = clf.predict_proba(transform(w, c, val))
predv = cls[Pv.argmax(1)]; confv = Pv.max(1)
yv = val.category.values; tv_true = np.array([tier_of[k] for k in yv]); tv_pred = np.array([tier_of[k] for k in predv])

def route(P, pred, conf, t_true, tau, theta, rule='max_predicted'):
    """rule: 'confidence_only' (P0), 'predicted_tier' (P1), 'risk_conservative' (P2)."""
    if rule == 'confidence_only':
        return conf >= tau
    if rule == 'predicted_tier':
        tpred = np.array([tier_of[k] for k in pred])
        return (tpred == 1) & (conf >= tau)
    if rule == 'risk_conservative':
        tmax = np.array([tcls[row >= theta].max() if (row >= theta).any() else 3 for row in P])
        return (tmax == 1) & (conf >= tau)
    raise ValueError(rule)

def metrics(auto, ok, t_true):
    m = {}
    m['automation'] = auto.mean()
    m['uar_t3'] = auto[t_true==3].mean() if (t_true==3).any() else np.nan
    m['uar_t23'] = auto[t_true>=2].mean() if (t_true>=2).any() else np.nan
    m['harm'] = ((~ok)&(t_true>=2)&auto).sum()/max(auto.sum(),1)
    m['over_esc_t1'] = (~auto[t_true==1]).mean() if (t_true==1).any() else np.nan
    return m

# ------------------------------------------------- 2. THRESHOLD SELECTION ON VAL (not test)
# Rule, stated explicitly: among tau in a grid, pick the SMALLEST tau (most automation) such
# that unsafe-automation rate on Tier 2-3 requests is <=1% on the validation set, for P1 and P2.
# theta is fixed by the same logic within P2's own grid, nested.
tau_grid = np.round(np.arange(0.50, 0.96, 0.01), 2)
theta_grid = [0.05, 0.10, 0.15, 0.20]

def pick_tau(rule, theta=0.10):
    for tau in tau_grid:  # ascending: first tau meeting the safety bar = most permissive safe choice
        auto = route(Pv, predv, confv, tv_true, tau, theta, rule)
        ok = predv == yv
        m = metrics(auto, ok, tv_true)
        if not np.isnan(m['uar_t23']) and m['uar_t23'] <= 0.01:
            return tau, m
    return tau_grid[-1], metrics(route(Pv,predv,confv,tv_true,tau_grid[-1],theta,rule), predv==yv, tv_true)

tau_p0, _ = pick_tau('confidence_only')
tau_p1, _ = pick_tau('predicted_tier')
best = None
for theta in theta_grid:
    tau, m = pick_tau('risk_conservative', theta)
    score = m['automation']
    if best is None or score > best[2]:
        best = (tau, theta, score)
tau_p2, theta_p2, _ = best
print(f"Selected on validation set (not test): P0 tau={tau_p0}, P1 tau={tau_p1}, P2 tau={tau_p2} theta={theta_p2}")

# ------------------------------------------------- 3. REFIT ON FULL TRAIN, EVALUATE ON TEST ONCE
w2, c2, clf2 = fit_vectorizers_and_model(tr)  # use all training data for the final model, as before
cls2 = clf2.classes_; tcls2 = np.array([tier_of[k] for k in cls2])
Pt = clf2.predict_proba(transform(w2, c2, te))
predt = cls2[Pt.argmax(1)]; conft = Pt.max(1)
yt = te.category.values; tt_true = np.array([tier_of[k] for k in yt]); tt_pred = np.array([tier_of[k] for k in predt])
ok_t = predt == yt
print(f"Final test accuracy (refit on full train): {ok_t.mean():.4f}")

POLICIES_VAL = {'P0 confidence-only': ('confidence_only', tau_p0, None),
                 'P1 predicted-intent tier': ('predicted_tier', tau_p1, None),
                 'P2 risk-conservative tier': ('risk_conservative', tau_p2, theta_p2)}

def route2(P, pred, conf, rule, tau, theta):
    if rule == 'confidence_only': return conf >= tau
    if rule == 'predicted_tier':
        tpred = np.array([tier_of[k] for k in pred]); return (tpred==1)&(conf>=tau)
    tmax = np.array([tcls2[row>=theta].max() if (row>=theta).any() else 3 for row in P])
    return (tmax==1)&(conf>=tau)

base_auto = {}
for name,(rule,tau,theta) in POLICIES_VAL.items():
    base_auto[name] = route2(Pt, predt, conft, rule, tau, theta)

def full_metrics(auto, ok, t_true, extra=None):
    m = metrics(auto, ok, t_true)
    if extra == 'under_tier':
        pass
    return m

rows = []
for name, auto in base_auto.items():
    m = metrics(auto, ok_t, tt_true)
    rows.append({'policy': name, **{k: round(100*v,2) for k,v in m.items()}})
val_tuned_main = pd.DataFrame(rows)
val_tuned_main.to_csv('results_valtuned_main.csv', index=False)
print("\n=== Validation-tuned thresholds, evaluated once on untouched test set ===")
print(val_tuned_main.to_string(index=False))

# under-tiering for P1, P2
under_p1 = (tt_pred[tt_true==3] < 3).mean()
tmax_p2 = np.array([tcls2[row>=theta_p2].max() if (row>=theta_p2).any() else 3 for row in Pt])
under_p2 = (tmax_p2[tt_true==3] < 3).mean()
print(f"Under-tiering of true Tier 3: P1={100*under_p1:.2f}%  P2={100*under_p2:.2f}%")

# ------------------------------------------------- 4. PAIRED BOOTSTRAP ON DIFFERENCES
B = 1000
diffs = {f"{a} minus {b}": {'automation':[], 'uar_t23':[], 'harm':[]} for a,b in [('P1','P0'),('P2','P0'),('P2','P1')]}
name_map = {'P0':'P0 confidence-only','P1':'P1 predicted-intent tier','P2':'P2 risk-conservative tier'}
idx_all = np.arange(len(te))
for _ in range(B):
    idx = rng.integers(0, len(te), len(te))
    m = {}
    for short,name in name_map.items():
        auto_b = base_auto[name][idx]
        m[short] = metrics(auto_b, ok_t[idx], tt_true[idx])
    for a,b in [('P1','P0'),('P2','P0'),('P2','P1')]:
        for k in ['automation','uar_t23','harm']:
            diffs[f"{a} minus {b}"][k].append(m[a][k]-m[b][k])

boot_rows=[]
for pair, d in diffs.items():
    row={'comparison': pair}
    for k,v in d.items():
        v=np.array(v); lo,hi=np.percentile(v,[2.5,97.5])
        row[f'{k}_diff_pp']=f"{100*np.mean(v):.2f} [{100*lo:.2f}, {100*hi:.2f}]"
    boot_rows.append(row)
paired_df = pd.DataFrame(boot_rows)
paired_df.to_csv('results_paired_bootstrap.csv', index=False)
print("\n=== Paired bootstrap differences between policies (percentage points) ===")
print(paired_df.to_string(index=False))

# ------------------------------------------------- 5. CALIBRATION (ECE)
def ece(conf, correct, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    total=len(conf); e=0.0; rows=[]
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        m = (conf>lo)&(conf<=hi) if i>0 else (conf>=lo)&(conf<=hi)
        if m.sum()==0:
            rows.append((f"{lo:.1f}-{hi:.1f}",0,np.nan,np.nan)); continue
        acc=correct[m].mean(); avgconf=conf[m].mean(); frac=m.sum()/total
        e += frac*abs(acc-avgconf)
        rows.append((f"{lo:.1f}-{hi:.1f}", int(m.sum()), round(100*avgconf,1), round(100*acc,1)))
    return e, rows

ece_val, ece_rows = ece(conft, ok_t)
cal_df = pd.DataFrame(ece_rows, columns=['confidence_bin','n','mean_confidence_%','accuracy_%'])
cal_df.to_csv('results_calibration.csv', index=False)
print(f"\n=== Calibration: ECE = {ece_val*100:.2f}% (10 bins) ===")
print(cal_df.to_string(index=False))

# ------------------------------------------------- 6a. ROBUSTNESS: RANDOM RE-TIERING
true_tiers_arr = np.array([tier_of[k] for k in labels])
n_shuffle = 300
shuffled_uar = []
for _ in range(n_shuffle):
    perm = rng.permutation(true_tiers_arr)
    shuffled_tier_of = dict(zip(labels, perm))
    tpred_sh = np.array([shuffled_tier_of[k] for k in predt])
    auto_sh = (tpred_sh==1)&(conft>=tau_p1)
    ttrue_sh = np.array([shuffled_tier_of[k] for k in yt])
    shuffled_uar.append(metrics(auto_sh, ok_t, ttrue_sh)['uar_t23'])
shuffled_uar = np.array(shuffled_uar)
true_uar_p1 = metrics(base_auto['P1 predicted-intent tier'], ok_t, tt_true)['uar_t23']
percentile_rank = (shuffled_uar < true_uar_p1).mean()
print(f"\n=== Random re-tiering control (n={n_shuffle} shuffles of the 77 intent-tier labels) ===")
print(f"True coding's P1 UAR(Tier2-3) = {100*true_uar_p1:.2f}%; random-shuffle UAR mean = {100*shuffled_uar.mean():.2f}% (sd {100*shuffled_uar.std():.2f}); "
      f"true coding is lower than {100*(1-percentile_rank):.1f}% of random shuffles" if percentile_rank<0.5 else
      f"true coding beats {100*percentile_rank:.1f}% of random shuffles")
pd.DataFrame({'shuffled_uar_t23_pct': 100*shuffled_uar}).to_csv('results_random_retiering.csv', index=False)

# ------------------------------------------------- 6b. ROBUSTNESS: ALTERNATIVE AGGREGATION RULES
def code_lookup(intent): return CODE[intent]
sum_tier = {}
weighted_tier = {}
for k,(C,U,I) in CODE.items():
    s = C+U+I
    sum_tier[k] = 3 if s>=8 else (2 if s>=5 else 1)
    wscore = 0.5*I + 0.3*U + 0.2*C
    weighted_tier[k] = 3 if wscore>=2.5 else (2 if wscore>=1.6 else 1)

n_changed_sum = sum(1 for k in CODE if sum_tier[k]!=tier_of[k])
n_changed_wt = sum(1 for k in CODE if weighted_tier[k]!=tier_of[k])
print(f"\n=== Alternative aggregation rules vs. MAX rule (77 intents) ===")
print(f"SUM rule: {n_changed_sum}/77 intents change tier")
print(f"WEIGHTED rule (0.5I+0.3U+0.2C): {n_changed_wt}/77 intents change tier")

def metrics_under_rule(rule_dict, tau):
    tpred_r = np.array([rule_dict[k] for k in predt])
    auto_r = (tpred_r==1)&(conft>=tau)
    ttrue_r = np.array([rule_dict[k] for k in yt])
    return metrics(auto_r, ok_t, ttrue_r)

m_max = metrics(base_auto['P1 predicted-intent tier'], ok_t, tt_true)
m_sum = metrics_under_rule(sum_tier, tau_p1)
m_wt  = metrics_under_rule(weighted_tier, tau_p1)
agg_rows=[{'rule':'MAX (used in paper)', **{k:round(100*v,2) for k,v in m_max.items()}},
          {'rule':'SUM (C+U+I thresholded)', **{k:round(100*v,2) for k,v in m_sum.items()}},
          {'rule':'WEIGHTED (0.5I+0.3U+0.2C)', **{k:round(100*v,2) for k,v in m_wt.items()}}]
agg_df = pd.DataFrame(agg_rows); agg_df.to_csv('results_aggregation_robustness.csv', index=False)
print(agg_df.to_string(index=False))

json.dump({'tau_p0':float(tau_p0),'tau_p1':float(tau_p1),'tau_p2':float(tau_p2),'theta_p2':float(theta_p2),
           'test_accuracy_refit':float(ok_t.mean())}, open('selected_params.json','w'), indent=2)
print("\nStage A (validation thresholds, paired bootstrap, calibration, robustness) complete.")

# ------------------------------------------------- 7. ADDITIONAL BASELINES
print("\n" + "="*70 + "\nADDITIONAL BASELINES\n" + "="*70)

# 7a. Intent allowlist: resolve autonomously whenever predicted top-1 intent is in the
# Tier-1 allowlist, with NO confidence gate at all (this is what many real deployments do:
# a fixed list of "safe" intents, trusted whenever the classifier's top guess lands there).
allowlist = set(k for k,v in tier_of.items() if v == 1)
auto_allow = np.array([p in allowlist for p in predt])
m_allow = metrics(auto_allow, ok_t, tt_true)
print("Allowlist (top-1 in Tier-1 set, no confidence gate):", {k: round(100*v,2) for k,v in m_allow.items()})

# 7b. Selective prediction / learning-to-defer via margin between top-1 and top-2 probability.
# Margin threshold chosen on validation set using the same safety rule as before (UAR<=1%).
Pv2 = clf.predict_proba(transform(w,c,val))  # reuse val-split model for threshold selection
def margin_of(P):
    s = np.sort(P, axis=1)
    return s[:,-1]-s[:,-2]
marginv = margin_of(Pv2)
margin_grid = np.round(np.arange(0.05,0.95,0.02),2)
chosen_margin = margin_grid[-1]
for mg in margin_grid:
    tpredv2 = np.array([tier_of[k] for k in predv])
    autov = (tpredv2==1)&(marginv>=mg)
    mm = metrics(autov, predv==yv, tv_true)
    if not np.isnan(mm['uar_t23']) and mm['uar_t23']<=0.01:
        chosen_margin = mg; break
margint = margin_of(Pt)
auto_margin = (tt_pred==1)&(margint>=chosen_margin)
m_margin = metrics(auto_margin, ok_t, tt_true)
print(f"Selective prediction (margin>={chosen_margin}, picked on val):", {k: round(100*v,2) for k,v in m_margin.items()})

# 7c. Split-conformal prediction sets (calibrated on the validation split).
# Nonconformity score = 1 - predicted probability of the true class (standard softmax score).
Pv3 = Pv2
true_idx_v = np.array([np.where(cls==y)[0][0] for y in yv])
scores_v = 1 - Pv3[np.arange(len(yv)), true_idx_v]
alpha = 0.10  # target 90% coverage
qhat = np.quantile(scores_v, np.ceil((len(yv)+1)*(1-alpha))/len(yv))
def conformal_set(P, qhat):
    return [set(cls[row >= 1-qhat]) for row in P]
sets_t = conformal_set(Pt, qhat)
set_sizes = np.array([len(s) for s in sets_t])
coverage = np.mean([yt[i] in sets_t[i] for i in range(len(yt))])
singleton = set_sizes==1
singleton_tier1 = np.array([ (len(sets_t[i])==1) and (tier_of[list(sets_t[i])[0]]==1) for i in range(len(yt))])
m_conf = metrics(singleton_tier1, ok_t, tt_true)
print(f"Conformal (90% target coverage, qhat={qhat:.3f}): empirical coverage={100*coverage:.1f}%, "
      f"mean set size={set_sizes.mean():.2f}, resolves only if singleton & Tier1:", {k: round(100*v,2) for k,v in m_conf.items()})

baseline_rows = [
 {'baseline':'P1 (confidence+predicted-tier, this paper)', **{k:round(100*v,2) for k,v in m_max.items()}},
 {'baseline':'Intent allowlist (no confidence gate)', **{k:round(100*v,2) for k,v in m_allow.items()}},
 {'baseline':'Selective prediction (margin-based deferral)', **{k:round(100*v,2) for k,v in m_margin.items()}},
 {'baseline':f'Split-conformal (90% coverage, singleton+Tier1 only)', **{k:round(100*v,2) for k,v in m_conf.items()}},
]
pd.DataFrame(baseline_rows).to_csv('results_additional_baselines.csv', index=False)
print(pd.DataFrame(baseline_rows).to_string(index=False))

# ------------------------------------------------- 8. EXPECTED-LOSS / HANDLING-COST MODEL
# Illustrative unit costs (clearly NOT empirical figures; stated as assumptions in the paper).
C_AUTO = 0.20     # USD, fully-automated resolution (compute + negligible marginal cost)
C_HUMAN = 4.00    # USD, average fully-human-handled or human-reviewed ticket
C_HARM = 500.00   # USD, expected loss from one harmful automated error on a Tier>=2 request
N_TICKETS = 100000  # illustrative monthly volume for scaling

def expected_cost(auto, ok, t_true):
    n = len(auto)
    n_auto = auto.sum()
    n_unsafe = ((t_true>=2)&auto).sum()
    n_human = n - n_auto
    # Primary risk-cost event matches the paper's unsafe-autonomy rate:
    # any autonomous resolution of a true Tier 2-3 request, whether or not intent prediction was correct.
    total = n_auto*C_AUTO + n_human*C_HUMAN + n_unsafe*C_HARM
    per_ticket = total/n
    return per_ticket*N_TICKETS, n_unsafe, n_auto

cost_rows=[]
for name, auto in base_auto.items():
    cost, n_harm, n_auto = expected_cost(auto, ok_t, tt_true)
    cost_rows.append({'policy':name, 'monthly_expected_cost_usd': round(cost,0),
                       'unsafe_autonomous_cases_per_'+str(N_TICKETS): round(n_harm/len(te)*N_TICKETS),
                       'automated_tickets_per_'+str(N_TICKETS): round(n_auto/len(te)*N_TICKETS)})
cost_df = pd.DataFrame(cost_rows); cost_df.to_csv('results_expected_cost.csv', index=False)
print("\n=== Illustrative expected-cost model (assumed unit costs, NOT empirical) ===")
print(f"Assumptions: automated ticket=${C_AUTO}, human ticket=${C_HUMAN}, unsafe autonomous Tier>=2 resolution=${C_HARM}, volume={N_TICKETS}/month")
print(cost_df.to_string(index=False))

# sensitivity: at what C_HARM does P0 stop being cheaper than P1, if ever (it starts much more automated)
def cost_at(auto, ok, t_true, c_harm):
    n=len(auto); n_auto=auto.sum(); n_unsafe=((t_true>=2)&auto).sum(); n_human=n-n_auto
    return (n_auto*C_AUTO + n_unsafe*c_harm + n_human*C_HUMAN)/n*N_TICKETS
harm_grid = [50,100,250,500,1000,2500,5000,10000]
sens_rows=[]
for ch in harm_grid:
    row={'assumed_cost_per_unsafe_autonomous_case_usd': ch}
    for name,auto in base_auto.items():
        row[name]=round(cost_at(auto,ok_t,tt_true,ch),0)
    sens_rows.append(row)
sens_df=pd.DataFrame(sens_rows); sens_df.to_csv('results_cost_sensitivity.csv', index=False)
print("\n=== Cost sensitivity to the assumed unsafe-autonomy event cost ===")
print(sens_df.to_string(index=False))
