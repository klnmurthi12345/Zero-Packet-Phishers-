#!/usr/bin/env python3
"""
final_analysis.py -- Zero-Packet Phishers (ES 335)

Reproduces the numbers in audit.ipynb and adds the extra analyses shown on the website:
  0. integrity checks (features recomputed from raw URLs == stored features; split sizes; notebook numbers)
  1. trivial baselines (majority class, hand-written "HTTP or has a path" rule)
  2. bootstrap 95% confidence intervals on the held-out test set
  3. validation-only hyper-parameter tuning (test set touched once, at the end)
  4. leakage diagnostics: how many test URLs/hostnames also appear in train; domain-grouped re-split
  5. benchmark-artifact ablation (drop structure features), hard-subset evaluation
  6. extra lexical features (hostname entropy, consonant run)
  7. error analysis (shared-hosting platforms, buckets) and calibration of the Random Forest

Usage (from repo root, with phiusiil_phishing.csv present):
    python final_analysis.py
Outputs: results/final_results.json and results/final_analysis_log.txt
Every number on the website's "extra evidence" blocks comes from results/final_results.json.
"""
import json, math, os, re, sys, time, warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
N_BOOT = 1000
RAW_CSV = "phiusiil_phishing.csv"
WORK_CSV = "phiusiil_working_dataset.csv"
os.makedirs("results", exist_ok=True)
R = {}                       # everything that goes to JSON
_log = open("results/final_analysis_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg); _log.write(msg + "\n"); _log.flush()


FEATS = ['url_length', 'hostname_length', 'dot_count', 'hyphen_count', 'digit_count', 'digit_ratio',
         'special_symbols', 'subdomain_count', 'uses_https', 'host_is_ip', 'suspicious_words',
         'path_len_ratio', 'tld_length']
SUSPICIOUS_TOKENS = ['login', 'verify', 'secure', 'account', 'update', 'signin', 'banking', 'confirm']
IP_REGEX = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')


def extract_single_url_features(url_str):          # identical to audit.ipynb
    url_str = str(url_str).strip()
    if not (url_str.startswith('http://') or url_str.startswith('https://')):
        parsed = urlparse('http://' + url_str); uses_https = 0
    else:
        parsed = urlparse(url_str); uses_https = 1 if parsed.scheme.lower() == 'https' else 0
    hostname = parsed.hostname or ''
    path = parsed.path or ''
    url_length = len(url_str)
    digit_count = sum(c.isdigit() for c in url_str)
    host_parts = hostname.split('.')
    return [url_length, len(hostname), url_str.count('.'), hostname.count('-'), digit_count,
            digit_count / url_length if url_length > 0 else 0.0,
            sum(url_str.count(s) for s in ['@', '?', '=', '&', '%']),
            max(0, len(host_parts) - 2) if len(host_parts) > 2 else 0, uses_https,
            1 if IP_REGEX.match(hostname) else 0,
            sum(1 for kw in SUSPICIOUS_TOKENS if kw in url_str.lower()),
            len(path) / url_length if url_length > 0 else 0.0,
            len(host_parts[-1]) if len(host_parts) >= 2 else 0]


def auc(y, p):
    return float(roc_auc_score(y, p))


def cm_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                accuracy=(tn + tp) / (tn + fp + fn + tp),
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else float('nan'),
                fpr=fp / (fp + tn) if fp + tn else float('nan'),
                f1=f1_score(y, pred))


# --------------------------------------------------------------------------------------------------
# 0. Load, recompute features from raw URLs, reproduce split
# --------------------------------------------------------------------------------------------------
log("== 0. Data, integrity checks ==")
raw = pd.read_csv(RAW_CSV)
raw = raw.drop_duplicates(subset=['URL']).reset_index(drop=True)
work = pd.read_csv(WORK_CSV)
assert len(raw) == len(work) == 235370, (len(raw), len(work))
t0 = time.time()
Xre = pd.DataFrame([extract_single_url_features(u) for u in raw['URL']], columns=FEATS)
log(f"re-extracted {len(Xre)} URLs in {time.time()-t0:.2f}s")
assert np.allclose(Xre.values.astype(float), work[FEATS].values.astype(float)), "stored features != recomputed"
assert ((raw['label'] == 0).astype(int).values == work['is_phishing'].values).all(), "labels misaligned"
log("OK: stored features equal features recomputed from raw URLs; labels aligned")

X, y = work[FEATS], work['is_phishing']
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=SEED)
assert (len(X_train), len(X_val), len(X_test)) == (141222, 47074, 47074)
R['split'] = dict(train=len(X_train), val=len(X_val), test=len(X_test),
                  test_legit=int((y_test == 0).sum()), test_phish=int((y_test == 1).sum()))
log("split:", R['split'])

scaler = StandardScaler().fit(X_train)
Xs = {k: scaler.transform(v) for k, v in [('train', X_train), ('val', X_val), ('test', X_test)]}
lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xs['train'], y_train)
dt = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
rf = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(X_train, y_train)
p_te = {'Logistic Regression': lr.predict_proba(Xs['test'])[:, 1],
        'Decision Tree': dt.predict_proba(X_test)[:, 1],
        'Random Forest': rf.predict_proba(X_test)[:, 1]}
p_va_rf = rf.predict_proba(X_val)[:, 1]
fpr_v, tpr_v, thr_v = roc_curve(y_val, p_va_rf)
idx = np.where(fpr_v <= 0.01)[0]; best = idx[np.argmax(tpr_v[idx])]
TAU = float(thr_v[best])
log(f"tuned tau = {TAU:.4f} (notebook: 0.0809)")
assert abs(TAU - 0.0809) < 5e-5
for k, v in p_te.items():
    log(f"  test ROC-AUC {k}: {auc(y_test, v):.4f}")
assert round(auc(y_test, p_te['Random Forest']), 4) == 0.9969
assert round(auc(y_test, p_te['Logistic Regression']), 4) == 0.9961
assert round(auc(y_test, p_te['Decision Tree']), 4) == 0.9948
R['tau_tuned'] = TAU
# ---- check against what audit.ipynb exported (simulator_data.json): Experiment A and all 100 slider thresholds
SIM = json.load(open("simulator_data.json"))
_wk = pd.read_csv(WORK_CSV)
SUS = ['TLDLegitimateProb', 'URLSimilarityIndex']
for _name in ('Logistic Regression', 'Decision Tree'):
    _r = {}
    for _tag, _cols in (('without', FEATS), ('with', FEATS + SUS)):
        _sc = StandardScaler().fit(_wk.loc[X_train.index, _cols])
        if _name == 'Logistic Regression':
            _m = LogisticRegression(max_iter=1000, random_state=SEED).fit(_sc.transform(_wk.loc[X_train.index, _cols]), y_train)
            _p = _m.predict_proba(_sc.transform(_wk.loc[X_val.index, _cols]))[:, 1]
        else:
            _m = DecisionTreeClassifier(random_state=SEED).fit(_wk.loc[X_train.index, _cols], y_train)
            _p = _m.predict_proba(_wk.loc[X_val.index, _cols])[:, 1]
        _r[_tag] = auc(y_val, _p)
    exp = SIM['experiment_a_validation'][_name]
    assert round(_r['without'], 4) == exp['without'] and round(_r['with'], 4) == exp['with'], (_name, _r, exp)
    assert round(_r['with'] - _r['without'], 4) == exp['jump'], (_name, _r, exp)
    log(f"Experiment A reproduced for {_name}: {round(_r['without'],4)} -> {round(_r['with'],4)}")
for _pt in SIM['slider_curve']:
    _pr = (p_te['Random Forest'] >= _pt['threshold']).astype(int)
    _tn, _fp, _fn, _tp = confusion_matrix(y_test, _pr).ravel()
    assert (int(_tn), int(_fp), int(_fn), int(_tp)) == (_pt['tn'], _pt['fp'], _pt['fn'], _pt['tp']), _pt
log("OK: all", len(SIM['slider_curve']), "slider thresholds in simulator_data.json reproduce exactly")
R['reproduces_notebook_exports'] = dict(experiment_a_validation=SIM['experiment_a_validation'],
                                        slider_thresholds_checked=len(SIM['slider_curve']))
R['main_default_models'] = {k: dict(roc_auc=auc(y_test, v), **cm_metrics(y_test, (v >= 0.5).astype(int)))
                            for k, v in p_te.items()}
R['rf_at_tuned_tau'] = cm_metrics(y_test, (p_te['Random Forest'] >= TAU).astype(int))
log("RF @ tuned tau:", {k: round(v, 4) if isinstance(v, float) else v for k, v in R['rf_at_tuned_tau'].items()})

# --------------------------------------------------------------------------------------------------
# 1. Trivial baselines (parameter-free, evaluated on the test split)
# --------------------------------------------------------------------------------------------------
log("\n== 1. Baselines ==")
maj = np.zeros(len(y_test), dtype=int)
rule = ((X_test['uses_https'] == 0) | (X_test['path_len_ratio'] > 0)).astype(int).values
R['baselines'] = {
    'majority_class_always_legitimate': cm_metrics(y_test, maj),
    'rule_http_or_has_path': cm_metrics(y_test, rule),
    'note': 'The rule has no fitted parameters but was written after exploratory inspection of class-conditional '
            'feature rates on the full corpus (Section 3), so it is a diagnostic baseline, not an independent one.'}
for k in ('majority_class_always_legitimate', 'rule_http_or_has_path'):
    log(k, {a: round(b, 4) for a, b in R['baselines'][k].items() if isinstance(b, float)},
        {a: R['baselines'][k][a] for a in ('tn', 'fp', 'fn', 'tp')})

# Class-conditional presence rates on the full corpus (feeds the EDA chart)
full = work
pres = {'Uses HTTPS': full.uses_https == 1, 'Has a path': full.path_len_ratio > 0,
        'Has ? = & % @': full.special_symbols > 0, 'Has a digit': full.digit_count > 0,
        'Hyphen in host': full.hyphen_count > 0, 'Suspicious keyword': full.suspicious_words > 0,
        'Host is IP': full.host_is_ip == 1}
R['presence_by_class_pct'] = {k: [round(100 * v[full.is_phishing == c].mean(), 2) for c in (0, 1)]
                              for k, v in pres.items()}
log("presence % [legit, phish]:", R['presence_by_class_pct'])

# --------------------------------------------------------------------------------------------------
# 2. Bootstrap CIs on the test set (paired resamples)
# --------------------------------------------------------------------------------------------------
log("\n== 2. Bootstrap (", N_BOOT, "resamples of the 47,074 test rows) ==")
rng = np.random.default_rng(SEED)
yt = y_test.values
n = len(yt)
boot = {k: [] for k in p_te}
boot_rf_tau = {'recall': [], 'fpr': [], 'precision': []}
boot_diff = {'RF-LR': [], 'RF-DT': [], 'LR-DT': []}
for b in range(N_BOOT):
    ii = rng.integers(0, n, n)
    yb = yt[ii]
    a = {k: auc(yb, v[ii]) for k, v in p_te.items()}
    for k in a: boot[k].append(a[k])
    boot_diff['RF-LR'].append(a['Random Forest'] - a['Logistic Regression'])
    boot_diff['RF-DT'].append(a['Random Forest'] - a['Decision Tree'])
    boot_diff['LR-DT'].append(a['Logistic Regression'] - a['Decision Tree'])
    pr = (p_te['Random Forest'][ii] >= TAU).astype(int)
    tp = ((pr == 1) & (yb == 1)).sum(); fn_ = ((pr == 0) & (yb == 1)).sum()
    fp = ((pr == 1) & (yb == 0)).sum(); tn = ((pr == 0) & (yb == 0)).sum()
    boot_rf_tau['recall'].append(tp / (tp + fn_)); boot_rf_tau['fpr'].append(fp / (fp + tn))
    boot_rf_tau['precision'].append(tp / (tp + fp))
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
R['bootstrap'] = dict(
    n_resamples=N_BOOT,
    roc_auc={k: dict(point=auc(y_test, p_te[k]), ci95=ci(v)) for k, v in boot.items()},
    auc_difference={k: dict(point=float(np.mean(v)), ci95=ci(v), frac_resamples_positive=float(np.mean(np.array(v) > 0)))
                    for k, v in boot_diff.items()},
    rf_at_tuned_tau={k: dict(ci95=ci(v)) for k, v in boot_rf_tau.items()})
for k, v in R['bootstrap']['roc_auc'].items(): log(f"  AUC {k}: {v['point']:.4f}  95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
for k, v in R['bootstrap']['auc_difference'].items(): log(f"  dAUC {k}: {v['point']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
for k, v in R['bootstrap']['rf_at_tuned_tau'].items(): log(f"  RF@tau {k}: 95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")

# --------------------------------------------------------------------------------------------------
# 3. Validation-only tuning
# --------------------------------------------------------------------------------------------------
log("\n== 3. Tuning on validation only ==")
tune = {}
best_lr = max(({'C': C} for C in [0.01, 0.1, 1, 10]),
              key=lambda g: auc(y_val, LogisticRegression(C=g['C'], max_iter=1000, random_state=SEED)
                                .fit(Xs['train'], y_train).predict_proba(Xs['val'])[:, 1]))
grid_dt = [dict(max_depth=d, min_samples_leaf=m) for d in [5, 10, 15, 20, None] for m in [1, 5, 20]]
dt_scores = [(auc(y_val, DecisionTreeClassifier(random_state=SEED, **g).fit(X_train, y_train).predict_proba(X_val)[:, 1]), g) for g in grid_dt]
best_dt = max(dt_scores, key=lambda t: t[0])
grid_rf = [dict(n_estimators=100, max_depth=None, min_samples_leaf=m) for m in [1, 5, 20]] + \
          [dict(n_estimators=100, max_depth=20, min_samples_leaf=5)]
rf_scores = [(auc(y_val, RandomForestClassifier(random_state=SEED, n_jobs=-1, **g).fit(X_train, y_train).predict_proba(X_val)[:, 1]), g) for g in grid_rf]
best_rf = max(rf_scores, key=lambda t: t[0])
log("LR best:", best_lr, "| DT best:", best_dt, "| RF best:", best_rf)
tuned_models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=SEED, **best_lr).fit(Xs['train'], y_train), Xs['val'], Xs['test']),
    'Decision Tree': (DecisionTreeClassifier(random_state=SEED, **best_dt[1]).fit(X_train, y_train), X_val, X_test),
    'Random Forest': (RandomForestClassifier(random_state=SEED, n_jobs=-1, **best_rf[1]).fit(X_train, y_train), X_val, X_test)}
R['tuning'] = {'grid_dt': [dict(val_auc=s, **g) for s, g in dt_scores],
               'grid_rf': [dict(val_auc=s, **g) for s, g in rf_scores],
               'selected': {'Logistic Regression': best_lr, 'Decision Tree': best_dt[1], 'Random Forest': best_rf[1]},
               'val_auc_selected': {'Decision Tree': best_dt[0], 'Random Forest': best_rf[0]},
               'test_roc_auc_after_tuning': {}}
for k, (m, xv, xt) in tuned_models.items():
    R['tuning']['test_roc_auc_after_tuning'][k] = auc(y_test, m.predict_proba(xt)[:, 1])
    log(f"  tuned {k}: test AUC {R['tuning']['test_roc_auc_after_tuning'][k]:.4f} (default {R['bootstrap']['roc_auc'][k]['point']:.4f})")

# --------------------------------------------------------------------------------------------------
# 4. Leakage diagnostics: overlap and domain-grouped split
# --------------------------------------------------------------------------------------------------
log("\n== 4. Overlap + domain-grouped split ==")
norm = (raw['URL'].str.lower().str.replace(r'^https?://', '', regex=True)
        .str.replace(r'^www\.', '', regex=True).str.rstrip('/'))
host = raw['Domain'].str.lower().str.replace(r'^www\.', '', regex=True)
tr_i, te_i = X_train.index, X_test.index
R['overlap_random_split'] = dict(
    test_urls_whose_normalised_url_in_train=float(norm.loc[te_i].isin(set(norm.loc[tr_i])).mean()),
    test_urls_whose_hostname_in_train=float(host.loc[te_i].isin(set(host.loc[tr_i])).mean()),
    test_phishing_hostname_in_train=float(host.loc[te_i][y_test.values == 1].isin(set(host.loc[tr_i])).mean()),
    test_legit_hostname_in_train=float(host.loc[te_i][y_test.values == 0].isin(set(host.loc[tr_i])).mean()),
    unique_hostnames=int(host.nunique()), rows=len(host),
    note='Hostnames = Domain column with leading "www." removed.')
log(R['overlap_random_split'])

groups = host.values
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
tmp_idx, te_g = next(gss1.split(X, y, groups))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
tr_rel, va_rel = next(gss2.split(X.iloc[tmp_idx], y.iloc[tmp_idx], groups[tmp_idx]))
tr_g, va_g = tmp_idx[tr_rel], tmp_idx[va_rel]
assert not (set(groups[tr_g]) & set(groups[te_g])), "group leak"
sc_g = StandardScaler().fit(X.iloc[tr_g])
lr_g = LogisticRegression(max_iter=1000, random_state=SEED).fit(sc_g.transform(X.iloc[tr_g]), y.iloc[tr_g])
dt_g = DecisionTreeClassifier(random_state=SEED).fit(X.iloc[tr_g], y.iloc[tr_g])
rf_g = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(X.iloc[tr_g], y.iloc[tr_g])
yg = y.iloc[te_g]
pg = {'Logistic Regression': lr_g.predict_proba(sc_g.transform(X.iloc[te_g]))[:, 1],
      'Decision Tree': dt_g.predict_proba(X.iloc[te_g])[:, 1],
      'Random Forest': rf_g.predict_proba(X.iloc[te_g])[:, 1]}
R['grouped_split'] = dict(
    group_key='hostname without leading www.', train=len(tr_g), val=len(va_g), test=len(te_g),
    test_phishing_share=float(yg.mean()), shared_hostnames_train_test=0,
    test_roc_auc={k: auc(yg, v) for k, v in pg.items()},
    rf_at_0p5=cm_metrics(yg, (pg['Random Forest'] >= 0.5).astype(int)))
log("grouped-split test AUC:", {k: round(v, 4) for k, v in R['grouped_split']['test_roc_auc'].items()},
    "| test phishing share", round(float(yg.mean()), 4))

# --------------------------------------------------------------------------------------------------
# 5. Benchmark-artifact ablation and hard subset
# --------------------------------------------------------------------------------------------------
log("\n== 5. Ablation of structure features ==")
STRUCT = ['uses_https', 'path_len_ratio', 'special_symbols']
keep = [f for f in FEATS if f not in STRUCT]
sc_k = StandardScaler().fit(X_train[keep])
lr_k = LogisticRegression(max_iter=1000, random_state=SEED).fit(sc_k.transform(X_train[keep]), y_train)
dt_k = DecisionTreeClassifier(random_state=SEED).fit(X_train[keep], y_train)
rf_k = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(X_train[keep], y_train)
R['ablation_drop_structure_features'] = dict(
    dropped=STRUCT, kept=keep,
    test_roc_auc={'Logistic Regression': auc(y_test, lr_k.predict_proba(sc_k.transform(X_test[keep]))[:, 1]),
                  'Decision Tree': auc(y_test, dt_k.predict_proba(X_test[keep])[:, 1]),
                  'Random Forest': auc(y_test, rf_k.predict_proba(X_test[keep])[:, 1])})
log(R['ablation_drop_structure_features']['test_roc_auc'])

bare = ((X_test['uses_https'] == 1) & (X_test['path_len_ratio'] == 0) & (X_test['special_symbols'] == 0)).values
p_rf = p_te['Random Forest']
R['hard_subset_bare_https_root'] = dict(
    definition='test rows with uses_https=1, no path, no special symbols',
    n=int(bare.sum()), legit=int((yt[bare] == 0).sum()), phishing=int((yt[bare] == 1).sum()),
    rf_roc_auc=auc(yt[bare], p_rf[bare]),
    rf_at_0p5=cm_metrics(yt[bare], (p_rf[bare] >= 0.5).astype(int)),
    rf_at_tuned_tau=cm_metrics(yt[bare], (p_rf[bare] >= TAU).astype(int)))
log("hard subset:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in R['hard_subset_bare_https_root'].items() if not isinstance(v, dict)})
log("  RF@0.5 on subset:", R['hard_subset_bare_https_root']['rf_at_0p5'])

# --------------------------------------------------------------------------------------------------
# 6. Extra lexical features (entropy / consonant run)
# --------------------------------------------------------------------------------------------------
log("\n== 6. Extra lexical features ==")
def shannon(s):
    if not s: return 0.0
    c = pd.Series(list(s)).value_counts(normalize=True).values
    return float(-(c * np.log2(c)).sum())
def cons_run(s):
    m = re.findall(r'[bcdfghjklmnpqrstvwxyz]+', s.lower()); return max((len(x) for x in m), default=0)
hostcore = raw['Domain'].str.lower().str.replace(r'^www\.', '', regex=True).str.split('.').str[0].fillna('')
ext = pd.DataFrame({'host_entropy': hostcore.map(shannon), 'host_consonant_run': hostcore.map(cons_run)})
X2 = pd.concat([X, ext], axis=1)
rf2 = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(X2.loc[X_train.index], y_train)
p2 = rf2.predict_proba(X2.loc[X_test.index])[:, 1]
R['extra_features'] = dict(added=['host_entropy (Shannon entropy of the first hostname label)',
                                  'host_consonant_run (longest consonant run in it)'],
                           rf_test_roc_auc_13=auc(y_test, p_rf), rf_test_roc_auc_15=auc(y_test, p2),
                           rf15_at_0p5=cm_metrics(y_test, (p2 >= 0.5).astype(int)),
                           rf13_at_0p5=cm_metrics(y_test, (p_rf >= 0.5).astype(int)))
log(R['extra_features']['rf_test_roc_auc_13'], '->', R['extra_features']['rf_test_roc_auc_15'])

# --------------------------------------------------------------------------------------------------
# 7. Error analysis + calibration
# --------------------------------------------------------------------------------------------------
log("\n== 7. Error analysis ==")
import tldextract
ex_pr = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
ex_ic = tldextract.TLDExtract(suffix_list_urls=())
def platform(h):
    a, b = ex_pr(h), ex_ic(h)
    return a.suffix if a.suffix != b.suffix else ''
test_host = raw.loc[X_test.index, 'Domain'].str.lower().values
plat = np.array([platform(h) for h in test_host])
shared = plat != ''
ev = pd.DataFrame({'y': yt, 'p': p_rf, 'plat': plat, 'shared': shared,
                   'https': X_test['uses_https'].values, 'path': (X_test['path_len_ratio'].values > 0),
                   'tld': raw.loc[X_test.index, 'TLD'].values, 'hlen': X_test['hostname_length'].values,
                   'hyph': X_test['hyphen_count'].values, 'dig': X_test['digit_count'].values,
                   'url': raw.loc[X_test.index, 'URL'].values})
def err_block(tau):
    pr = (ev.p >= tau).astype(int)
    fn = ev[(ev.y == 1) & (pr == 0)]; fp = ev[(ev.y == 0) & (pr == 1)]
    ph = ev[ev.y == 1]
    out = dict(tau=tau, n_fn=len(fn), n_fp=len(fp),
               fn_on_shared_hosting=int(fn.shared.sum()),
               phishing_on_shared_hosting_total=int(ph.shared.sum()),
               fn_rate_shared_hosting=float(fn.shared.sum() / max(1, ph.shared.sum())),
               fn_rate_not_shared=float((~fn.shared).sum() / max(1, (~ph.shared).sum())),
               fn_with_https_and_no_path=int(((fn.https == 1) & (~fn.path)).sum()),
               phishing_https_no_path_total=int(((ph.https == 1) & (~ph.path)).sum()),
               top_fn_platforms=fn[fn.shared].plat.value_counts().head(8).to_dict(),
               top_platforms_in_test_phishing=ph[ph.shared].plat.value_counts().head(8).to_dict())
    lg = ev[ev.y == 0]
    out['fpr_by_hostname_length'] = {lab: dict(n=int(len(g)), fp=int(((g.p >= tau)).sum()), rate=float((g.p >= tau).mean()))
        for lab, g in [('<=15', lg[lg.hlen <= 15]), ('16-25', lg[(lg.hlen > 15) & (lg.hlen <= 25)]), ('>25', lg[lg.hlen > 25])]}
    out['fpr_by_hyphen'] = {lab: dict(n=int(len(g)), fp=int((g.p >= tau).sum()), rate=float((g.p >= tau).mean()))
        for lab, g in [('no hyphen', lg[lg.hyph == 0]), ('>=1 hyphen', lg[lg.hyph >= 1])]}
    out['fpr_by_digits'] = {lab: dict(n=int(len(g)), fp=int((g.p >= tau).sum()), rate=float((g.p >= tau).mean()))
        for lab, g in [('no digit', lg[lg.dig == 0]), ('>=1 digit', lg[lg.dig >= 1])]}
    topt = lg.tld.value_counts().head(6).index
    out['fpr_by_tld_top6_legit'] = {t: dict(n=int((lg.tld == t).sum()), fp=int(((lg.tld == t) & (lg.p >= tau)).sum()),
                                            rate=float(((lg.tld == t) & (lg.p >= tau)).sum() / (lg.tld == t).sum())) for t in topt}
    out['first5_fp_in_test_order'] = fp.head(5)[['url', 'p']].assign(p=lambda d: d.p.round(4)).to_dict('records')
    out['first5_fn_in_test_order'] = fn.head(5)[['url', 'p']].assign(p=lambda d: d.p.round(4)).to_dict('records')
    return out
R['errors_tuned_tau'] = err_block(TAU)
R['errors_tau_0p5'] = err_block(0.5)
R['shared_hosting_in_test'] = dict(phishing=int(ev[(ev.y == 1)].shared.sum()), legit=int(ev[(ev.y == 0)].shared.sum()))
for k in ('errors_tuned_tau', 'errors_tau_0p5'):
    e = R[k]; log(k, {a: e[a] for a in ('tau', 'n_fn', 'n_fp', 'fn_on_shared_hosting', 'phishing_on_shared_hosting_total',
                                         'fn_rate_shared_hosting', 'fn_rate_not_shared', 'fn_with_https_and_no_path',
                                         'phishing_https_no_path_total', 'top_fn_platforms')})
log("shared hosting in test:", R['shared_hosting_in_test'])
log("FPR by host length @tau:", R['errors_tuned_tau']['fpr_by_hostname_length'])
log("FPR by hyphen @tau:", R['errors_tuned_tau']['fpr_by_hyphen'])

bins = np.linspace(0, 1, 11)
cal = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (p_rf >= lo) & ((p_rf < hi) if hi < 1 else (p_rf <= hi))
    if m.sum(): cal.append(dict(bin=f"{lo:.1f}-{hi:.1f}", n=int(m.sum()), mean_pred=float(p_rf[m].mean()), frac_phishing=float(yt[m].mean())))
ece = sum(c['n'] * abs(c['mean_pred'] - c['frac_phishing']) for c in cal) / len(yt)
R['calibration_rf'] = dict(bins=cal, ece=float(ece), brier=float(brier_score_loss(yt, p_rf)),
                           note='Calibration is measured on the benchmark test split (about 43% phishing). It says nothing about calibration under a different phishing prevalence.')
log("calibration ECE", round(ece, 4), "Brier", round(R['calibration_rf']['brier'], 4))
for c in cal: log("  ", c)

# prevalence-adjusted precision at the two operating points (Bayes, using measured TPR/FPR)
def ppv(tpr, fpr, pi): return tpr * pi / (tpr * pi + fpr * (1 - pi))
ops = {}
for name, tau in [('tuned', TAU), ('tau_0.90', 0.90)]:
    m = cm_metrics(yt, (p_rf >= tau).astype(int))
    ops[name] = dict(tau=tau, tpr=m['recall'], fpr=m['fpr'], ppv_pi_1pct=ppv(m['recall'], m['fpr'], 0.01),
                     ppv_pi_0p1pct=ppv(m['recall'], m['fpr'], 0.001))
R['prevalence_adjusted_ppv'] = ops
log(ops)

json.dump(R, open("results/final_results.json", "w"), indent=2, default=float)
log("\nWrote results/final_results.json")
