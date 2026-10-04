"""Evaluate face-geometry age features from ml/face_features.py on FairFace validation.

Usage: python ml/face_eval.py [strict|loose]
strict: |yaw|, |pitch| < 15 deg and neutral expression (closest to a webcam check-in); loose: < 25 deg, any expression.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

d = np.load(Path(__file__).resolve().parent.parent / 'data/face-fairface.npz')
mode = sys.argv[1] if len(sys.argv) > 1 else 'strict'
lim = 15 if mode == 'strict' else 25
keep = (np.abs(d['yaw']) < lim) & (np.abs(d['pitch']) < lim)
if mode == 'strict':
    keep &= (d['smile'] < 0.3) & (d['jaw'] < 0.2)
print(f'== {mode}: |yaw|,|pitch| < {lim} deg' + (', neutral expression' if mode == 'strict' else '') + f'; kept {keep.sum()}/{len(keep)}')
F, L, band, split, names = d['F'][keep], d['L'][keep], d['band'][keep], d['split'][keep], d['names']
LAB = {3: '20s', 4: '30s', 5: '40s', 6: '50s', 7: '60s', 8: '70+'}
y = 25 + 10 * (band - 3.0)  # band midpoint in years; 70+ -> 75
tr, te = split == 'train', split == 'validation'
w = 1 / np.bincount(band)[band]; w = w / w[tr].mean()
print(f'train {tr.sum()}  test {te.sum()}  test per band:', {LAB[b]: int((band[te] == b).sum()) for b in LAB})

print('\nPer-feature mean by band (test+train), expected direction from table:')
exp = ['up', 'down', 'down', 'down', 'down', 'up', 'up']
print(f'{"feature":20s}' + ''.join(f'{LAB[b]:>8s}' for b in LAB) + '   rho   expected')
for i, n in enumerate(names):
    rho = spearmanr(F[:, i], y).statistic
    print(f'{n:20s}' + ''.join(f'{F[band == b, i].mean():8.3f}' for b in LAB) + f'  {rho:+.2f}   {exp[i]}')


def show(name, p):
    e = np.abs(p - y[te])
    print(f'\n{name}: MAE {e.mean():.1f} y vs midpoint, spearman {spearmanr(p, y[te]).statistic:.2f}, '
          f'within one band {np.mean(e <= 10):.0%}')
    print('  ' + '  '.join(f'{LAB[b]} {p[band[te] == b].mean():.0f}' for b in LAB))



# Decade-balanced weights pull young predictions up; on this young-heavy test set, Spearman is the fairer metric.
base = np.full(te.sum(), np.average(y[tr], weights=w[tr]))
show('baseline (predict weighted mean)', base)
g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
show('7 table ratios -> gradient boosting', g.fit(F[tr], y[tr], sample_weight=w[tr]).predict(F[te]))
m = make_pipeline(StandardScaler(), PCA(128, random_state=0), SVR(C=10))
show('all 478 landmarks (geometry ceiling) -> SVR', m.fit(L[tr], y[tr], svr__sample_weight=w[tr]).predict(L[te]))
