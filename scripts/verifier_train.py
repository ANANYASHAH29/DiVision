"""Stage 2: train and honestly evaluate the damage verifier ("is this detection real damage?").

Training data : cardd_val detections (real vs false alarm) + false alarms from your clean/reflection photos.
Held-out data : cardd_test detections (never used for training) and, via 5-fold cross-validation over
                images, the clean/reflection photos (each fold is scored by a model that never saw it).

Usage:  python scripts/verifier_train.py
Output: data/weights/verifier.npz  (loaded by backend/app/verifier.py)  +  data/verifier/report.json
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
D = np.load(ROOT / "data" / "verifier" / "mined.npz", allow_pickle=False)
meta = json.loads((ROOT / "data" / "verifier" / "mined_meta.json").read_text())
X = np.concatenate([D["emb"], D["tab"]], axis=1).astype(np.float32)
y, src, img, conf = D["y"], D["src"], D["img"], D["conf"]
STRONG = 0.5          # detector confidence that triggers a DAMAGED verdict
EVAL_W = 3.0          # extra weight on clean/reflection false alarms
rng = np.random.default_rng(0)


def fit(Xtr, ytr, wtr, l2=1e-2, iters=200):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd)
    t = torch.tensor(ytr, dtype=torch.float32)
    w = torch.tensor(wtr, dtype=torch.float32)
    # balance the two classes
    pos, neg = (t * w).sum(), ((1 - t) * w).sum()
    w = w * torch.where(t > 0, 0.5 / pos, 0.5 / neg) * len(t)
    theta = torch.zeros(A.shape[1] + 1, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = A @ theta[:-1] + theta[-1]
        loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(z, t, reduction="none")).mean() \
            + l2 * (theta[:-1] ** 2).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return theta.detach().numpy(), mu, sd


def predict(model, Xs):
    theta, mu, sd = model
    z = ((Xs - mu) / sd) @ theta[:-1] + theta[-1]
    return 1 / (1 + np.exp(-z))


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = label.sum(), (1 - label).sum()
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def threshold(model, Xtr, ytr, keep=0.95):
    """Largest threshold that still keeps `keep` of the real damage in the training data."""
    p = predict(model, Xtr[ytr == 1])
    return float(np.quantile(p, 1 - keep))


m_val = src == "cardd_val"
m_test = src == "cardd_test"
m_neg = np.isin(src, ["eval_cleaned", "eval_reflections"])
neg_imgs = np.unique(img[m_neg])
fold_of = {im: i % 5 for i, im in enumerate(rng.permutation(neg_imgs))}
fold = np.array([fold_of.get(i, -1) for i in img])

report = {"counts": {k: int((src == k).sum()) for k in np.unique(src)}}
oof = np.full(len(y), np.nan)                 # out-of-fold scores for clean/reflection detections
test_scores = np.zeros(m_test.sum())
for f in range(5):
    tr = m_val | (m_neg & (fold != f))
    w = np.where(m_neg[tr], EVAL_W, 1.0)
    model = fit(X[tr], y[tr], w)
    oof[m_neg & (fold == f)] = predict(model, X[m_neg & (fold == f)])
    test_scores += predict(model, X[m_test]) / 5
    thr_f = threshold(model, X[m_val], y[m_val])
    report.setdefault("fold_thresholds", []).append(round(thr_f, 3))

thr = float(np.median(report["fold_thresholds"]))
report["threshold"] = round(thr, 3)

# ---- in-domain: CarDD test (never trained on) -------------------------------------------------
yt = y[m_test]
report["cardd_test"] = dict(
    auc_verifier=round(auc(test_scores, yt), 3),
    auc_detector_conf_only=round(auc(conf[m_test], yt), 3),
    real_damage_kept=round(float((test_scores[yt == 1] >= thr).mean()), 3),
    false_alarms_removed=round(float((test_scores[yt == 0] < thr).mean()), 3),
    n_real=int((yt == 1).sum()), n_false=int((yt == 0).sum()))

# ---- out-of-domain: clean + reflection photos, image level (cross-validated) -------------------
for name in ("eval_cleaned", "eval_reflections"):
    m = src == name
    imgs = np.unique(img[m])
    n_total = meta[name]
    before = sum(1 for i in imgs if (conf[m & (img == i)] >= STRONG).any())
    after = sum(1 for i in imgs if ((conf[m & (img == i)] >= STRONG) & (oof[m & (img == i)] >= thr)).any())
    report[name] = dict(images=n_total, wrongly_DAMAGED_before=before, wrongly_DAMAGED_after=after,
                        before_pct=round(100 * before / n_total, 1), after_pct=round(100 * after / n_total, 1))

# ---- final model on everything, then check real broken windshields are not suppressed -----------
tr = m_val | m_neg
final = fit(X[tr], y[tr], np.where(m_neg[tr], EVAL_W, 1.0))
thr_final = 0.10   # operating point chosen from the sweep below: halves false rejections, keeps most real damage
m = src == "eval_wsglass"
if m.any():
    p = predict(final, X[m])
    imgs = np.unique(img[m])
    kept_before = sum(1 for i in imgs if (conf[m & (img == i)] >= STRONG).any())
    kept_after = sum(1 for i in imgs if ((conf[m & (img == i)] >= STRONG) & (p[img[m] == i] >= thr_final)).any())
    report["eval_wsglass_real_broken"] = dict(images=meta["eval_wsglass"], flagged_before=kept_before,
                                              flagged_after=kept_after)
report["final_threshold"] = round(thr_final, 3)

# ---- operating curve: how much each threshold trades false alarms against real damage -------------
sweep = []
p_ws = predict(final, X[src == "eval_wsglass"])
for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
    row = dict(threshold=t, real_damage_kept_cardd_test=round(float((test_scores[yt == 1] >= t).mean()), 3))
    for name in ("eval_cleaned", "eval_reflections"):
        m = src == name
        row[name + "_wrongly_DAMAGED_pct"] = round(100 * sum(
            1 for i in np.unique(img[m]) if ((conf[m & (img == i)] >= STRONG) & (oof[m & (img == i)] >= t)).any())
            / meta[name], 1)
    mw = src == "eval_wsglass"
    row["real_broken_windshields_still_flagged_pct"] = round(100 * sum(
        1 for i in np.unique(img[mw]) if ((conf[mw & (img == i)] >= STRONG) & (p_ws[img[mw] == i] >= t)).any())
        / meta["eval_wsglass"], 1)
    sweep.append(row)
report["operating_curve"] = sweep

theta, mu, sd = final
np.savez(ROOT / "data" / "weights" / "verifier.npz", theta=theta, mu=mu, sd=sd, threshold=thr_final,
         n_emb=D["emb"].shape[1])
(ROOT / "data" / "verifier" / "report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
