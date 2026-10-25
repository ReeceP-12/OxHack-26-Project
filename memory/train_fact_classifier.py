"""Train the fact classifier and compare it with the regex rules on held-out sentences.

    python -m memory.train_fact_classifier
    python -m memory.train_fact_classifier --threshold 0.5 --l2 0.1

Needs the real MiniLM (sentence-transformers). Writes memory/fact_clf.npz, which you commit so
teammates do not have to train. Takes a few seconds on a laptop CPU.

Two held-out sets are scored. SET A guided some of the training data (errors from the first report
were turned into extra training sentences), so treat it as partly used up. SET B was written after
that and nothing has been tuned on it: it is the honest check.

Read FP rate and prec@10%. A real conversation is mostly chatter, not facts, so the share of
chatter that gets pinned matters more than the precision on a 50/50 test set.
"""
import argparse
import numpy as np

from memory.fact_data import train_examples, heldout_examples, heldout2_examples
from memory.fact_classifier import FactClassifier, fit_logistic, DEFAULT_PATH
from memory.scorer import Embedder, importance, PIN_IMPORTANCE

FACT_SHARE = 0.10   # assumed share of real messages that are facts, for the prec@10% column


def _prf(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    n_neg = int((~y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    denom = r * FACT_SHARE + fpr * (1 - FACT_SHARE)
    p10 = (r * FACT_SHARE) / denom if denom else 0.0
    return dict(p=p, r=r, f=f, fn=fn, fp=fp, fpr=fpr, p10=p10)


def _row(label, m, note=""):
    return (f"{label:<27}{m['p']:>6.2f}{m['r']:>8.2f}{m['f']:>6.2f}{m['fpr']:>9.2f}{m['p10']:>9.2f}"
            f"   {m['fn']:>2} / {m['fp']:<2}{note}")


def evaluate(name, examples, clf, embedder, threshold):
    texts = [t for t, _ in examples]
    y = np.array([l for _, l in examples])
    p_clf = np.array(clf.probs(texts, embedder))
    regex = np.array([importance(t, "user") >= PIN_IMPORTANCE for t in texts])
    print(f"\n=== {name}: {len(texts)} sentences, {int(y.sum())} facts ===")
    print(f"{'method':<27}{'prec':>6}{'recall':>8}{'F1':>6}{'FP rate':>9}{'prec@10%':>9}   missed / false pins")
    print(_row("regex rules only", _prf(regex, y)))
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        print(_row(f"classifier @ {th:.1f}", _prf(p_clf >= th, y), "  <- saved" if abs(th - threshold) < 1e-9 else ""))
    used = (p_clf >= threshold) | regex
    print(_row("classifier OR regex (used)", _prf(used, y)))
    missed = [(p_clf[i], t) for i, (t, lab, u) in enumerate(zip(texts, y, used)) if lab and not u]
    false_pins = [(p_clf[i], t) for i, (t, lab, u) in enumerate(zip(texts, y, used)) if not lab and u]
    print("missed facts (the costly mistake):" + ("  none" if not missed else ""))
    for pr, t in missed:
        print(f"  p={pr:.2f}  {t}")
    print("false pins:" + ("  none" if not false_pins else ""))
    for pr, t in false_pins:
        print(f"  p={pr:.2f}  {t}")
    return _prf(used, y)


def train(embedder, l2=0.1, pos_weight=1.5, threshold=0.4, verbose=True):
    tr, a, b = train_examples(), heldout_examples(), heldout2_examples()
    Xtr = embedder.embed_many([t for t, _ in tr])
    ytr = np.array([l for _, l in tr])
    w, bias = fit_logistic(Xtr, ytr, l2=l2, pos_weight=pos_weight)
    clf = FactClassifier(w, bias, threshold, getattr(embedder, "model_name", "unknown"))
    report = {"train_n": len(tr), "test_n": len(a), "test2_n": len(b)}
    if verbose:
        print(f"train: {len(tr)} sentences ({int(ytr.sum())} facts) | threshold {threshold}")
        print(f"prec@10% = precision if only {int(FACT_SHARE * 100)}% of real messages were facts")
        evaluate("SET A (partly used to guide data)", a, clf, embedder, threshold)
        evaluate("SET B (fresh, untouched)", b, clf, embedder, threshold)
    return clf, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", type=float, default=0.1, help="regularisation strength (higher = smoother)")
    ap.add_argument("--pos-weight", type=float, default=1.5, help="how much worse a missed fact is than a false pin")
    ap.add_argument("--threshold", type=float, default=0.4, help="lower = catches more facts, pins more chatter")
    ap.add_argument("--out", default=DEFAULT_PATH)
    a = ap.parse_args()
    clf, _ = train(Embedder(), a.l2, a.pos_weight, a.threshold)
    clf.save(a.out)
    print(f"\nsaved {a.out}")
