"""Run with:  python tests/test_fact_classifier.py   (or pytest). No model download needed."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from test_memory import FakeEmbedder
from memory.fact_data import train_examples, heldout_examples, heldout2_examples, TEST_POS, TEST_NEG, TEST2_POS, TEST2_NEG
from memory.fact_classifier import FactClassifier, fit_logistic, split_sentences
from memory.scorer import score_messages, importance, STRONG_IMPORTANCE
from memory.context_manager import ContextManager
from memory.train_fact_classifier import train
import subprocess


class StubClf:
    """Pretends to be a trained classifier: anything mentioning a zebra is a 'fact'."""
    threshold = 0.4
    def probs(self, texts, embedder):
        return [0.9 if "zebra" in t.lower() else 0.05 for t in texts]


def test_dataset_is_clean_and_balanced():
    tr = train_examples(); te = heldout_examples()
    assert len(tr) > 400 and len(te) > 100
    texts = [t for t, _ in tr]
    assert len(texts) == len(set(texts)), "duplicate training sentences"
    assert not set(texts) & {t for t, _ in te}, "test sentences leaked into training"
    b = heldout2_examples()
    assert len(b) > 60
    assert not set(texts) & {t for t, _ in b}, "set B leaked into training"
    assert not {t for t, _ in te} & {t for t, _ in b}, "set A and set B overlap"
    assert set(TEST2_POS).isdisjoint(TEST2_NEG)
    frac_pos = sum(l for _, l in tr) / len(tr)
    assert 0.35 < frac_pos < 0.65, frac_pos
    assert set(TEST_POS).isdisjoint(TEST_NEG)


def test_fit_logistic_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 20)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = (X @ rng.normal(size=20) > 0).astype(int)
    w, b = fit_logistic(X, y, l2=0.01)
    assert (((X @ w + b) > 0) == y).mean() > 0.95


def test_save_load_roundtrip_and_missing_file():
    clf = FactClassifier(np.arange(5, dtype=np.float32), 0.25, 0.37, "some-model")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.npz")
        clf.save(path)
        back = FactClassifier.load(path)
        assert np.allclose(back.w, clf.w) and back.b == 0.25 and abs(back.threshold - 0.37) < 1e-6
        assert back.model_name == "some-model"
        assert FactClassifier.load(os.path.join(d, "nope.npz")) is None


def test_split_sentences():
    out = split_sentences("Okay thanks. By the way I\u2019m allergic to peanuts! Anyway, what's for dinner?")
    assert len(out) == 3 and "I'm allergic to peanuts!" in out[1]
    assert split_sentences("hello") == ["hello"]


def test_classifier_pins_a_fact_the_regex_misses():
    text = "We keep a zebra in the garden."
    assert importance(text) < 0.8                      # regex alone does not see it
    m = [{"role": "system", "content": "x"}, {"role": "user", "content": text},
         {"role": "assistant", "content": "Wow."}]
    s_off = score_messages(m, FakeEmbedder(), None)[1]["importance"]
    s_on = score_messages(m, FakeEmbedder(), StubClf())[1]["importance"]
    assert s_off < 0.8 and s_on == STRONG_IMPORTANCE


def test_classifier_never_lowers_regex_importance_and_ignores_assistant():
    m = [{"role": "system", "content": "x"},
         {"role": "user", "content": "I'm severely allergic to peanuts."},      # regex pins, stub says 0.05
         {"role": "assistant", "content": "A zebra, how nice."}]                # stub says 0.9, but assistant
    s = score_messages(m, FakeEmbedder(), StubClf())
    assert s[1]["importance"] >= 0.9
    assert s[2]["importance"] < 0.8


def test_fact_survives_in_a_long_conversation_via_classifier():
    m = [{"role": "system", "content": "You are a helpful voice assistant."},
         {"role": "user", "content": "We keep a zebra in the garden."},
         {"role": "assistant", "content": "Wow, that is unusual."}]
    for i in range(40):
        m += [{"role": "user", "content": f"Tell me about topic {i}. " + "Please include plenty of extra detail. " * 8},
              {"role": "assistant", "content": f"Here is a long answer about topic {i}. " + "More detail follows here. " * 10}]
    m.append({"role": "user", "content": "Which animals should I feed tonight?"})
    cm = ContextManager(FakeEmbedder(), fact_clf=StubClf())
    _, tiers = cm.build(m, budget=1500)
    assert tiers[1]["tier"] in ("full", "summary"), tiers[1]
    _, tiers_off = ContextManager(FakeEmbedder(), fact_clf=None).build(m, budget=1500)
    assert tiers_off[1]["tier"] == "archived", "control: without the classifier this fact is lost"


def test_auto_mode_ignores_a_classifier_trained_for_another_embedder():
    # FakeEmbedder has no model_name, so a saved MiniLM classifier must not be applied to it.
    cm = ContextManager(FakeEmbedder())               # fact_clf="auto"
    assert cm.fact_clf is None


def test_training_pipeline_runs_end_to_end():
    clf, report = train(FakeEmbedder(), verbose=False)
    assert report["train_n"] > 400 and report["test_n"] > 100 and report["test2_n"] > 60
    assert clf.w.shape == (64,)
    p = clf.probs(["I'm allergic to peanuts.", "What's the capital of France?"], FakeEmbedder())
    assert all(0.0 <= x <= 1.0 for x in p)



class _NamedFake(FakeEmbedder):
    model_name = "fake-model"


def _with_env(env, fn):
    saved = {k: os.environ.get(k) for k in env}
    orig = FactClassifier.load
    try:
        for k, v in env.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
        FactClassifier.load = classmethod(lambda cls, path=None: FactClassifier(np.zeros(64), 0.0, 0.6, "fake-model"))
        return fn()
    finally:
        FactClassifier.load = orig
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def test_env_threshold_override_and_off_switch():
    assert _with_env({"FACT_CLF": None, "FACT_THRESHOLD": None}, lambda: ContextManager(_NamedFake()).fact_clf.threshold) == 0.6
    assert _with_env({"FACT_CLF": None, "FACT_THRESHOLD": "0.45"}, lambda: ContextManager(_NamedFake()).fact_clf.threshold) == 0.45
    assert _with_env({"FACT_CLF": None, "FACT_THRESHOLD": "banana"}, lambda: ContextManager(_NamedFake()).fact_clf.threshold) == 0.6
    assert _with_env({"FACT_CLF": "off", "FACT_THRESHOLD": None}, lambda: ContextManager(_NamedFake()).fact_clf) is None


def test_strong_regex_can_be_switched_off_for_ablation():
    code = ("import sys; sys.path.insert(0, %r); from memory.scorer import importance; "
            "print(importance(\"I'm severely allergic to peanuts.\"))") % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    on = float(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env={**os.environ, "FACT_STRONG_REGEX": ""}).stdout)
    off = float(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env={**os.environ, "FACT_STRONG_REGEX": "0"}).stdout)
    assert on >= 0.9 and off < 0.8, (on, off)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)
