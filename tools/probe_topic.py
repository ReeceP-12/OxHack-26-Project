"""Does a topic you talked about a lot, but that is not 'important', survive? And does it come back when you return to it?

    python tools/probe_topic.py                       # real MiniLM, uses your MEMORY_* settings (e.g. MEMORY_SCALE=arctan)
    python tools/probe_topic.py --fake                # no model download, word-overlap stand-in (rough)
    python tools/probe_topic.py --gap 60 --prompts 100

Scenario: 10 turns of other chat, then 5 questions about black holes, then GAP turns of unrelated chat, then either an
unrelated question or a question that returns to black holes. Prints the tier of each black-hole question both times.
Nothing is deleted for good: the voice agent resends the whole chat every turn and every turn is re-scored, so an
archived message can come back as soon as the current question is similar to it.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.accuracy import TOPICS, PADS, REPLY, SYSTEM

BH = ["How do black holes form when a massive star collapses?",
      "What happens to light that crosses the event horizon of a black hole?",
      "Can black holes evaporate through Hawking radiation?",
      "How big is the supermassive black hole at the centre of our galaxy?",
      "What would happen to a person falling into a black hole?"]


def chat(i):
    return [{"role": "user", "content": f"Tell me about {TOPICS[i * 7 % len(TOPICS)]}, number {i}. {PADS[i % len(PADS)]}"},
            {"role": "assistant", "content": REPLY.format(t=TOPICS[i * 5 % len(TOPICS)].capitalize())}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--gap", type=int, default=40, help="unrelated turns between the black-hole questions and the test question")
    ap.add_argument("--budget", type=int, default=int(os.environ.get("MEMORY_BUDGET", "1500")))
    a = ap.parse_args()
    from memory.context_manager import ContextManager
    if a.fake:
        from tests.test_memory import FakeEmbedder
        cm = ContextManager(FakeEmbedder(), fact_clf=None)
    else:
        cm = ContextManager()
    print(f"scaling mode: {os.environ.get('MEMORY_SCALE', '(default)')}, base budget {a.budget}, gap {a.gap} turns\n")
    for ask, label in (("Tell me something about volcanoes.", "unrelated question"),
                       ("Going back to black holes, what exactly is the event horizon?", "back on black holes")):
        m = [{"role": "system", "content": SYSTEM}]
        for i in range(10):
            m += chat(i)
        first = len(m)
        for q in BH:
            m += [{"role": "user", "content": q}, {"role": "assistant", "content": REPLY.format(t="Black holes")}]
        for i in range(a.gap):
            m += chat(100 + i)
        m.append({"role": "user", "content": ask})
        _, tiers = cm.build(m, a.budget)
        bh = [tiers[first + 2 * j] for j in range(5)]
        print(f"{label:22s} black-hole questions -> " + ", ".join(f"{t['tier']} ({t['score']:.2f})" for t in bh))


if __name__ == "__main__":
    main()
