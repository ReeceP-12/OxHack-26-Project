"""Scripted 50-turn conversation with facts planted early and recall tested at the end."""

# turn index (0-based) -> planted fact message
PLANTED = {
    1: "Quick heads up, I'm severely allergic to peanuts, so keep that in mind for anything food related.",
    4: "My project deadline is Friday the 14th, please remember that.",
    7: "My dog is called Biscuit and he's a three year old beagle.",
    11: "My budget for the trip is 1200 pounds in total.",
    15: "My flight is BA 249 and it leaves at 6:40am.",
    30: "Update on my deadline: it has moved from Friday to Monday the 17th, so please use Monday from now on.",
    40: "Just confirming, my flight is still BA 249 and it still leaves at 6:40am.",
}

# Distinct filler lines (50 turns minus the planted ones), so nothing repeats.
CHATTER = [
    "What's a good way to learn guitar chords quickly?",
    "Tell me something interesting about octopuses.",
    "I had a really long day, the train was late again.",
    "How do I make a decent cup of coffee without a machine?",
    "Do you think it will rain tomorrow?",
    "Can you recommend a podcast about history?",
    "I'm trying to decide what to watch tonight.",
    "What's the difference between a crocodile and an alligator?",
    "My laptop fan is really loud lately, any ideas?",
    "Explain how tides work in simple terms.",
    "What's a fun fact about the moon?",
    "I think I'll go for a run later.",
    "How do bees make honey?",
    "Suggest a short stretching routine.",
    "Why is the sky blue?",
    "What's the tallest mountain in Europe?",
    "How do vaccines train the immune system?",
    "Recommend a board game for four people.",
    "What causes the northern lights?",
    "How long does it take to learn to juggle?",
    "Why do cats purr?",
    "Give me a quick tip for better sleep.",
    "What's the origin of the word salary?",
    "How does a refrigerator keep things cold?",
    "Tell me about the history of the bicycle.",
    "What's the best way to remember names?",
    "How do airplanes stay in the air?",
    "What's the deepest part of the ocean?",
    "Suggest a hobby I can start with no equipment.",
    "How do I stop my glasses fogging up?",
    "What's the difference between weather and climate?",
    "Why do onions make people cry?",
    "Explain what an eclipse is.",
    "How do I get better at mental arithmetic?",
    "What's a good way to spend a rainy Sunday?",
    "How do whales communicate?",
    "Tell me a fun fact about volcanoes.",
    "What makes a good cup of tea?",
    "How does a compass work?",
    "Why is the ocean salty?",
    "Suggest a simple pasta sauce idea.",
    "How do birds know where to migrate?",
    "What's the story behind the Eiffel Tower?",
    "How do I keep a houseplant alive?",
    "What's the speed of sound?",
]

N_TURNS = 50
assert len(CHATTER) >= N_TURNS - len(PLANTED), "need more distinct filler lines"

def build_turns():
    turns = []
    c = 0
    for i in range(N_TURNS):
        if i in PLANTED:
            turns.append(PLANTED[i])
        else:
            turns.append(CHATTER[c])
            c += 1
    return turns

# (question, keywords: ANY of these in the reply counts as correct)
# Note: keywords are checked against the reply text only. A reply that mentions both the
# old and the new deadline ("moved from Friday to Monday") still passes on "monday".
RECALL = [
    ("What am I allergic to?", ["peanut"]),
    # The deadline was CHANGED at turn 30, so the correct answer is the NEW one (Monday).
    ("What is my current deadline?", ["monday"]),
    ("What's my dog's name?", ["biscuit"]),
    ("What's my total trip budget?", ["1200", "1,200"]),
    ("What is my flight number?", ["ba 249", "ba249", "ba-249"]),
]
