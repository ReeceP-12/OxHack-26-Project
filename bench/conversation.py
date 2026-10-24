"""Scripted 50-turn conversation with facts planted early and recall tested at the end."""

# turn index (0-based) -> planted fact message
PLANTED = {
    1: "Quick heads up, I'm severely allergic to peanuts, so keep that in mind for anything food related.",
    4: "My project deadline is Friday the 14th, please remember that.",
    7: "My dog is called Biscuit and he's a three year old beagle.",
    11: "My budget for the trip is 1200 pounds in total.",
    15: "My flight is BA 249 and it leaves at 6:40am.",
}

# Chatter that should be safe to shrink or forget
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
]

N_TURNS = 50

def build_turns():
    turns = []
    c = 0
    for i in range(N_TURNS):
        if i in PLANTED:
            turns.append(PLANTED[i])
        else:
            turns.append(CHATTER[c % len(CHATTER)])
            c += 1
    return turns

# (question, keywords: ANY of these in the reply counts as correct)
RECALL = [
    ("What am I allergic to?", ["peanut"]),
    ("What deadline did I give you?", ["friday", "14"]),
    ("What's my dog's name?", ["biscuit"]),
    ("What's my total trip budget?", ["1200", "1,200"]),
    ("What is my flight number?", ["ba 249", "ba249", "249"]),
]
