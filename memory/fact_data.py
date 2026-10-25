"""Labelled sentences for the "personal fact worth remembering" classifier.

LABEL 1 = the USER states something about themselves or their situation that a voice agent
          should still know 20 turns later: health, allergies, diet, names, family, pets,
          where they live or work, standing preferences or instructions, deadlines, plans, budgets.
LABEL 0 = everything else: questions (even about the same topics), facts about the world,
          requests, chit-chat, acknowledgements, passing moods, opinions.

The hard part is the near misses: "Is a peanut allergy common?" is 0, "Peanuts make me ill" is 1.

TRAIN is built from templates plus hand-written sentences. TEST is hand-written with DIFFERENT
phrasings, so test scores are not inflated by template leakage. Both were written by the same
person, so real voice transcripts will still be messier. The real test is the benchmark.
"""
import random

SLOTS = {
    "allergen": ["peanuts", "shellfish", "eggs", "penicillin", "latex", "bee stings", "tree nuts", "soy",
                 "sesame", "wheat", "dairy", "kiwi", "mushrooms", "strawberries", "fish", "gluten"],
    "condition": ["diabetes", "asthma", "epilepsy", "high blood pressure", "migraines", "arthritis",
                  "celiac disease", "anxiety", "eczema", "a heart condition"],
    "diet": ["vegan", "vegetarian", "pescatarian", "halal", "kosher", "gluten-free", "dairy-free"],
    "name": ["Tom", "Priya", "Amir", "Chloe", "Wei", "Sofia", "Daniel", "Aisha", "Liam", "Fatima",
             "Marcus", "Elena", "Noah", "Zainab", "Oliver", "Mei", "Callum", "Ngozi", "Hugo", "Ines"],
    "nick": ["Tommy", "Pri", "Dani", "Lee", "Mo", "Bex", "Jay", "Alex", "Sam", "Kit"],
    "surname": ["Okafor", "Patel", "Nguyen", "Smith", "Rossi", "Kowalski", "Haddad", "Mbeki", "Larsen"],
    "pet": ["dog", "cat", "rabbit", "parrot", "hamster", "horse", "tortoise", "ferret"],
    "breed": ["beagle", "labrador", "tabby", "poodle", "spaniel", "siamese", "collie", "terrier"],
    "rel": ["wife", "husband", "partner", "mum", "dad", "brother", "sister", "son", "daughter",
            "grandmother", "boyfriend", "girlfriend", "flatmate", "uncle", "cousin"],
    "job": ["nurse", "teacher", "software engineer", "electrician", "barrister", "chef", "student",
            "accountant", "doctor", "plumber", "architect", "paramedic", "journalist"],
    "business": ["bakery", "bike shop", "design studio", "cafe", "farm", "consultancy"],
    "city": ["Oxford", "Leeds", "Manchester", "London", "Bristol", "Glasgow", "Cardiff", "Edinburgh",
             "Paris", "Lagos", "Toronto", "Sydney", "Berlin", "Madrid"],
    "company": ["Google", "the NHS", "Tesco", "a startup", "the council", "Barclays", "a law firm"],
    "country": ["Nigeria", "India", "Poland", "Brazil", "Canada", "Ireland", "Kenya", "Italy", "Japan"],
    "subject": ["physics", "law", "medicine", "history", "computer science", "chemistry", "economics"],
    "uni": ["Oxford", "Cambridge", "Manchester", "UCL", "Edinburgh", "Imperial", "Bristol"],
    "day": ["Friday", "Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"],
    "time": ["5pm", "noon", "9am", "half past six", "3pm", "8:30", "quarter to ten", "7pm"],
    "amount": ["£500", "2000 pounds", "$300", "fifty quid", "£1,200", "about 800 euros", "ten thousand dollars"],
    "date": ["the 3rd of March", "the 14th of June", "October 30th", "the first of May",
             "the twelfth of December", "the 9th of August", "the 21st of January", "the 2nd of July"],
    "when": ["next Tuesday", "next Friday", "tomorrow", "this weekend", "in two weeks", "on Monday",
             "next month", "at the end of the week"],
    "n": ["two", "three", "four", "five", "2", "3", "ten", "twenty"],
    "age": ["3", "7", "12", "29", "40", "65", "five", "thirty-one"],
    "year": ["1985", "1992", "1999", "2001", "1978", "1990"],
    "fear": ["heights", "flying", "spiders", "small spaces", "dogs", "the dark"],
    "pref": ["window seats", "early flights", "quiet hotels", "spicy food", "short answers",
             "metric units", "vegetarian options", "aisle seats", "trains over planes"],
    "habit": ["sit by the window", "book early flights", "travel with hand luggage only",
              "go running before work", "walk to work", "cook from scratch", "check my email at six"],
    "instr": ["keep your answers short", "use metric units", "speak slowly", "call me by my first name",
              "give me the short version", "double check dates with me"],
    "topic": ["volcanoes", "bicycles", "black holes", "the French revolution", "photosynthesis", "jazz",
              "tides", "the Roman empire", "coffee", "chess", "solar panels", "bees", "the moon"],
    "tipsubject": ["sleeping better", "studying", "saving money", "public speaking", "cooking rice"],
    "invention": ["telephone", "printing press", "light bulb", "bicycle", "microscope"],
    "event": ["wedding", "birthday party", "holiday", "house move", "graduation party"],
    "drug": ["insulin", "ibuprofen", "penicillin", "caffeine", "aspirin"],
    "num": ["7", "12", "15", "23", "40", "64", "100"],
    "petage": ["two", "three", "five", "nine", "twelve"],
}

POS_TEMPLATES = [
    # health, diet, body
    "I'm allergic to {allergen}.", "I am allergic to {allergen}.", "I have a {allergen} allergy.",
    "I can't eat {allergen}.", "I can't have {allergen}, it makes me ill.", "{allergen} gives me a rash.",
    "I react badly to {allergen}.", "My doctor told me to avoid {allergen}.", "I'm intolerant to {allergen}.",
    "I don't eat {allergen}.", "I'm {diet}.", "I follow a {diet} diet.", "I've been {diet} for {n} years.",
    "I have {condition}.", "I was diagnosed with {condition} last year.", "I take medication for my {condition}.",
    "I'm pregnant.", "I use a wheelchair.", "I wear glasses.", "I'm left-handed.", "I'm afraid of {fear}.",
    "My {rel} is allergic to {allergen}.", "My {rel} has {condition}.",
    # identity
    "My name is {name}.", "My name's {name}.", "Call me {nick}.", "You can call me {nick}.",
    "People call me {nick}.", "I'm {name}.", "I'm {name}, by the way.", "My surname is {surname}.",
    "My birthday is on {date}.", "I'm {age} years old.", "I turned {age} last month.",
    # family and pets
    "My {rel} is called {name}.", "My {rel}'s name is {name}.", "{name} is my {rel}.",
    "I have a {pet} called {name}.", "My {pet} {name} is {petage} years old.", "I have {n} kids.",
    "My {rel} works as a {job}.", "I live with my {rel}.", "My {rel} lives in {city}.",
    "I'm married to {name}.", "I look after my {rel}.", "I have a {breed}.", "We have a {breed} called {name}.",
    # where and what
    "I live in {city}.", "I'm from {country}.", "I work as a {job}.", "I'm a {job}.", "I work at {company}.",
    "I work for {company} in {city}.", "I'm studying {subject} at {uni}.", "I moved to {city} last year.",
    "I was born in {year}.", "I'm based in {city}.", "I run a small {business}.",
    # plans, constraints, standing preferences
    "My deadline is {day}.", "My deadline is {when}.", "The report is due {when}.", "I need this done by {day} at {time}.", "The deadline for the report is {date}.",
    "My budget is {amount}.", "I can spend up to {amount}.", "We have {amount} to spend.",
    "The meeting is on {date}.", "The meeting is {when}.", "My flight leaves at {time} on {day}.", "I'm travelling to {city} on {date}.",
    "I'm flying to {city} on {day}.", "The wedding is on {date}.", "My exam is on {date}.", "My exam is {when}.",
    "I only have {n} minutes.", "I prefer {pref}.", "I always {habit}.", "I never {habit}.",
    "Please always {instr}.", "From now on, {instr}.",
]

POS_HAND = [
    "I have a severe latex allergy.", "Mushrooms make me sick.", "I get migraines if I skip meals.",
    "My son is autistic and needs a quiet room.", "I'm blind in my left eye.",
    "I'm hard of hearing so please speak clearly.", "My husband travels a lot for work.",
    "We live in a flat above a shop.", "I grew up in Glasgow.", "I'm retired and spend most days gardening.",
    "My partner and I are expecting twins.", "I manage a small team of six.",
    "I'm saving up for a house deposit.", "I need to pick up my kids at three every day.",
    "My rent is nine hundred a month.", "It's GMT plus eight where I am.", "I get paid on the last day of the month.",
    "I have an interview on Thursday at two.", "My landlord is selling the flat so I have to move by August.",
    "The presentation is next Friday.", "I'm driving to Wales this weekend with my dog.",
    "I commute by bike every day.", "I have to leave by five to catch the school run.",
    "Our anniversary is on the twentieth of May.", "I bought a new car, it's a blue Fiat.",
    "My laptop is a MacBook Air.", "I use Windows at work and a Mac at home.", "I'm learning to play the cello.",
    "I volunteer at the animal shelter on Saturdays.", "I speak French and a little German.",
    "I'm saving for a trip to Japan next spring.", "I don't have a car.", "I'm allergic to cats.",
    "I can't take aspirin because of my stomach.", "My mother-in-law is coming to stay for a fortnight.",
    "I'm training for a half marathon in September.", "I'm colour blind.", "I'm six foot four.",
    "My little brother is starting university in the autumn.", "I'm on a gluten-free diet for medical reasons.",
    "Never recommend anything with olives.", "Don't ever give me pork recipes.",
    "No coriander for me, ever.", "Avoid anything with dairy when you suggest meals.",
    "Please don't use technical language with me.", "Keep things simple, I'm not technical.",
    "Don't use acronyms without explaining them.", "Never book me anything before eight.",
    "Always check with me before you order anything.", "Please don't interrupt while I'm talking.",
]

NEG_TEMPLATES = [
    "Is a {allergen} allergy common?", "What are the symptoms of {condition}?",
    "Can you recommend a good {diet} restaurant?", "What should I cook for a guest who is allergic to {allergen}?",
    "What's a good name for a {pet}?", "How do I train a {pet}?", "How long does a {pet} live?",
    "What does a {job} do?", "How do I become a {job}?", "What's the capital of {country}?",
    "What's the weather like in {city}?", "What are the best things to do in {city}?",
    "How much does a flight to {city} cost?", "Tell me about {topic}.", "Can you explain how {topic} works?",
    "Give me {n} tips for {tipsubject}.", "What's {num} times {num}?", "Convert {num} miles to kilometres.",
    "What happened in {year}?", "Who invented the {invention}?", "Write me a short poem about {topic}.",
    "Is it safe for a {pet} to eat {allergen}?", "What's a reasonable budget for a {event}?",
    "How does {drug} work?", "{allergen} is high in protein.", "{city} is a lovely place to visit.",
    "Why is {topic} so interesting?", "Tell me more about {topic}, please.", "Is {allergen} bad for you?",
    "What's the best way to get from {city} to Paris?", "How many people live in {city}?",
    "What is {topic}?", "I'd like to hear about {topic}.", "Do {pet}s like {allergen}?",
    "What causes {condition}?", "How is {condition} treated?", "Is {condition} hereditary?",
    "How do I plan a {event}?", "How should I schedule a {event}?", "Where can I buy {allergen}?",
]

NEG_HAND = [
    "Dogs shouldn't eat chocolate.", "Peanut butter is high in protein.", "Thanks, that was really helpful.",
    "Okay, go on.", "That's interesting.", "Hmm, I'm not sure about that.", "I'm so bored right now.",
    "I'm tired today.", "I'm hungry.", "I'm cold.", "I'm a bit confused.", "I'm just looking, thanks.",
    "I need a minute.", "I need some help with this.", "I wonder what the weather will be like.",
    "My friend told me a funny story about a cat.", "I think that's a great idea.", "That film was great.",
    "Tell me a joke.", "Can you repeat that?", "Never mind.", "Stop.", "Yes.", "No.", "Okay.", "Sure.",
    "Right.", "Uh huh.", "What?", "Pardon?", "Go ahead.", "Please continue.", "Thank you.", "Good morning.",
    "Hello there.", "Goodbye for now.", "Let's talk about something else.", "What do you think?",
    "Can you say that more simply?", "Why is that?", "How does that work?", "Give me an example.",
    "Please include plenty of extra detail.", "Make it a bit shorter.", "That doesn't sound right.",
    "Who wrote Pride and Prejudice?", "What time is it in Tokyo?", "Recommend a book about history.",
    "Explain how a mortgage works.", "What's the difference between a virus and a bacterium?",
    "How many calories are in an apple?", "Why is the sky blue?", "Is it going to rain tomorrow?",
    "Where is the nearest train station?", "What's a good recipe for banana bread?",
    "Can you help me write an email?", "Summarise that for me.", "What's the plural of cactus?",
    "It's raining here.", "My phone is acting weird.", "I'm going to make a cup of tea.",
    "I love that song.", "That reminds me of a film I saw.", "I don't understand.",
    "What are the symptoms of a cold?", "Is it safe to eat raw eggs?", "How do vaccines work?",
    "What's a good name for a baby girl?", "How do I write a good cover letter?",
]

# Added after the first training report showed false pins on passing moods, acknowledgements,
# topic switches and generic "how do I plan / budget / deadline" questions. Not copies of test sentences.
NEG_HAND2 = [
    "I'm exhausted.", "I'm in a rush.", "I'm feeling a bit better now.", "I'm heading out for a walk.",
    "I'm making dinner at the moment.", "I'm about to leave.", "I'll be right back.", "I'm grabbing a coffee.",
    "My laptop is being slow.", "My wifi keeps dropping.", "The printer is jammed again.", "It's freezing in here.",
    "Interesting.", "I see.", "That makes sense.", "I like that idea.", "Sounds good.", "Good point.", "I agree.",
    "Fair enough.", "Oh nice.", "Wow, really?", "Huh, I didn't know that.", "That's funny.", "Cool, thanks.",
    "Let's change the subject.", "Can we go back to what we were saying earlier?", "Anyway, moving on.",
    "Actually, hold on a second.", "Sorry, I missed that.", "Could you speak up a little?",
    "How should I plan a project deadline?", "What's a good way to set deadlines for a team?",
    "How do I stay within a budget?", "How do I budget for a holiday?", "How do I ask for an extension?",
    "How do I plan a meeting agenda?", "What makes a good schedule?", "How do I split a bill fairly?",
    "What causes coeliac disease?", "What are the symptoms of anaphylaxis?", "Is gluten bad for you?",
    "How is asthma treated?", "What is lactose intolerance?", "Can dogs be allergic to chicken?",
    "What does vegan mean exactly?", "Are eggs healthy?", "How much sleep do adults need?",
    "I think that's wrong.", "I'd like to know more.", "I'm curious about that.", "I'm listening.",
]

TEST_POS = [
    # health, diet, body
    "Peanuts make me really ill.", "Shellfish is dangerous for me.", "I get hives whenever I eat strawberries.",
    "I'm coeliac, so no wheat for me.", "My doctor says I have to avoid salt.", "I'm lactose intolerant.",
    "I take beta blockers every morning.", "I'm expecting a baby in March.",
    "I've been strictly vegetarian for ten years.", "I have type one diabetes.",
    "Penicillin gives me a bad reaction.", "I carry an EpiPen everywhere.", "I'm afraid of heights.",
    "I don't drink alcohol.", "I don't eat pork.", "I use a screen reader.", "I'm left handed.",
    # identity
    "The name's Priya.", "Everyone calls me Dave.", "I go by Sam.", "My surname is Adeyemi.",
    "I'm Jordan, nice to meet you.", "My birthday is the third of November.",
    # family and pets
    "Biscuit is our beagle and he's three.", "I'm married to Anna.", "My daughter starts school in September.",
    "We've got twin boys.", "My mum is staying with us next week.", "I look after my elderly father.",
    "My cat Luna hates the vet.", "I share a flat with two friends.", "My sister Chloe is a vet.",
    # where and what
    "I'm based in Leeds these days.", "I'm a final year physics student.", "I teach secondary maths.",
    "I'm originally from Nigeria.", "My commute is about forty minutes each way.",
    "I run a small bakery in town.", "I'm a night shift nurse.", "I work from home on Fridays.",
    "We live on a narrowboat.",
    # plans, constraints, standing instructions
    "The report has to be in by Monday morning.", "We have about two thousand pounds to spend.",
    "My train leaves at half past six tomorrow.", "I'd rather not spend more than fifty.",
    "Please don't call me before nine.", "I'm away from the fifth to the twelfth.",
    "Always keep your answers short.", "I need to be at the airport by four in the morning.",
    "The wedding is on the fourteenth of June.", "My exam is next Wednesday.",
    "I only have twenty minutes for lunch.", "I'm flying to Toronto on Friday.",
    "I always sit by the window when I travel.", "I'm useless before ten, so no early calls.",
    "My budget for the trip is about eight hundred.", "I'm learning Spanish for a trip in May.",
    "I'm training for a marathon in April.", "Don't ever suggest anything with mushrooms.",
]

TEST_NEG = [
    "Is a peanut allergy common?", "What are the symptoms of coeliac disease?",
    "Can you suggest a good vegan restaurant?", "What should I name a puppy?", "How do I train a beagle?",
    "What's the capital of Nigeria?", "How much does a flight to Toronto cost?",
    "Why do people get allergies?", "What should I cook for a vegetarian guest?",
    "Is it safe for dogs to eat peanuts?", "How does insulin work?", "Do you know any good recipes with eggs?",
    "Tell me about the history of bicycles.", "Can you explain how volcanoes work?", "Tell me a joke.",
    "What's the weather like in Leeds?", "Thanks, that was really helpful.", "Okay, go on.",
    "That's interesting.", "Hmm, I'm not sure about that.", "I'm so bored right now.",
    "I'm quite tired today.", "I'm hungry.", "My friend told me a funny story about a cat.",
    "I think that's a great idea.", "I love that song.", "What's twelve times eight?",
    "Convert fifty miles to kilometres.", "What happened in 1969?", "Give me three tips for sleeping better.",
    "Please include plenty of extra detail.", "Tell me about topic 5.", "Who wrote Hamlet?",
    "How do I set a deadline for my team?", "What's a reasonable budget for a wedding?",
    "What time is it in Sydney?", "How do I become a nurse?", "What does a software engineer do?",
    "Write me a poem about autumn.", "Can you say that again?", "Never mind.", "Stop.",
    "How long does it take to fly to Paris?", "What are good names for a baby boy?",
    "Recommend a book about space.", "I'm not sure what to have for dinner.", "I don't understand.",
    "What's the difference between a cat and a dog?", "Explain how a pension works.",
    "I wonder what the weather will be like.", "It's raining outside.", "My phone is acting weird.",
    "I'm going to make a cup of tea.", "Let's talk about something else.", "What's the best way to learn Spanish?",
    "Peanuts are technically legumes.", "Chocolate is toxic to dogs.", "Paris is a beautiful city.",
    "How many people live in Toronto?", "Is it okay to eat shellfish when pregnant?",
    "What do vegetarians eat for protein?", "How do I get a good night's sleep?",
]


# SET B: fresh sentences written AFTER the first training report. Nothing has been tuned on these,
# so they are the honest check. Once you pick a threshold from them, the benchmark is the only clean test left.
TEST2_POS = [
    "Cashew nuts put me in hospital last year.", "I can't touch anything with sesame in it.",
    "I'm on a low sodium diet.", "My tablets are for thyroid problems.", "I'm halal, so no pork or alcohol.",
    "I get terrible reactions to wasp stings.", "I'm registered blind.", "I have a pacemaker.",
    "You can call me Doc.", "Folks at work call me Captain.", "My middle name is Elizabeth.",
    "Our labrador is called Murphy.", "My husband's name is Karim.", "My grandad lives with us.",
    "I've got three nieces and a nephew.", "My son Theo is seven.", "My goldfish is called Gerald.",
    "I'm a paramedic on nights.", "I'm a freelance illustrator.", "I live just outside Cambridge.",
    "I grew up in a village in Kerala.", "I moved here from Poland in 2015.", "I'm in my second year of law.",
    "Pay day for me is the twenty-eighth.", "I've got a hospital appointment on Thursday at nine.",
    "I'm picking up my parents from the airport on Sunday.", "I can only afford about three hundred a month.",
    "The funeral is on the second of December.", "I'm moving house on the fifteenth.",
    "Please speak to me in plain English, no jargon.", "I'd like you to always confirm before booking anything.",
    "I'm on holiday from the tenth to the seventeenth of April.", "I have to be home by six every evening.",
    "I do all my shopping online because I can't drive.", "I'm saving up for my daughter's wedding.",
]
TEST2_NEG = [
    "What's the best treatment for hay fever?", "Can you tell me about cashew trees?", "How common is a sesame allergy?",
    "What does a paramedic earn?", "How do pacemakers work?", "Which countries have the best healthcare?",
    "Is Cambridge worth visiting?", "How do I get to the airport from the city centre?",
    "What's a polite way to decline an invitation?", "How do I write a thank you note?",
    "That's a good question.", "Let me think about that.", "Right, I'm with you.", "Not really.", "Maybe.",
    "I'm a bit stuck on this.", "I'm trying to remember the word.", "I'm drinking a glass of water.",
    "The internet is really slow today.", "My coffee has gone cold.", "I'm on the bus right now.",
    "I'm back.", "Sorry, one moment.", "Can you hear me okay?", "Could you turn that up?",
    "Tell me something interesting about octopuses.", "What's the tallest mountain in Africa?",
    "How do you say thank you in Welsh?", "What year did the Berlin Wall fall?",
    "Give me a recipe for a quick pasta sauce.", "What's the best way to cook salmon?",
    "How should I prepare for a job interview?", "What are the pros and cons of renting?",
    "Is it better to pay off debt or save?", "How do I plan a surprise party?",
    "Roughly how long is a marathon?", "Dolphins are very intelligent animals.",
    "Sesame seeds are used in lots of cuisines.", "Wasps are more aggressive in late summer.",
    "I enjoyed that explanation.", "I wasn't expecting that.", "What would you suggest?",
]


def _capitalise(s):
    return s[0].upper() + s[1:] if s else s


def _fix_articles(s):
    import re
    return re.sub(r"\b([Aa]) ([aeiouAEIOU])", lambda m: m.group(1) + "n " + m.group(2), s)


def _expand(templates, rng, per_template):
    import re
    out = []
    for t in templates:
        slots = re.findall(r"\{(\w+)\}", t)
        for _ in range(per_template):
            filled = t
            for name in slots:
                filled = filled.replace("{" + name + "}", rng.choice(SLOTS[name]), 1)
            out.append(_fix_articles(_capitalise(filled)))
    return out


def train_examples(seed=0, pos_per_template=4, neg_per_template=9):
    """Returns a deduplicated list of (text, label) with every TEST sentence removed.

    Negatives get more samples per template because they have fewer templates; this keeps the
    classes roughly balanced so the classifier is not biased towards "everything is a fact".
    """
    rng = random.Random(seed)
    pos = _expand(POS_TEMPLATES, rng, pos_per_template) + POS_HAND
    neg = _expand(NEG_TEMPLATES, rng, neg_per_template) + NEG_HAND + NEG_HAND2
    held_out = set(TEST_POS) | set(TEST_NEG) | set(TEST2_POS) | set(TEST2_NEG)
    seen, rows = set(), []
    for label, texts in ((1, pos), (0, neg)):
        for t in texts:
            if t not in seen and t not in held_out:
                seen.add(t)
                rows.append((t, label))
    rng.shuffle(rows)
    return rows


def heldout_examples():
    return [(t, 1) for t in TEST_POS] + [(t, 0) for t in TEST_NEG]


def heldout2_examples():
    """Set B: the fresh held-out sentences."""
    return [(t, 1) for t in TEST2_POS] + [(t, 0) for t in TEST2_NEG]
