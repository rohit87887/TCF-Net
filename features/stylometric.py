"""
Modality B: Stylometric & Affective Feature Extractor (22-D)
Grounded in deception linguistics: deceptive and sensational texts exhibit higher hedging,
amplified superlatives, heavier punctuation, altered pronoun distributions, and skewed affect loading.
Supports English and code-mixed (Hinglish, Tamil-English) Romanized inputs.
"""

import re
import numpy as np

# Linguistic lexicons for deception analysis across English, Tamil, Tanglish, Hindi, and Hinglish
HEDGE = {
    'maybe', 'perhaps', 'possibly', 'allegedly', 'reportedly', 'apparently', 'seems', 'suggests',
    'claimed', 'rumour', 'rumor', 'supposedly', 'purportedly', 'could', 'might', 'unconfirmed',
    # Hinglish & Hindi
    'shayad', 'lagta', 'bolte', 'afwaah', 'kahate', 'suna', 'शायद', 'अफवाह',
    # Tamil & Tanglish
    'oruvela', 'solraanga', 'kelvipatten', 'doubt-ah', 'ஒருவேளை', 'கூறப்படுகிறது', 'சொல்கிறார்கள்', 'வதந்தி'
}

SENSATIONAL = {
    'shocking', 'unbelievable', 'breaking', 'urgent', 'exposed', 'secret', 'banned', 'hoax',
    'terrifying', 'miracle', 'destroy', 'outrageous', 'insane', 'bombshell', 'explosive',
    # Hinglish & Hindi
    'khatarnak', 'hungama', 'dhamaal', 'khulasa', 'sharmnak', 'bhasad', 'सनसनीखेज', 'खुलासा', 'हंगामा', 'धमाका',
    # Tamil & Tanglish
    'adhirchi', 'breaking-la', 'marmam', 'bayangarama', 'viral-ah', 'அதிர்ச்சி', 'மர்மம்', 'வெளியானது', 'சதி', 'பரபரப்பு'
}

SUPERLATIVE = {
    'best', 'worst', 'greatest', 'biggest', 'smallest', 'most', 'least', 'ever', 'never',
    'always', 'ultimate', 'perfect',
    # Indic
    'sabse', 'ekdam', 'सबसे', 'periya', 'mutrilum', 'மிகப்பெரிய', 'முதல்முறை', 'முழுவதும்'
}

NEGATION = {
    'no', 'not', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', "n't",
    # Hindi / Hinglish
    'nahi', 'mat', 'nhi', 'jhooth', 'नहीं', 'मत', 'झूठ',
    # Tamil / Tanglish
    'ille', 'illa', 'vendam', 'pannatheenga', 'poi', 'இல்லை', 'கூடாது', 'வேண்டாம்', 'பொய்'
}

FIRST_P = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours',
    'main', 'mera', 'hum', 'hamara', 'நான்', 'நாம்', 'நமது', 'naan', 'namba', 'enaku'
}
SECOND_P = {
    'you', 'your', 'yours',
    'tu', 'tum', 'aap', 'tera', 'apka', 'நீ', 'நீங்கள்', 'உன்', 'nee', 'neenga', 'ungala'
}
MODALS = {'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must'}

AFFECT_POS = {
    'great', 'amazing', 'wonderful', 'success', 'win', 'safe', 'proven', 'trusted',
    'shandar', 'badhiya', 'शानदार', 'बढ़िया', 'azhagu', 'nalladhu', 'super', 'வெற்றி', 'நன்மை', 'அழகு'
}

AFFECT_NEG = {
    'danger', 'fear', 'death', 'crisis', 'attack', 'fraud', 'lie', 'corrupt', 'toxic', 'panic',
    'hate', 'kill', 'threat', 'barbaad', 'dhokha', 'jhooth', 'धोखा', 'खतरा', 'मौत',
    'kolai', 'thurogi', 'mosadi', 'கொலை', 'துரோகம்', 'மோசடி', 'அபாயம்'
}

STYLE_NAMES = [
    'char_len', 'word_len', 'avg_word_len', 'sent_count', 'avg_sent_len',
    'excl_ratio', 'ques_ratio', 'caps_word_ratio', 'digit_ratio', 'punct_ratio',
    'quote_ratio', 'hedge_density', 'sensational_density', 'superlative_density',
    'negation_density', 'first_person', 'second_person', 'modal_density',
    'lexical_diversity', 'flesch_ease', 'affect_polarity', 'punct_burstiness'
]

# Python's \w excludes Indic vowel signs and viramas (Unicode Mn/Mc), which would split
# Tamil and Devanagari words into fragments; include those script blocks explicitly.
WORD_RE = re.compile(r"[\w'\u0900-\u097F\u0B80-\u0BFF]+", re.UNICODE)

def _safe_flesch_reading_ease(text: str) -> float:
    """Computes Flesch Reading Ease normalized to ~[0, 1], with fallback if textstat unavailable."""
    try:
        import textstat
        return float(textstat.flesch_reading_ease(text)) / 100.0
    except Exception:
        # Heuristic fallback: 206.835 - 1.015*(words/sents) - 84.6*(syllables/words)
        words = WORD_RE.findall(text)
        sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        w_cnt = max(len(words), 1)
        s_cnt = max(len(sents), 1)
        avg_w_len = sum(len(w) for w in words) / w_cnt
        score = 1.0 - (0.015 * (w_cnt / s_cnt) + 0.1 * avg_w_len)
        return float(np.clip(score, 0.0, 1.0))

def extract_stylometric_features(text: str) -> list:
    """
    Extracts a 22-dimensional stylometric feature vector from raw input text.
    """
    t = str(text) if text is not None else ""
    words = WORD_RE.findall(t)
    lw = [w.lower() for w in words]
    n = max(len(words), 1)
    sents = [s for s in re.split(r'[.!?]+', t) if s.strip()]
    ns = max(len(sents), 1)

    caps = sum(1 for w in words if len(w) > 2 and w.isupper())
    puncts = re.findall(r'[.,!?;:\-—]', t)
    pos = [m.start() for m in re.finditer(r'[.,!?;:]', t)]
    gaps = np.diff(pos) if len(pos) > 2 else np.array([0.0])

    def dens(S):
        return sum(1 for w in lw if w in S) / n

    flesch = _safe_flesch_reading_ease(t)

    vec = [
        len(t) / 1000.0,                                      # char_len
        n / 100.0,                                            # word_len
        float(np.mean([len(w) for w in words])) if words else 0.0,  # avg_word_len
        ns / 10.0,                                            # sent_count
        float(n / ns),                                        # avg_sent_len
        float(t.count('!') / n),                              # excl_ratio
        float(t.count('?') / n),                              # ques_ratio
        float(caps / n),                                      # caps_word_ratio
        float(sum(c.isdigit() for c in t) / max(len(t), 1)),  # digit_ratio
        float(len(puncts) / max(len(t), 1)),                  # punct_ratio
        float((t.count('"') + t.count("'")) / n),             # quote_ratio
        float(dens(HEDGE)),                                   # hedge_density
        float(dens(SENSATIONAL)),                             # sensational_density
        float(dens(SUPERLATIVE)),                             # superlative_density
        float(dens(NEGATION)),                                # negation_density
        float(dens(FIRST_P)),                                 # first_person
        float(dens(SECOND_P)),                                # second_person
        float(dens(MODALS)),                                  # modal_density
        float(len(set(lw)) / n),                              # lexical_diversity
        float(flesch),                                        # flesch_ease
        float(dens(AFFECT_POS) - dens(AFFECT_NEG)),           # affect_polarity
        float(np.std(gaps)) / 100.0                           # punct_burstiness
    ]
    return vec

def batch_extract_stylometric(texts: list) -> np.ndarray:
    """Extracts stylometric matrices for a list of texts and cleans NaNs."""
    mat = np.array([extract_stylometric_features(t) for t in texts], dtype=np.float32)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

