"""
Data Loaders and Dataset Utilities for TCF-Net
Supports:
  1. ISOT Fake News Dataset (fake.csv, true.csv)
  2. LIAR Dataset (zero-shot transfer evaluation)
  3. Code-Mixed Datasets: HASOC (Hinglish) & DravidianLangTech (Tamil-English)
  4. Built-in Multi-Task Synthetic Generator for standalone verification & demo fallbacks
  5. Confound-Scrubbed Text Pipeline and Subject-Transfer Splitting Protocol
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from features.scrubber import scrub_text

SEARCH_DIRS = [
    '.',
    'data',
    '../',
    '/content',
    '/content/drive/MyDrive'
]

def find_isot_files() -> Tuple[Optional[str], Optional[str]]:
    """Searches for ISOT fake.csv and true.csv files across common locations."""
    for d in SEARCH_DIRS:
        for fn, tn in [('fake.csv', 'true.csv'), ('Fake.csv', 'True.csv')]:
            fp, tp = os.path.join(d, fn), os.path.join(d, tn)
            if os.path.exists(fp) and os.path.exists(tp):
                return fp, tp
    return None, None

def load_isot(max_samples: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Loads ISOT dataset (Real Reuters wire articles vs Fake blog posts)."""
    fp, tp = find_isot_files()
    if fp is None:
        return None
    
    print(f"[DataLoader] Found ISOT at: {fp} | {tp}")
    df_fake = pd.read_csv(fp)
    df_true = pd.read_csv(tp)
    df_fake['label_veracity'] = 1
    df_true['label_veracity'] = 0
    df_fake['label_hate'] = -1  # Not annotated for hate speech
    df_true['label_hate'] = -1

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df['title'] = df.get('title', pd.Series([''] * len(df))).fillna('')
    df['text'] = df.get('text', pd.Series([''] * len(df))).fillna('')
    df['raw'] = (df['title'] + '. ' + df['text']).str.strip()
    df['body'] = df['text']
    df['subject'] = df.get('subject', pd.Series(['na'] * len(df))).fillna('na')
    df['lang'] = 'en'

    df = df[df['raw'].str.split().str.len() >= 10].reset_index(drop=True)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    return df[['raw', 'body', 'subject', 'label_veracity', 'label_hate', 'lang']]

def load_liar() -> Optional[pd.DataFrame]:
    """Loads LIAR dataset for cross-dataset generalization evaluation."""
    try:
        from datasets import load_dataset
        ds = load_dataset('liar', trust_remote_code=True)
        rows = []
        # LIAR labels: 0: false, 1: half-true, 2: mostly-true, 3: true, 4: barely-true, 5: pants-fire
        fake_ids, real_ids = {0, 4, 5}, {2, 3}
        for split in ['train', 'validation', 'test']:
            for r in ds[split]:
                lb = r['label']
                if lb in fake_ids:
                    y = 1
                elif lb in real_ids:
                    y = 0
                else:
                    continue  # Discard ambiguous half-true in standard evaluation
                rows.append({
                    'raw': r['statement'],
                    'body': r['statement'],
                    'subject': str(r.get('subject', 'na')),
                    'label_veracity': y,
                    'label_hate': -1,
                    'lang': 'en'
                })
        print(f"[DataLoader] Successfully loaded {len(rows)} samples from LIAR (HuggingFace)")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[DataLoader] LIAR load from HuggingFace skipped ({e})")
        return None

def load_code_mixed(file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Loads HASOC (Hinglish) or DravidianLangTech (Tamil-English) code-mixed datasets if present."""
    candidate_paths = [file_path] if file_path else [
        'data/hasoc_hinglish.csv',
        'data/dravidian_tamil_english.csv',
        'hasoc_hinglish.csv',
        'dravidian_tamil_english.csv'
    ]
    for p in candidate_paths:
        if p and os.path.exists(p):
            print(f"[DataLoader] Found code-mixed dataset at: {p}")
            df = pd.read_csv(p)
            df['raw'] = df.get('text', df.get('raw', '')).astype(str)
            df['body'] = df['raw']
            df['subject'] = 'social_media'
            df['label_veracity'] = df.get('label_veracity', pd.Series([-1]*len(df)))
            df['label_hate'] = df.get('label_hate', pd.Series([-1]*len(df)))
            df['lang'] = df.get('lang', 'code-mixed')
            return df[['raw', 'body', 'subject', 'label_veracity', 'label_hate', 'lang']]
    return None

def make_synthetic_multitask(n: int = 4000, noise: float = 0.03, seed: int = 42) -> pd.DataFrame:
    """
    Generates a controlled, diverse synthetic multi-task corpus for English, Hinglish, and Tanglish text.
    Includes both Veracity (Real vs Fake) and Hate Speech (Not-Hateful vs Hateful) targets,
    with borderline cases to evaluate SelectiveNet abstention.
    """
    rng = np.random.default_rng(seed)

    # English pools
    en_sensational = ['SHOCKING', 'BREAKING', "You WON'T believe", 'EXPOSED', 'Doctors HATE this',
                      'The truth they hide', 'MUST SHARE', 'URGENT', 'Secret revealed', 'BANNED']
    en_hedge = ['reportedly', 'allegedly', 'sources say', 'it is claimed', 'rumour has it', 'apparently']
    en_topic = ['vaccine', 'election', '5G network', 'celebrity', 'stock market', 'climate report',
                'immigration bill', 'new virus', 'space mission', 'tax reform']
    en_verb_f = ['causes mass destruction', 'is secretly controlling us', 'will destroy human health',
                 'is a proven international hoax', 'triggers global panic']
    en_verb_r = ['was thoroughly reviewed by', 'was published in a peer-reviewed journal',
                 'received formal budget approval from', 'was officially confirmed by']
    en_org = ['the health ministry', 'the research council', 'the central bank', 'parliament',
              'the independent statistics office']

    # Code-mixed Hinglish & Tamil-English pools
    hi_sensational = ['ARE YAAR DEKHO', 'KHATARNAK SACH', 'VIRAL VIDEO EXPOSED', 'SABSE BADA SCAM']
    hi_hate_words = ['gaddaar', 'deshdrohi', 'chor hai sab', 'bhagao inko', 'khatam karo']
    hi_normal = ['Aaj ki taza khabar', 'Karyakram shuru ho chuka hai', 'Aam janta ke hit me nirnay']

    ta_sensational = ['ADHIROOPAM BREAKING', 'MARMARA SEIDHI', 'UNMAI VELIYIL VANDHADHU', 'VIRAL ALERT']
    ta_hate_words = ['thurogi', 'ozhithu kattu', 'naasamai poga', 'kevalamana aal']
    ta_normal = ['Inraya thagaval', 'Arasu arivithu irukkirom', 'Makkalukku mukkiya arivippu', 'Kalloori ninaivu naal']

    # Pure Tamil script pools
    ta_script_sensational = ['அதிர்ச்சி செய்தி', 'மர்ம உண்மை வெளியானது', 'பரபரப்பு தகவல்', 'ரகசிய சதி அம்பலம்']
    ta_script_hate_words = ['துரோகிகள்', 'நாட்டை பாழாக்கும் கும்பல்', 'இவர்களை விரட்ட வேண்டும்']
    ta_script_normal = ['தமிழக அரசின் அதிகாரப்பூர்வ செய்திக்குறிப்பு', 'சுகாதாரத்துறை வெளியிட்ட அறிக்கை', 'பல்கலைக்கழக தேர்வு முடிவுகள்']

    rows = []
    subjects = ['politics', 'worldnews', 'health', 'technology', 'social_media']

    for i in range(n):
        lang_choice = rng.choice(['en', 'en', 'hi', 'ta', 'ta_script']) # Multi-lingual & code-mixed distribution
        is_hate = 0
        subj = rng.choice(subjects)

        if lang_choice == 'en':
            r = rng.random()
            if r < 0.38:
                # Clearly Fake
                text = f"{rng.choice(en_sensational)}!! {rng.choice(en_topic).title()} {rng.choice(en_verb_f)} - {rng.choice(en_hedge)}. SHARE before it gets deleted!!!"
                y_v = 1
                is_hate = 1 if rng.random() < 0.35 else 0
            elif r < 0.74:
                # Clearly Real
                text = f"The {rng.choice(en_topic).title()} {rng.choice(en_verb_r)} {rng.choice(en_org)} on {rng.integers(1, 28)} March, according to verified records."
                y_v = 0
                is_hate = 0
            else:
                # Ambiguous / Borderline -> models should exhibit lower confidence
                text = f"{rng.choice(en_topic).title()}: {rng.choice(en_hedge)}, {rng.choice(en_org)} discussed recent observations. Further details are awaited."
                y_v = int(rng.random() < 0.5)
                is_hate = 1 if rng.random() < 0.15 else 0

        elif lang_choice == 'hi':
            # Hinglish
            is_hate = 1 if rng.random() < 0.40 else 0
            y_v = 1 if rng.random() < 0.55 else 0
            if is_hate:
                text = f"{rng.choice(hi_sensational)}: Ye {rng.choice(hi_hate_words)} hamare desh ko loot rahe hai! {rng.choice(en_topic)} par jhoot bolte hai sab."
            else:
                text = f"{rng.choice(hi_normal)}: {rng.choice(en_topic).title()} ke bare me naye niyam jari kiye gaye hai official statement ke sath."

        elif lang_choice == 'ta':
            # Tamil-English (Tanglish)
            is_hate = 1 if rng.random() < 0.38 else 0
            y_v = 1 if rng.random() < 0.50 else 0
            if is_hate:
                text = f"{rng.choice(ta_sensational)}!! Indha {rng.choice(ta_hate_words)} naatta paazhadikiraanga. Please share with everyone immediately."
            else:
                text = f"{rng.choice(ta_normal)}: {rng.choice(en_topic).title()} patriya thagavalgalai aatchi thura therivithulladhu."

        else:
            # Pure Tamil script (தமிழ்)
            is_hate = 1 if rng.random() < 0.35 else 0
            y_v = 1 if rng.random() < 0.50 else 0
            if is_hate:
                text = f"{rng.choice(ta_script_sensational)}!! {rng.choice(ta_script_hate_words)} மீது நடவடிக்கை எடுக்க வேண்டும். உடனே பகிருங்கள்!"
            else:
                text = f"{rng.choice(ta_script_normal)}: {rng.choice(en_topic).title()} தொடர்பாக புதிய திட்டத்தை அரசு இன்று அறிவித்துள்ளது."

        if rng.random() < noise:
            y_v = 1 - y_v

        rows.append({
            'raw': text,
            'body': text,
            'subject': subj,
            'label_veracity': y_v,
            'label_hate': is_hate,
            'lang': lang_choice
        })

    print(f"[DataLoader] Generated {len(rows)} diverse multi-task samples (en/hi/ta)")
    return pd.DataFrame(rows)

TCF_MULTITASK_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tcf_multitask_dataset.csv'),
    'data/tcf_multitask_dataset.csv',
    'tcf_multitask_dataset.csv'
]

def load_tcf_multitask(max_samples: Optional[int] = None, seed: int = 42) -> Optional[pd.DataFrame]:
    """
    Loads the persisted 5,000-sample multilingual benchmark produced by data/generate_dataset.py
    (English, Tamil, Tanglish, Hinglish, Hindi) with veracity + hate speech + ambiguity labels.
    """
    for p in TCF_MULTITASK_PATHS:
        if os.path.exists(p):
            print(f"[DataLoader] Found TCF multi-task benchmark at: {p}")
            src = pd.read_csv(p, encoding='utf-8-sig')
            df = pd.DataFrame({
                'raw': src['statement_text'].astype(str),
                'body': src['statement_text'].astype(str),
                'subject': src['domain_subject'].astype(str),
                'label_veracity': src['veracity_label'].astype(int),
                'label_hate': src['hate_speech_label'].astype(int),
                'lang': src['language'].astype(str),
                'ambiguity': src['ambiguity_level'].astype(str)
            })
            if max_samples and len(df) > max_samples:
                df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
            return df
    return None

def load_dataset(max_samples: Optional[int] = 45000, seed: int = 42,
                 dataset: str = 'tcf_multitask') -> Tuple[pd.DataFrame, str]:
    """
    Loads the primary dataset.
      dataset='tcf_multitask' -> persisted multilingual benchmark (default)
      dataset='isot'          -> ISOT -> falls back to LIAR -> falls back to synthetic
      dataset='synthetic'     -> on-the-fly synthetic generator
    """
    df, source = None, None
    if dataset == 'tcf_multitask':
        df = load_tcf_multitask(max_samples=max_samples, seed=seed)
        source = 'TCF-Net Multilingual Multi-Task Benchmark (5 languages)'

    if dataset == 'isot' or (dataset == 'tcf_multitask' and df is None):
        df = load_isot(max_samples=max_samples)
        source = 'ISOT Fake/Real News (Kaggle)'

    if dataset == 'isot' and (df is None or len(df) < 500):
        df = load_liar()
        source = 'LIAR Dataset (HuggingFace)'

    if df is None or len(df) < 500:
        df = make_synthetic_multitask(n=4000, seed=seed)
        source = 'TCF-Net Multi-Task Synthetic Corpus (DEMO/TESTING)'

    df['raw'] = df['raw'].astype(str)
    df = df[df['raw'].str.split().str.len() >= 4].reset_index(drop=True)
    
    # Generate confound-scrubbed version for semantic stream
    df['content'] = [scrub_text(x) for x in df['raw']]
    df['style_src'] = df['raw'].str.slice(0, 1500)

    print(f"[DataLoader] Active Dataset: {source} (Total Samples: {len(df)})")
    return df, source

