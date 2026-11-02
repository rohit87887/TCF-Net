"""
Confound Scrubber for TCF-Net
Neutralizes publisher-specific artifacts (Reuters/AP wire datelines, Unicode fingerprints,
(VIDEO) tags, URLs, social handles, stripped contractions) from article bodies before semantic encoding.
"""

import re
import unicodedata

AGENCY_RE = r'(Reuters|AP|Associated Press|AFP|Bloomberg|21st Century Wire|Getty Images)'

def scrub_text(text: str, max_chars: int = 1500) -> str:
    """
    Scrubs publisher-identifying tokens and typographic artifacts from text.
    The stylometric stream continues to receive raw text so sensational punctuation/casing
    remains available as genuine stylistic signal.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    s = unicodedata.normalize('NFKC', text)
    # Neutralize typographic quote variations that form publisher fingerprints
    for a, b in [('\u2019', "'"), ('\u2018', "'"), ('\u201c', '"'), ('\u201d', '"'),
                 ('\u2013', '-'), ('\u2014', '-'), ('\u2026', '...')]:
        s = s.replace(a, b)
        
    # Remove wire datelines: e.g. "WASHINGTON (Reuters) -"
    s = re.sub(r'^\s*[A-Z][A-Za-z .,/\'-]{0,45}\(' + AGENCY_RE + r'\)\s*[-:]*\s*', '', s)
    s = re.sub(r'\(' + AGENCY_RE + r'\)', ' ', s)
    s = re.sub(AGENCY_RE, ' ', s)
    
    # Strip multimedia tags and social artifacts
    s = re.sub(r'\((VIDEO|IMAGES?|TWEETS?|PHOTOS?|AUDIO|WATCH)\)', ' ', s, flags=re.I)
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r'@\w+', ' ', s)
    s = re.sub(r'\bFeatured image via[^.]*\.', ' ', s, flags=re.I)
    
    # Standardize contractions to defeat quote-stripping fingerprints
    s = s.replace("'", '').lower()
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s[:max_chars]

