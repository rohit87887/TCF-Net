"""
Dataset Generator & Exporter for TCF-Net (Review 3)
Generates and persists the full Multi-Task, Multilingual & Code-Mixed Benchmark Dataset:
  - Languages: English, Tamil (தமிழ்), Tamil-English (Tanglish), Hinglish, Hindi (हिन्दी)
  - Tasks: Veracity (Real vs Fake) & Hate Speech (Non-Hateful vs Hateful)
  - Ambiguity labels for SelectiveNet abstention evaluation
Exports to:
  - TCF_Net/data/tcf_multitask_dataset.csv
  - TCF_Net/data/tamil_tanglish_benchmark.csv
"""

import os
import numpy as np
import pandas as pd

def generate_full_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # 1. English Pools
    en_sensational = [
        'SHOCKING!! Scientists EXPOSED', 'URGENT ALERT: Truth they hide', 'BANNED VIDEO: You wont believe',
        'BREAKING: Secret government report leaks', 'MUST SHARE BEFORE DELETED', 'EXPOSED: Massive fraud in health system'
    ]
    en_real_orgs = [
        'World Health Organization', 'Ministry of Health and Family Welfare', 'Reuters Fact Check Desk',
        'Indian Council of Medical Research (ICMR)', 'National Statistics Bureau', 'Peer-reviewed Medical Journal'
    ]
    en_topics = ['COVID booster shot', '5G telecom network', 'parliamentary election bill', 'central bank digital rupee', 'clean energy policy', 'border security report']
    en_fake_verbs = ['causes permanent cell mutation', 'is secretly embedding tracking chips', 'is an international billionaire conspiracy', 'poisons local groundwater']
    en_real_verbs = ['published quarterly clinical trial metrics', 'confirmed phase-3 efficacy within projected safety standards', 'issued official regulatory guidance on']

    # 2. Pure Tamil Script Pools (தமிழ்)
    ta_sensational = [
        'அதிர்ச்சி செய்தி!! ரகசிய உண்மை வெளியானது', 'மர்ம மரணம் - அரசாங்கம் மறைக்கும் சதி',
        'பரபரப்பு தகவல்! உடனே பாருங்கள்', 'வாட்ஸ்அப் வைரல்: யாரும் சொல்லாத உண்மை'
    ]
    ta_real_orgs = [
        'தமிழக சுகாதாரத்துறை', 'சென்னை உயர்நீதிமன்றம்', 'அண்ணா பல்கலைக்கழகம்',
        'இந்திய விண்வெளி ஆராய்ச்சி நிறுவனம் (ISRO)', 'தமிழ்நாடு அரசு தேர்வுத்துறை'
    ]
    ta_topics = ['புதிய தடுப்பூசி', 'பள்ளி மாணவர் நலத்திட்டம்', '5G அலைக்கற்றை சோதனை', 'ரேஷன் அட்டை புதிய விதி', 'மழை வெள்ள முன்னெச்சரிக்கை']
    ta_fake_verbs = ['பறவைகள் மற்றும் விலங்குகளை கொன்று குவிக்கும் ஆபத்து', 'உடலில் ரகசியமாக விஷத்தை கலக்கும் திட்டம்', 'மக்களை ஏமாற்றும் மிகப்பெரிய சதி']
    ta_real_verbs = ['தொடர்பாக அதிகாரப்பூர்வ செய்திக்குறிப்பு இன்று வெளியிடப்பட்டுள்ளது', 'வெற்றிகரமாக செயல்படுத்தப்பட்டு வருவதாக அறிவித்துள்ளது', 'திட்டத்திற்கான புதிய வழிகாட்டுதல்களை வெளியிட்டுள்ளது']
    ta_hate_phrases = [
        'இந்த துரோகி கும்பலை தமிழ்நாட்டை விட்டே விரட்ட வேண்டும், நாசமாய் போக வேண்டியவர்கள்.',
        'இவர்கள் நமது கலாச்சாரத்தை அழிக்க வந்த நச்சு விஷங்கள், இவர்களை சும்மா விடக்கூடாது.',
        'அந்நிய நாட்டு கைக்கூலிகள் நாட்டை கொள்ளையடிக்கிறார்கள், உடனடியாக துரத்த வேண்டும்.'
    ]

    # 3. Tamil + English (Tanglish Code-Mixed) Pools
    tanglish_sensational = [
        'ADHIROOPAM BREAKING!! Indha news yaarum solla maattanga', 'VIRAL ALERT: Secrete conspiracy exposed',
        'SHOCKING LEAK: WhatsApp-la fast-ah share pannunga', 'BAYANGARAMAANA UNMAI: Doctors hide panraanga'
    ]
    tanglish_real_orgs = [
        'Tamil Nadu Health Department', 'Anna University Exam Controller', 'Chennai Metro Rail Corporation',
        'TNPSC Official Board', 'District Collectorate Office'
    ]
    tanglish_topics = ['semester exam schedule', 'bus fare subsidy', 'new tech park project', 'COVID precaution norms', 'electricity tariff revision']
    tanglish_fake_verbs = ['dangerous poison irukku nu proof vandhachu', 'secret chip insert panraanga nu doctor reveal pannitaaru', 'full scam nu report leak aagirukku']
    tanglish_real_verbs = ['official website-la formal notification update pannirukaanga', 'successfully review panni press release release pannitanga', 'budget approval kedaichadhum implementation start aagum nu announce pannitaanga']
    tanglish_hate_phrases = [
        'Indha thurogi kootam namba naatta muzhusa paazhadikiranga! Inga irundhu ivangala ellam thurathi adikanum, romba kevalamana aalunga.',
        'Vera state aalunga ellam namba rights-ah pudunguraanga, ivangala tholacha dhaan namba vazha mudiyum.',
        'Indha corrupt politicians and fraud gangs-ah ozhithu kattu-na dhaan naadu urupadum.'
    ]

    # 4. Hinglish (Hindi + English) Pools
    hi_sensational = [
        'KHATARNAK SACH EXPOSED!! Dekhiye ye viral sach', 'SABSE BADA SCAM LEAK: Media chup hai',
        'ARE YAAR DEKHO: Aaj raat se sab badal jayega', 'SAVDHAN!! Secret circular viral ho gaya'
    ]
    hi_real_orgs = [
        'Reserve Bank of India (RBI)', 'Health Ministry Official Portal', 'Railway Recruitment Board',
        'Supreme Court Public Relations Office', 'IIT Delhi Research Team'
    ]
    hi_topics = ['500 rupee notes', 'railway exam schedule', 'new digital tax rules', 'free ration scheme', 'vaccination drive']
    hi_fake_verbs = ['aaj raat se illegal declare hone wala hai', 'secretly sabka bank balance block karne ka plan hai', 'poora public ko bewaqoof banane ki saazish hai']
    hi_real_verbs = ['par clarifying statement jaari karke afwaaho ka khandan kiya hai', 'ke liye new guidelines official portal pe upload kar diya hai', 'ke targets ko successfully achieve kar liya hai']
    hi_hate_phrases = [
        'Ye gaddaar log desh ko andar se khokhla kar rahe hai! Inko sabak sikhana bohot zaroori hai.',
        'Gaddaron ko desh se bahar nikalo, inke wajah se poora samaaj barbaad ho raha hai.',
        'In fraud logon ko support karne wale bhi barabar ke gunahgar hai, inko jail bhejo.'
    ]

    # 5. Pure Hindi Script Pools (हिन्दी)
    hi_script_sensational = ['सनसनीखेज खुलासा!! इस बड़े घोटाले की सच्चाई आई सामने', 'वायरल अलर्ट: स्वास्थ्य विभाग ने छिपाई यह गंभीर बात']
    hi_script_real_orgs = ['स्वास्थ्य एवं परिवार कल्याण मंत्रालय', 'भारतीय रिजर्व बैंक (RBI)', 'नीति आयोग']
    hi_script_topics = ['राष्ट्रीय स्वास्थ्य मिशन', 'डिजिटल मुद्रा दिशानिर्देश', 'नई शिक्षा नीति']

    rows = []
    langs = ['English', 'Tamil (தமிழ்)', 'Tamil-English (Tanglish)', 'Hinglish', 'Hindi (हिन्दी)']
    weights = [0.35, 0.20, 0.20, 0.15, 0.10] # Balanced real-world distribution

    for i in range(1, n + 1):
        chosen_lang = rng.choice(langs, p=weights)
        is_hate = 0
        is_fake = 0
        ambiguity = "Clear Fact"
        subj = rng.choice(['Health & Science', 'Governance', 'National Politics', 'Social Media Alerts', 'Education & Tech'])

        if chosen_lang == 'English':
            mode = rng.random()
            if mode < 0.42:
                # Fake
                text = f"{rng.choice(en_sensational)}: {rng.choice(en_topics).title()} {rng.choice(en_fake_verbs)}. Share before authorities delete this link!!!"
                is_fake = 1
                ambiguity = "Sensational Deception"
                is_hate = 1 if rng.random() < 0.25 else 0
            elif mode < 0.82:
                # Real
                text = f"The {rng.choice(en_real_orgs)} has {rng.choice(en_real_verbs)} {rng.choice(en_topics)} during its scheduled review on {rng.integers(1, 28)} March."
                is_fake = 0
                ambiguity = "Clear Fact"
            else:
                # Borderline / Ambiguous (for SelectiveNet abstention)
                text = f"Unconfirmed reports suggest officials may consider revising parameters regarding {rng.choice(en_topics)}, though formal documentation is still awaited."
                is_fake = int(rng.random() < 0.5)
                ambiguity = "Borderline / Needs Fact-Check"

        elif chosen_lang == 'Tamil (தமிழ்)':
            mode = rng.random()
            if mode < 0.42:
                text = f"{rng.choice(ta_sensational)}!! {rng.choice(ta_topics)} {rng.choice(ta_fake_verbs)}! உடனடியாக வாட்ஸ்அப்பில் அனைவருக்கும் பகிருங்கள்."
                is_fake = 1
                ambiguity = "Sensational Deception"
                if rng.random() < 0.30:
                    text += " " + rng.choice(ta_hate_phrases)
                    is_hate = 1
            elif mode < 0.82:
                text = f"{rng.choice(ta_real_orgs)} {rng.choice(ta_topics)} {rng.choice(ta_real_verbs)}."
                is_fake = 0
                ambiguity = "Clear Fact"
            else:
                text = f"{rng.choice(ta_topics)} தொடர்பாக அடுத்த வாரம் புதிய மாற்றம் வர வாய்ப்பு உள்ளதாக கூறப்படுகிறது, ஆனால் அதிகாரப்பூர்வ உறுதிப்படுத்தல் இன்னும் வரவில்லை."
                is_fake = int(rng.random() < 0.5)
                ambiguity = "Borderline / Needs Fact-Check"

        elif chosen_lang == 'Tamil-English (Tanglish)':
            mode = rng.random()
            if mode < 0.42:
                text = f"{rng.choice(tanglish_sensational)}!! {rng.choice(tanglish_topics).title()} {rng.choice(tanglish_fake_verbs)}! Please share with everyone immediately."
                is_fake = 1
                ambiguity = "Sensational Deception"
                if rng.random() < 0.35:
                    text += " " + rng.choice(tanglish_hate_phrases)
                    is_hate = 1
            elif mode < 0.82:
                text = f"{rng.choice(tanglish_real_orgs)} has officially announced that {rng.choice(tanglish_topics)} {rng.choice(tanglish_real_verbs)}."
                is_fake = 0
                ambiguity = "Clear Fact"
            else:
                text = f"Oruvela {rng.choice(tanglish_topics)} patriya decision adutha month change aaga chance iruku nu news kelvipatten, but unmaiya nu confirm aagala."
                is_fake = int(rng.random() < 0.5)
                ambiguity = "Borderline / Needs Fact-Check"

        elif chosen_lang == 'Hinglish':
            mode = rng.random()
            if mode < 0.45:
                text = f"{rng.choice(hi_sensational)}!! {rng.choice(hi_topics).title()} {rng.choice(hi_fake_verbs)}. Jaldi se jaldi share karein sabhi dosto ko!"
                is_fake = 1
                ambiguity = "Sensational Deception"
                if rng.random() < 0.35:
                    text += " " + rng.choice(hi_hate_phrases)
                    is_hate = 1
            elif mode < 0.82:
                text = f"{rng.choice(hi_real_orgs)} ne {rng.choice(hi_topics)} {rng.choice(hi_real_verbs)}."
                is_fake = 0
                ambiguity = "Clear Fact"
            else:
                text = f"Social media par khabar hai ki {rng.choice(hi_topics)} me kuch badlav ho sakta hai, halanki official confirmation ka intezar hai."
                is_fake = int(rng.random() < 0.5)
                ambiguity = "Borderline / Needs Fact-Check"

        else:
            # Pure Hindi (हिन्दी)
            mode = rng.random()
            if mode < 0.45:
                text = f"{rng.choice(hi_script_sensational)}: {rng.choice(hi_script_topics)} को लेकर अधिकारियों पर लगे बड़े आरोप। इस संदेश को अधिक से अधिक साझा करें।"
                is_fake = 1
                ambiguity = "Sensational Deception"
            else:
                text = f"{rng.choice(hi_script_real_orgs)} ने आज एक आधिकारिक प्रेस विज्ञप्ति जारी कर {rng.choice(hi_script_topics)} की प्रगति रिपोर्ट प्रस्तुत की है।"
                is_fake = 0
                ambiguity = "Clear Fact"

        rows.append({
            'sample_id': f"TCF_{i:05d}",
            'statement_text': text,
            'language': chosen_lang,
            'veracity_label': is_fake,
            'veracity_class': 'Fake' if is_fake == 1 else 'Real',
            'hate_speech_label': is_hate,
            'hate_speech_class': 'Hateful' if is_hate == 1 else 'Non-Hateful',
            'domain_subject': subj,
            'ambiguity_level': ambiguity
        })

    df = pd.DataFrame(rows)
    return df

def main():
    os.makedirs('data', exist_ok=True)
    print("Generating comprehensive TCF-Net Multi-Task Dataset (5,000 samples)...")
    df = generate_full_dataset(n=5000, seed=42)

    main_csv = os.path.join('data', 'tcf_multitask_dataset.csv')
    df.to_csv(main_csv, index=False, encoding='utf-8-sig')
    print(f"-> Saved primary dataset: {main_csv} ({len(df)} rows)")

    # Save dedicated Tamil / Tanglish slice for Review 3 Committee
    indic_df = df[df['language'].isin(['Tamil (தமிழ்)', 'Tamil-English (Tanglish)'])].copy()
    indic_csv = os.path.join('data', 'tamil_tanglish_benchmark.csv')
    indic_df.to_csv(indic_csv, index=False, encoding='utf-8-sig')
    print(f"-> Saved Tamil/Tanglish benchmark slice: {indic_csv} ({len(indic_df)} rows)")

    # Save breakdown statistics
    summary = {
        'total_samples': len(df),
        'language_distribution': df['language'].value_counts().to_dict(),
        'veracity_distribution': df['veracity_class'].value_counts().to_dict(),
        'hate_speech_distribution': df['hate_speech_class'].value_counts().to_dict(),
        'ambiguity_breakdown': df['ambiguity_level'].value_counts().to_dict()
    }
    import json
    sum_json = os.path.join('data', 'dataset_summary.json')
    with open(sum_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"-> Saved dataset summary: {sum_json}")

if __name__ == "__main__":
    main()
