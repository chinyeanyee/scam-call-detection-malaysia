# ==============================================
# 01_data_preparation.py
# Step 1: Clean raw MagicHub → Cleaned_Convo_Transcript
# Step 2: Load all sources into master CSV
#   - real_scam (Scam_Transcript)
#   - synthetic_scam (Synthetic_Scam_Transcript)
#   - normal_convo (Cleaned_Convo_Transcript)
#   - normal_callcentre (Cleaned_CallCenter_Transcript)
# ==============================================

import os
import re
import pandas as pd
from config import (
    SCAM_TRANSCRIPT_FOLDER,
    SCAM_SYNTHETIC_FOLDER,
    RAW_NORMAL_FOLDER,
    CLEANED_CONVO_FOLDER,
    CLEANED_CALLCENTER_FOLDER,
    MASTER_CSV,
    RANDOM_SEED
)

# ==============================================
# TEXT CLEANERS
# ==============================================

def clean_magichub_transcript(raw_text):
    """
    Strips MagicHub format:
    [0.730,2.599]  G0369  female,Malaysia  dialogue text
    Returns dialogue text only.
    """
    lines          = raw_text.strip().split('\n')
    dialogue_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split('\t')

        if len(parts) >= 4:
            dialogue = parts[3].strip()
        elif line.startswith('['):
            match    = re.match(r'^\[[\d.,]+\]\s+\S+\s+\S+\s+(.+)$', line)
            dialogue = match.group(1).strip() if match else ''
        else:
            dialogue = line

        dialogue = re.sub(r'\[UNK\]', '', dialogue).strip()
        if dialogue:
            dialogue_lines.append(dialogue)

    return ' '.join(dialogue_lines).strip()


def clean_text(text):
    """
    Universal cleaner applied to ALL transcripts.
    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    - Remove 'scammer' word — YouTube video title artifact
      that makes classification trivially easy
    - Does NOT remove stopwords — handled separately
      in TF-IDF pipeline only
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove 'scammer' — data artifact from YouTube titles
    # Forces model to learn real fraud linguistic patterns
    text = re.sub(r'scammer\w*', '', text)  # removes scammer, scammers, scammering etc
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ==============================================
# STEP 1: CLEAN RAW MAGICHUB → Cleaned_Convo_Transcript
# Resume-safe: skips already cleaned files
# ==============================================

def clean_raw_magichub():
    if not os.path.exists(RAW_NORMAL_FOLDER):
        print(f"⚠️  RAW_NORMAL_FOLDER not found: {RAW_NORMAL_FOLDER}")
        return

    os.makedirs(CLEANED_CONVO_FOLDER, exist_ok=True)

    raw_files = [f for f in sorted(os.listdir(RAW_NORMAL_FOLDER))
                 if f.endswith('.txt')]

    print(f"Found {len(raw_files)} raw MagicHub files")

    cleaned_count = 0
    skipped_count = 0

    for filename in raw_files:
        output_path = os.path.join(CLEANED_CONVO_FOLDER, filename)

        if os.path.exists(output_path):
            print(f"  ⏭️  Already cleaned: {filename}")
            skipped_count += 1
            continue

        input_path = os.path.join(RAW_NORMAL_FOLDER, filename)
        with open(input_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            raw = f.read()

        cleaned    = clean_text(clean_magichub_transcript(raw))
        word_count = len(cleaned.split())

        if word_count < 100:
            print(f"  ⚠️  SKIP {filename} — only {word_count} words")
            skipped_count += 1
            continue

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)

        print(f"  ✅ Cleaned {filename} — {word_count} words")
        cleaned_count += 1

    print(f"\n  Newly cleaned : {cleaned_count}")
    print(f"  Skipped       : {skipped_count}")


# ==============================================
# STEP 2: LOAD FUNCTIONS
# ==============================================

def load_scam_transcripts():
    files = [f for f in sorted(os.listdir(SCAM_TRANSCRIPT_FOLDER))
             if f.endswith('.txt')]

    print(f"\nFound {len(files)} real scam transcripts")
    rows = []

    for filename in files:
        path = os.path.join(SCAM_TRANSCRIPT_FOLDER, filename)
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            content = f.read()

        rows.append({
            'source_file' : filename,
            'content'     : clean_text(content),
            'label'       : 1,
            'data_type'   : 'real_scam'
        })

    print(f"  ✅ Loaded {len(rows)} real scam transcripts")
    return rows


def load_synthetic_scam():
    if not os.path.exists(SCAM_SYNTHETIC_FOLDER):
        print(f"⚠️  SCAM_SYNTHETIC_FOLDER not found: {SCAM_SYNTHETIC_FOLDER}")
        return []

    files = [f for f in sorted(os.listdir(SCAM_SYNTHETIC_FOLDER))
             if f.endswith('.txt')]

    print(f"\nFound {len(files)} synthetic scam transcripts")
    rows = []

    for filename in files:
        path = os.path.join(SCAM_SYNTHETIC_FOLDER, filename)
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        rows.append({
            'source_file' : filename,
            'content'     : clean_text(content),
            'label'       : 1,
            'data_type'   : 'synthetic_scam'
        })

    print(f"  ✅ Loaded {len(rows)} synthetic scam transcripts")
    return rows


def load_normal_folder(folder, data_type):
    if not os.path.exists(folder):
        print(f"⚠️  Folder not found: {folder}")
        return []

    files = [f for f in sorted(os.listdir(folder))
             if f.endswith('.txt')]

    print(f"\nFound {len(files)} files in {os.path.basename(folder)}")
    rows    = []
    skipped = 0

    for filename in files:
        path = os.path.join(folder, filename)
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        cleaned    = clean_text(content)
        word_count = len(cleaned.split())

        if word_count < 50:
            print(f"  ⚠️  SKIP {filename} — {word_count} words")
            skipped += 1
            continue

        rows.append({
            'source_file' : filename,
            'content'     : cleaned,
            'label'       : 0,
            'data_type'   : data_type
        })

    print(f"  ✅ Kept {len(rows)} | Skipped {skipped}")
    return rows


# ==============================================
# STEP 3: BUILD MASTER CSV
# ==============================================

def build_master_csv():
    all_data = []

    print("\n" + "="*50)
    print("Loading all data sources")
    print("="*50)

    all_data += load_scam_transcripts()
    all_data += load_synthetic_scam()
    all_data += load_normal_folder(CLEANED_CONVO_FOLDER, 'normal_convo')
    all_data += load_normal_folder(CLEANED_CALLCENTER_FOLDER, 'normal_callcentre')

    df = pd.DataFrame(all_data)

    before = len(df)
    df     = df[df['content'].str.strip() != ''].reset_index(drop=True)
    if before != len(df):
        print(f"\n⚠️  Dropped {before - len(df)} empty rows")

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df.to_csv(MASTER_CSV, index=False)

    print("\n" + "="*50)
    print(f"✅ Master CSV saved: {MASTER_CSV}")
    print(f"📊 Total samples  : {len(df)}")
    print(f"   Scam  (1)      : {(df['label']==1).sum()}")
    print(f"   Normal (0)     : {(df['label']==0).sum()}")
    print(f"\n   By data type:")
    print(df.groupby('data_type')['label'].count().to_string())

    return df


# ==============================================
# STEP 4: VALIDATE
# ==============================================

def validate_master_csv():
    df             = pd.read_csv(MASTER_CSV)
    df['word_count'] = df['content'].str.split().str.len()

    print("\n" + "="*50)
    print("📋 VALIDATION REPORT")
    print("="*50)
    print(f"Total rows    : {len(df)}")
    print(f"NaN content   : {df['content'].isna().sum()}")
    print(f"Empty content : {(df['content'].str.strip()=='').sum()}")
    print(f"\nWord count by class:")
    print(df.groupby('label')['word_count'].describe().round(1).to_string())

    # Check if scammer word still present
    has_scammer = df['content'].str.contains('scammer', case=False).sum()
    print(f"\n✅ Rows still containing 'scammer': {has_scammer} (should be 0)")

    short = df[df['word_count'] < 50]
    if len(short) > 0:
        print(f"\n⚠️  {len(short)} samples under 50 words:")
        print(short[['source_file', 'label', 'word_count']].to_string())
    else:
        print("\n✅ All samples have 50+ words")


# ==============================================
# MAIN
# ==============================================

if __name__ == "__main__":
    print("="*50)
    print("STEP 1: Cleaning raw MagicHub transcripts")
    print("="*50)
    clean_raw_magichub()

    print("\n" + "="*50)
    print("STEP 2: Building master CSV")
    print("="*50)
    df = build_master_csv()

    print("\n" + "="*50)
    print("STEP 3: Validating output")
    print("="*50)
    validate_master_csv()

    print("\n✅ Data preparation complete.")
    print("➡️  Next: run 04_tfidf_model.py")