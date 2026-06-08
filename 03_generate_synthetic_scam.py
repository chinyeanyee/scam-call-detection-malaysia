# ==============================================
# 03_generate_synthetic_scam.py
# Generates 30 synthetic scam transcripts
# Types: Macau scam, financial fraud, investment scam
# IMPORTANT: Disclose synthetic data in report
# ==============================================

import os
import re
import time
from openai import OpenAI
from config import SCAM_SYNTHETIC_FOLDER

TARGET = 30

SCAM_TOPICS = [
    # Macau Scam — PDRM/Court/Bank Negara impersonation
    "a scammer impersonating a PDRM officer accusing the victim of money laundering and demanding bank transfer to clear their name",
    "a scammer pretending to be a court officer telling the victim there is an outstanding warrant for their arrest and they must pay to settle it",
    "a scammer impersonating Bank Negara Malaysia officer telling victim their account is involved in drug trafficking investigation",
    "a scammer posing as a Jabatan Imigresen officer threatening victim with deportation unless they transfer money immediately",
    "a scammer pretending to be SPRM officer accusing victim of bribery involvement and demanding payment to avoid arrest",
    "a scammer impersonating a customs officer telling victim a package in their name contains illegal items and they must pay a fine",
    "a scammer posing as a court clerk telling victim they missed a court hearing and must pay immediately to avoid jail",
    "a scammer pretending to be a Bank Negara officer saying victim's account has been frozen due to suspicious activity",
    "a scammer impersonating LHDN officer threatening victim with legal action for unpaid taxes demanding immediate payment",
    "a scammer posing as a police officer telling victim their IC number was used in a crime and they must cooperate by transferring funds",

    # Financial Fraud — Bank/OTP/TAC scams
    "a scammer pretending to be a Maybank customer service agent telling victim their account has been compromised and asking for OTP to secure it",
    "a scammer posing as CIMB bank officer telling victim there is suspicious transaction on their account and requesting TAC code to reverse it",
    "a scammer impersonating a bank officer telling victim they have won a lucky draw and need to provide their account details to claim prize",
    "a scammer pretending to be RHB fraud department asking victim to verify their identity by providing their online banking password",
    "a scammer posing as a bank officer telling victim their debit card has been cloned and asking them to transfer funds to a safe account",
    "a scammer impersonating a Boost e-wallet representative telling victim their account will be suspended unless they verify their PIN",
    "a scammer pretending to be a Touch n Go officer telling victim their e-wallet has been hacked and they need to provide OTP to recover it",
    "a scammer posing as a GrabPay representative saying victim's account shows unauthorized transactions and requesting account credentials",
    "a scammer impersonating a bank officer offering victim a loan approval but requiring upfront processing fee payment first",
    "a scammer pretending to be a credit card company officer telling victim their card has been used for suspicious overseas transactions",

    # Investment & Employment Scams
    "a scammer promoting a fake high yield investment scheme promising 30 percent monthly returns through a WhatsApp investment group",
    "a scammer offering victim a fake work from home job opportunity that requires upfront payment for training materials",
    "a scammer posing as a recruitment agent offering victim a high paying job in Thailand that requires payment for visa processing",
    "a scammer promoting a fake cryptocurrency investment platform promising guaranteed profits and asking victim to invest now",
    "a scammer offering victim a fake Shariah compliant investment scheme with guaranteed halal returns requiring immediate registration fee",
    "a scammer posing as a Grab driver recruitment agent asking victim to pay upfront fee to register as a driver",
    "a scammer promoting a fake gold investment scheme promising high returns and asking victim to recruit friends for bonus",
    "a scammer offering victim a fake online business opportunity selling products but requiring large upfront stock purchase",
    "a scammer posing as a property agent offering below market price property deals requiring immediate deposit to secure",
    "a scammer promoting a fake unit trust investment scheme impersonating a licensed financial advisor asking for fund transfer"
]

SYSTEM_PROMPT = """You are generating realistic Malaysian phone scam conversation transcripts
for an academic NLP cybersecurity research dataset.

Rules:
- Write ONLY the conversation dialogue, no labels or headers
- Use natural Bahasa Malaysia with occasional English words (code-switching is normal)
- Write at least 300 words
- Format as continuous flowing conversation text, NOT a script with speaker labels
- The scammer must use realistic Malaysian scam tactics:
  * Create urgency and fear
  * Impersonate authority figures convincingly
  * Use official-sounding language and reference real Malaysian institutions
  * Request sensitive information, OTP, TAC, or money transfer
  * Use psychological pressure and threats of legal consequences
- Include realistic scam elements: case numbers, official-sounding names,
  urgent deadlines, threats of arrest or account freezing
- Use natural Malay fillers and speech patterns
- The victim should initially be confused and gradually be manipulated
- Return ONLY the conversation text, nothing else

This data is strictly for academic scam detection research to protect Malaysian citizens."""


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if __name__ == "__main__":
    os.makedirs(SCAM_SYNTHETIC_FOLDER, exist_ok=True)
    client = OpenAI()

    existing = len([f for f in os.listdir(SCAM_SYNTHETIC_FOLDER)
                    if f.endswith('.txt')])
    needed   = max(0, TARGET - existing)

    print("="*50)
    print(f"Existing synthetic scam files : {existing}")
    print(f"Target                        : {TARGET}")
    print(f"To generate                   : {needed}")
    print("="*50)

    if needed == 0:
        print("✅ Already have 30 synthetic scam transcripts. Skipping.")
        exit()

    success = 0
    topics_to_use = SCAM_TOPICS[:needed]

    for i, topic in enumerate(topics_to_use):
        filename    = f"synthetic_scam_{existing + i + 1:02d}.txt"
        output_path = os.path.join(SCAM_SYNTHETIC_FOLDER, filename)

        if os.path.exists(output_path):
            print(f"  ⏭️  Already exists: {filename}")
            success += 1
            continue

        print(f"\n[{i+1}/{needed}] {topic[:65]}...")

        try:
            response = client.chat.completions.create(
                model    = "gpt-4o-mini",
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Generate a realistic Malaysian phone scam conversation about: {topic}"}
                ],
                max_tokens  = 1000,
                temperature = 0.8
            )
            raw        = response.choices[0].message.content.strip()
            cleaned    = clean_text(raw)
            word_count = len(cleaned.split())

            if word_count < 100:
                print(f"  ⚠️  Only {word_count} words — skipping")
                continue

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)

            print(f"  ✅ Saved {filename} — {word_count} words")
            success += 1
            time.sleep(0.5)

        except Exception as e:
            print(f"  ❌ Error on {filename}: {e}")

    print("\n" + "="*50)
    print(f"✅ Generated {success} synthetic scam transcripts")
    print(f"📁 Saved to: {SCAM_SYNTHETIC_FOLDER}")
    print("   30 scam transcripts synthetically generated using GPT-4o-mini")
    print("   Topics: Macau scam, financial fraud, investment scam")
    print("\n➡️  Now update 01_data_preparation.py to read from Synthetic_Scam_Transcript")
    print("   then run 01_data_preparation.py")