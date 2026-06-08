# ==============================================
# 02_generate_synthetic.py
# Generates synthetic normal transcripts
# Type 1: 30 casual conversations
# Type 2: 50 call centre interactions
# Resume-safe: skips already generated files
# IMPORTANT: Disclose synthetic data in report
# ==============================================

import os
import re
import time
from openai import OpenAI
from config import CLEANED_CONVO_FOLDER, CLEANED_CALLCENTER_FOLDER

# ==============================================
# TARGETS
# ==============================================
TARGET_CASUAL     = 30
TARGET_CALLCENTRE = 50

# ==============================================
# TOPICS
# ==============================================
CASUAL_TOPICS = [
    "two friends catching up over the phone about their health and family",
    "a mother calling her child to discuss weekend plans and grocery shopping",
    "two colleagues discussing a work project deadline and task assignments",
    "a student calling a friend to discuss university assignments and exams",
    "two neighbors chatting about a community event happening this weekend",
    "a person calling a friend to plan a birthday surprise party",
    "two friends discussing a recent Malay drama series they are watching",
    "a person calling their sibling to discuss their parents visiting from kampung",
    "two friends talking about food and planning to meet at a restaurant",
    "a young adult calling home to update parents about their new job",
    "two friends discussing their plans for the upcoming Raya holiday",
    "a person calling their parents to ask for advice about a job offer",
    "two university friends catching up after not seeing each other for months",
    "a person calling a friend to ask for recommendations for a good mechanic",
    "two friends discussing their weekend hiking trip to Bukit Tabur",
    "a mother calling her daughter to check if she arrived safely at university",
    "two friends talking about a new cafe they want to try in Bangsar",
    "a person calling their cousin to discuss a family wedding coming up",
    "two colleagues chatting about their team lunch plans for the week",
    "a student calling home to tell parents about their exam results",
    "two friends discussing which movie to watch at the cinema this weekend",
    "a person calling a friend to cancel plans and reschedule for another day",
    "two friends talking about their fitness goals and gym routines",
    "a person calling their sibling to plan a surprise for their parents anniversary",
    "two friends discussing their experience working from home",
    "a father calling his son to remind him about a family gathering this Sunday",
    "two friends chatting about a local football match they watched together",
    "a person calling a friend to get advice about moving to a new apartment",
    "two friends discussing their Deepavali open house plans",
    "a person calling their best friend to share exciting news about a promotion"
]

CALLCENTRE_TOPICS = [
    # Banking & Finance
    "a customer calling Maybank to check account balance and recent transactions",
    "a customer calling CIMB to ask about a failed online transfer and get it resolved",
    "a customer calling RHB to inquire about applying for a personal loan",
    "a customer calling Bank Islam to ask about opening a savings account",
    "a customer calling AmBank to dispute an unrecognized charge on their credit card",
    "a customer calling Hong Leong Bank to request a new debit card after losing theirs",
    "a customer calling a bank to ask about fixed deposit interest rates",
    "a customer calling KWSP to ask about EPF savings balance and withdrawal eligibility",
    "a customer calling PTPTN to ask about loan repayment schedule and outstanding balance",
    "a customer calling LHDN to ask about filing income tax for the first time",
    # Telco
    "a customer calling Maxis to report internet connection issue and request technician visit",
    "a customer calling Celcom to inquire about phone bill and data plan upgrade",
    "a customer calling Digi to ask about roaming charges for upcoming trip to Thailand",
    "a customer calling TM Unifi to report slow internet speed and request line check",
    "a customer calling Yes 5G to ask about available home broadband packages",
    # Utilities & Government
    "a customer calling TNB to report a power outage and check restoration time",
    "a customer calling Air Selangor to report water supply disruption and ask for update",
    "a customer calling Indah Water to ask about sewerage bill payment options",
    "a customer calling JPJ to ask about renewing driving licence and required documents",
    "a customer calling MyEG to ask about renewing road tax and documents required",
    "a customer calling JPN to ask about replacing a lost MyKad",
    "a customer calling Jabatan Imigresen to ask about passport renewal appointment",
    # Delivery & E-commerce
    "a customer calling Pos Malaysia to track a parcel and reschedule delivery",
    "a customer calling J&T Express to ask why their parcel has been delayed",
    "a customer calling Lazada customer service to return a damaged item received",
    "a customer calling Shopee to ask about a refund status for a cancelled order",
    "a customer calling DHL to ask about customs clearance for an international parcel",
    # Healthcare
    "a customer calling a hospital to book a doctor appointment and ask about clinic hours",
    "a customer calling a clinic to ask about vaccination slots available",
    "a customer calling a pharmacy to check if a specific medication is in stock",
    "a customer calling a dental clinic to book a teeth cleaning appointment",
    "a customer calling a private hospital to ask about health screening packages",
    # Transportation & Travel
    "a customer calling Grab support to dispute an incorrect charge on a recent ride",
    "a customer calling AirAsia to reschedule a domestic flight booking",
    "a customer calling KTM to ask about train schedule from KL to Ipoh",
    "a customer calling a car rental company to book a vehicle for the weekend",
    "a customer calling a Proton service centre to book a car service appointment",
    "a customer calling a workshop to ask about repair cost for a car breakdown",
    # Hospitality & Services
    "a customer calling a hotel in KL to make room reservation and ask about facilities",
    "a customer calling a restaurant to make a reservation for a birthday dinner",
    "a customer calling a hair salon to book an appointment for the weekend",
    "a customer calling a gym to ask about membership packages and trial sessions",
    # Education & Housing
    "a customer calling a school to enquire about registration for a new student",
    "a customer calling a university to ask about postgraduate application deadlines",
    "a customer calling a tuition centre to ask about available classes and fees",
    "a customer calling a property agent to ask about renting an apartment in Subang",
    "a customer calling a plumber to report a pipe leak and ask for urgent visit",
    # Insurance & Others
    "a customer calling AIA insurance to ask about medical card coverage and claims",
    "a customer calling Prudential to ask about life insurance premium payment",
    "a customer calling a courier company to change delivery address for pending shipment"
]

# ==============================================
# SYSTEM PROMPTS
# ==============================================
CASUAL_SYSTEM_PROMPT = """You are generating realistic Malay phone conversation transcripts
for an academic NLP research dataset.

Rules:
- Write ONLY the conversation dialogue, no labels or headers
- Use natural Bahasa Malaysia with occasional English words (code-switching is normal)
- Write at least 300 words
- Format as continuous flowing conversation text, NOT a script with speaker labels
- Completely ordinary conversation — no urgency, no financial requests,
  no authority figures, no threats, no requests for personal information
- Use natural fillers like 'lah', 'kan', 'eh', 'tapi', 'memang' naturally
- Return ONLY the conversation text, nothing else"""

CALLCENTRE_SYSTEM_PROMPT = """You are generating realistic Malaysian call centre phone
conversation transcripts for an academic NLP research dataset.

Rules:
- Write ONLY the conversation dialogue, no labels or headers
- Use natural Bahasa Malaysia with occasional English words
- Write at least 300 words
- Format as continuous flowing text, NOT a script with speaker labels
- Agent must be HELPFUL, POLITE, PROFESSIONAL
- Agent asks for IC number or account number for verification ONLY
- Agent NEVER asks for TAC, OTP, PIN, password or money transfer
- Agent NEVER tells customer their account is frozen or under investigation
- Agent NEVER creates urgency or threatens consequences
- Include realistic elements: hold music, verification, case reference numbers
- Use fillers like 'lah', 'encik', 'puan', 'sebentar', 'boleh saya semak' naturally
- Return ONLY the conversation text, nothing else"""


# ==============================================
# HELPERS
# ==============================================
def count_files(folder, prefix):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder)
                if f.startswith(prefix) and f.endswith('.txt')])


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_batch(folder, prefix, topics, system_prompt, target, client):
    os.makedirs(folder, exist_ok=True)
    existing = count_files(folder, prefix)
    needed   = max(0, target - existing)

    print(f"\n{'='*50}")
    print(f"Type       : {prefix}")
    print(f"Folder     : {folder}")
    print(f"Existing   : {existing} | Target : {target} | To generate : {needed}")
    print(f"{'='*50}")

    if needed == 0:
        print(f"✅ Already have {target} {prefix} transcripts. Skipping.")
        return 0

    topics_to_use = (topics * 5)[:needed]
    success       = 0

    for i, topic in enumerate(topics_to_use):
        filename    = f"{prefix}_{existing + i + 1:02d}.txt"
        output_path = os.path.join(folder, filename)

        if os.path.exists(output_path):
            print(f"  ⏭️  Already exists: {filename}")
            success += 1
            continue

        print(f"\n  [{i+1}/{needed}] {topic[:65]}...")

        try:
            response = client.chat.completions.create(
                model    = "gpt-4o-mini",
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Generate a realistic Malay phone conversation about: {topic}"}
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

    return success


# ==============================================
# MAIN
# ==============================================
if __name__ == "__main__":
    client = OpenAI()

    print("="*50)
    print("SYNTHETIC DATA GENERATION")
    print("="*50)

    casual_done = generate_batch(
        folder        = CLEANED_CONVO_FOLDER,
        prefix        = "synthetic_normal",
        topics        = CASUAL_TOPICS,
        system_prompt = CASUAL_SYSTEM_PROMPT,
        target        = TARGET_CASUAL,
        client        = client
    )

    callcentre_done = generate_batch(
        folder        = CLEANED_CALLCENTER_FOLDER,
        prefix        = "callcentre",
        topics        = CALLCENTRE_TOPICS,
        system_prompt = CALLCENTRE_SYSTEM_PROMPT,
        target        = TARGET_CALLCENTRE,
        client        = client
    )

    print("\n" + "="*50)
    print("GENERATION COMPLETE")
    print(f"✅ Casual generated      : {casual_done}")
    print(f"✅ Call centre generated : {callcentre_done}")
    print("   Casual conversations — synthetically generated using GPT-4o-mini")
    print("   Call centre calls    — synthetically generated using GPT-4o-mini")
    print("\n➡️  Now run 01_data_preparation.py to rebuild master CSV")