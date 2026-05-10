import json
import pickle
import re
import time
import hashlib
import os
from typing import List

import openai

# ================= CONFIG ================= #
VERBOSE = True
MAX_RETRIES = 3
# ========================================== #

sid = None
symptom_list = None
client = None
GROQ_MODEL = None

# ================= SYNONYMS ================= #
SYNONYMS = {
    # Original
    "anxiety": "anxiety and nervousness",
    "nervousness": "anxiety and nervousness",
    "breathlessness": "shortness of breath",
    "breathing difficulty": "shortness of breath",
    "chest pain": "sharp chest pain",
    "low mood": "depression",
    "feeling low": "depression",
    "tired": "fatigue",
    "weak": "fatigue",
    "palpitations": "palpitations",
    # Respiratory
    "breathless": "shortness of breath",
    "persistent cough": "cough",
    "persistent dry cough": "cough",
    "dry cough": "cough",
    "chronic cough": "cough",
    "coughing blood": "hemoptysis",
    "coughing up blood": "hemoptysis",
    # Pain & body
    "body aches": "ache all over",
    "bone pain": "bones are painful",
    "back pain": "back pain",
    "pain in legs": "leg pain",
    "pain in joints": "joint pain",
    "pain": "lower body pain",
    "stiffness": "stiffness all over",
    "difficulty moving": "problems with movement",
    "slow movement": "problems with movement",
    "soreness": "muscle pain",
    "muscle cramps": "muscle cramps, contractures, or spasms",
    # Neurological
    "numbness": "paresthesia",
    "tingling": "paresthesia",
    "seizures": "seizures",
    "confusion": "disturbance of memory",
    "confused": "disturbance of memory",
    "memory problems": "disturbance of memory",
    "tremors": "abnormal involuntary movements",
    "sensitivity to light": "diminished vision",
    "lightheaded": "dizziness",
    # Skin
    "rash": "skin rash",
    "itchy skin": "itching of skin",
    # GI
    "burning sensation": "heartburn",
    "stomach discomfort": "upper abdominal pain",
    "bloating": "stomach bloating",
    "bloated": "stomach bloating",
    "abdominal pain": "sharp abdominal pain",
    # Mental health
    "persistent low mood": "depression",
    "lack of interest": "depression",
    "trouble sleeping": "insomnia",
    "restless": "restlessness",
    "racing thoughts": "anxiety and nervousness",
    "feel restless": "restlessness",
    # Cardiovascular
    "chest discomfort": "chest tightness",
    "irregular heartbeat": "irregular heartbeat",
    "faint": "fainting",
    "swelling": "leg swelling",
    # Metabolic / systemic
    "excessive thirst": "thirst",
    "dehydration": "fluid retention",
    "severe dehydration": "fluid retention",
    "low blood pressure": "dizziness",
    "heat intolerance": "feeling hot",
    "cold intolerance": "feeling cold",
    "high temperature": "fever",
    "high fever": "fever",
    "weight loss": "recent weight loss",
    "weight gain": "weight gain",
    "extreme fatigue": "fatigue",
    # Additional
    "severe abdominal pain": "sharp abdominal pain",
    "pain in my joints": "joint pain",
    "pain in joints": "joint pain",
    "faintness": "fainting",
    "severe headache": "headache",
    "frequent urination": "frequent urination",
    "joint stiffness": "stiffness all over",
    "sore throat": "sore throat",
}

# ================= INIT ================= #
def init_extractor():
    global sid, symptom_list, client, GROQ_MODEL

    if sid is not None:
        return

    with open("SID.p", "rb") as f:
        sid = pickle.load(f)

    symptom_list = list(sid.keys())

    GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    # 🔥 FIXED API KEY HANDLING
    client = openai.OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

# ================= CLEAN ================= #
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ================= SPLIT ================= #
def split_symptoms(symptoms):
    final = []

    for s in symptoms:
        parts = re.split(r'\band\b', s)

        for p in parts:
            p = p.strip()
            if p:
                final.append(p)

    return final

# ================= FILTER ================= #
def filter_noise(symptoms):
    return [s for s in symptoms if len(s) >= 3]

# ================= LLM EXTRACTION ================= #
llm_cache = {}

def extract_symptoms_llm(conversation_text: str):

    key = hashlib.md5(conversation_text.encode()).hexdigest()
    if key in llm_cache:
        return llm_cache[key]

    # 🔥 FIXED PROMPT
    prompt = f"""
Extract symptoms from the conversation.

STRICT RULES:
- Extract ONLY symptoms explicitly mentioned
- DO NOT infer or add new symptoms
- DO NOT hallucinate
- Return EACH symptom separately
- Split combined phrases:
  "anxiety and dizziness" → ["anxiety", "dizziness"]
- Keep medical phrases intact:
  "shortness of breath" stays as is

Return ONLY JSON:
{{"s_pos": [...], "s_neg": [...]}}

Conversation:
{conversation_text}
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )

            text = response.choices[0].message.content.strip()

            if VERBOSE:
                print(f"🔧 RAW LLM: {text[:200]}")

            text = re.sub(r'```json|```', '', text)

            data = json.loads(text)

            if VERBOSE:
                print(f"✅ Parsed JSON: {data}")

            # 🔥 CLEAN + SPLIT + FILTER
            s_pos = split_symptoms([clean_text(s) for s in data.get("s_pos", [])])
            s_neg = split_symptoms([clean_text(s) for s in data.get("s_neg", [])])

            s_pos = filter_noise(s_pos)
            s_neg = filter_noise(s_neg)

            if VERBOSE:
                print(f"📝 Final Symptoms: pos={s_pos}, neg={s_neg}")

            result = {"s_pos": s_pos, "s_neg": s_neg}
            llm_cache[key] = result

            return result

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            time.sleep(1)

    print("❌ Extraction failed")
    return {"s_pos": [], "s_neg": []}

# ================= MATCHING ================= #
def match_symptoms(extracted: List[str]):

    matched = []

    for s in extracted:

        # 1️⃣ EXACT
        if s in symptom_list:
            matched.append(s)
            if VERBOSE:
                print(f"{s} → {s} (exact)")
            continue

        # 2️⃣ SYNONYM
        if s in SYNONYMS:
            mapped = SYNONYMS[s]
            if mapped in symptom_list:
                matched.append(mapped)
                if VERBOSE:
                    print(f"{s} → {mapped} (synonym)")
                continue

        # 3️⃣ SAFE SUBSTRING
        found = False

        for known in symptom_list:

            if len(s) >= 5:

                if s in known:
                    matched.append(known)
                    if VERBOSE:
                        print(f"{s} → {known} (substring)")
                    found = True
                    break

                if len(known) >= 5 and known in s:
                    matched.append(known)
                    if VERBOSE:
                        print(f"{s} → {known} (reverse substring)")
                    found = True
                    break

        if found:
            continue

        # 4️⃣ NO MATCH
        if VERBOSE:
            print(f"❌ No match for: {s}")

    return list(set(matched))

# ================= MAIN ================= #
def extract_patient_symptoms(conversation_text, self_report_text="", conversation_turns=None):

    init_extractor()

    res = extract_symptoms_llm(conversation_text)

    s_pos = match_symptoms(res["s_pos"])
    s_neg = match_symptoms(res["s_neg"])

    if len(s_pos) < 1:
        print("⚠️ Weak extraction")

    return {
        "s_pos": s_pos,
        "s_neg": s_neg,
        "sid_pos": [sid[s] for s in s_pos if s in sid],
        "sid_neg": [sid[s] for s in s_neg if s in sid]
    }