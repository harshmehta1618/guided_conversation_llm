import json
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# DATA LOADERS
# ============================================================
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# SCHEMA UTILITIES
# ============================================================
def get_simplified_map(data):
    """Flatten user_attributes schema into {category: [{attr_id: display_name}]}."""
    categories = data.get("user_attribute_schema", {}).get("categories", {})
    final_output = {}
    for category_key, category_data in categories.items():
        attribute_list = []
        attributes = category_data.get("attributes", {})
        for attr_key, attr_details in attributes.items():
            if isinstance(attr_details, dict):
                display_name = attr_details.get("name", attr_key)
                attribute_list.append({attr_key: display_name})
            elif isinstance(attr_details, list):
                formatted_name = attr_key.replace("_", " ").title()
                attribute_list.append({attr_key: formatted_name})
        final_output[category_key] = attribute_list
    return final_output


def parse_llm_json_output(llm_response_text):
    clean_text = re.sub(r"```json\s*|\s*```", "", llm_response_text).strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        print(f"  [warn] JSON parse error: {e}")
        return None


# ============================================================
# SEMANTIC SEARCH (PHASE 0) — TOP-K MATCHING
# ============================================================
def build_problem_index(problems_db, embed_model):
    """
    Pre-compute embeddings for every synonym in the problems DB.
    Each synonym is concatenated with the problem's description so the
    embedding has richer semantic context than a bare keyword.
    """
    index = []
    for problem in problems_db.get("problems", []):
        synonyms = problem.get("synonyms", [])
        description = problem.get("description", "")
        if synonyms:
            texts = [f"{syn}. {description}".strip(". ") for syn in synonyms]
            embeddings = embed_model.encode(texts, convert_to_numpy=True)
            for synonym, emb in zip(synonyms, embeddings):
                index.append((problem, synonym, emb))
    return index


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def search_top_problems(user_messages, problems_db, embed_model, top_k=3):
    """
    Embed each user symptom message individually and compare against every
    synonym embedding. For each problem, take the BEST (max) similarity over
    all (message, synonym) pairs. Return the top_k problems by score.
    """
    if not user_messages:
        return []

    index = build_problem_index(problems_db, embed_model)
    user_embs = embed_model.encode(user_messages, convert_to_numpy=True)

    best_per_problem = {}
    for problem, synonym, syn_emb in index:
        pid = problem["id"]
        for u_emb, u_msg in zip(user_embs, user_messages):
            score = cosine_similarity(u_emb, syn_emb)
            if pid not in best_per_problem or score > best_per_problem[pid]["score"]:
                best_per_problem[pid] = {
                    "problem": problem,
                    "best_synonym": synonym,
                    "best_message": u_msg,
                    "score": score,
                }

    ranked = sorted(best_per_problem.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


# ============================================================
# CONVERSATION
# ============================================================
END_KEYWORDS = {
    "done", "no", "nope", "nothing", "nothing else", "that's all", "thats all",
    "finish", "finished", "stop", "exit", "quit", "no more", "i'm good",
    "im good", "that is all", "all good"
}


def is_user_done(user_msg):
    msg = user_msg.strip().lower()
    if not msg:
        return True
    return msg in END_KEYWORDS


def llm_conversational_turn(conversation_history, llm_chat):
    """Generate the assistant's next message given full conversation history."""
    system_prompt = (
        "You are a compassionate Ayurvedic health consultant gathering symptom "
        "information from a user.\n\n"
        "YOUR JOB on each turn:\n"
        "1. Briefly acknowledge what the user just said with empathy (1 short sentence).\n"
        "2. If their description is vague, ask ONE focused follow-up question about that symptom.\n"
        "3. Otherwise, ask if they have any OTHER symptoms or concerns to share.\n\n"
        "RULES:\n"
        "- Keep responses to 2-3 short sentences.\n"
        "- DO NOT diagnose. DO NOT recommend treatments.\n"
        "- DO NOT repeat what the user said verbatim.\n"
        "- Be warm but concise."
    )

    messages = [("system", system_prompt)]
    for turn in conversation_history:
        role = "human" if turn["role"] == "user" else "ai"
        messages.append((role, turn["content"]))

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm_chat
    response = chain.invoke({})
    return response.content.strip()


# ============================================================
# PERSONA BUILDING (INCREMENTAL)
# ============================================================
def update_persona(existing_persona, new_message, simplified_schema, llm_json):
    """
    Incrementally update the user persona with information from a new message.
    Existing entries are preserved; new info is merged in.
    """
    prompt_template = """You are an Ayurvedic medical data parser building a structured user persona turn-by-turn.

EXISTING USER PERSONA (built from earlier turns):
{existing_persona}

NEW USER MESSAGE:
"{new_message}"

SCHEMA (allowed categories and attribute keys — use ONLY these):
{schema}

INSTRUCTIONS:
1. Read the new message and identify any health-related facts, symptoms, or measurements.
2. Map each fact to the appropriate attribute in the schema.
3. MERGE with the existing persona: keep all prior entries intact unless the user explicitly contradicts them.
4. Add new entries for anything new mentioned. If no new info, return the existing persona unchanged.
5. Use ONLY the attribute keys defined in the schema. Do not invent new keys.

OUTPUT (strict JSON, no markdown):
{{"persona": {{"category_name": [{{"attribute_id": "value or description"}}, ...], ...}}}}
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm_json

    try:
        response = chain.invoke({
            "existing_persona": json.dumps(existing_persona, indent=2),
            "new_message": new_message,
            "schema": json.dumps(simplified_schema, indent=2),
        })
        result = parse_llm_json_output(response.content)
        if result and "persona" in result:
            return result["persona"]
        if isinstance(result, dict):
            return result
        return existing_persona
    except Exception as e:
        print(f"  [warn] persona update failed: {e}")
        return existing_persona


# ============================================================
# RECOMMENDATION (FINAL STAGE)
# ============================================================
def recommend_plan(persona, top_problems, tasks_db, llm_json):
    """
    Synthesize a personalized plan using the filled persona, the top detected
    problems (from semantic search), and a FOCUSED task subset pulled from
    tasks_by_problem for only the detected problem IDs.
    """
    problems_summary = []
    for p in top_problems:
        problems_summary.append({
            "id": p["problem"]["id"],
            "primary_term": p["problem"]["primary_term"],
            "iks_term": p["problem"].get("iks_term"),
            "category": p["problem"].get("category"),
            "similarity_score": round(p["score"], 4),
            "matched_via_synonym": p["best_synonym"],
            "relevant_attribute_paths": p["problem"].get("relevant_attribute_paths", []),
            "exploitation_rules": p["problem"].get("exploitation_rules", {}),
        })

    # Focused task subset: only the tasks for the detected problem IDs
    tasks_by_problem = tasks_db.get("tasks_by_problem", {})
    focused_tasks = {
        p["problem"]["id"]: tasks_by_problem.get(p["problem"]["id"], {})
        for p in top_problems
    }

    system_prompt = """You are an expert Ayurvedic Health Planner. Synthesize a personalized plan for the user.

INPUT YOU WILL RECEIVE:
1. User Persona — structured attributes describing the user's current state (filled from a turn-by-turn conversation).
2. Detected Problems — top 2-3 conditions identified from semantic search. Each has 'relevant_attribute_paths' (which persona attributes matter most) and 'exploitation_rules' (diagnostic hints).
3. Focused Task Pool — Ayurvedic tasks pre-curated for the detected problems (keyed by problem ID).

YOUR REASONING PROCESS:
1. For EACH detected problem:
   - Cross-reference the user's persona attributes against its 'relevant_attribute_paths'.
   - Apply the 'exploitation_rules' to determine the Dosha imbalance driving this problem for THIS user.
   - Pick 3-4 tasks from the problem's focused task list that address the imbalance.
   - Justify each task using the user's specific persona attributes.
2. Add 1-2 cross-cutting tasks if multiple problems share a root cause (e.g. stress aggravating both digestion and sleep).

OUTPUT FORMAT (strict JSON only — no markdown):
{{
  "analysis_summary": "Short paragraph on user's overall constitution and main imbalances",
  "per_problem_plan": [
    {{
      "problem": "Primary term",
      "iks_term": "Sanskrit term",
      "dosha_assessment": "Which dosha is involved and why, citing persona attributes",
      "recommended_tasks": [
        {{"task": "...", "category": "...", "reason": "..."}}
      ]
    }}
  ],
  "cross_cutting_tasks": [
    {{"task": "...", "addresses": ["problem1", "problem2"], "reason": "..."}}
  ]
}}"""

    human_prompt = (
        "User Persona:\n{persona}\n\n"
        "Detected Problems (top from semantic search):\n{problems}\n\n"
        "Focused Task Pool (curated per detected problem ID):\n{focused_tasks}\n\n"
        "Generate the personalized plan as instructed."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    chain = prompt | llm_json | JsonOutputParser()

    try:
        return chain.invoke({
            "persona": json.dumps(persona, indent=2),
            "problems": json.dumps(problems_summary, indent=2),
            "focused_tasks": json.dumps(focused_tasks, indent=2),
        })
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# CONVERSATION DRIVER
# ============================================================
def run_conversation(llm_chat, llm_json, simplified_schema):
    print("\n========= AYURVEDIC HEALTH CONSULTATION =========")
    print(
        "Assistant: Namaste! Please tell me about the symptoms or health "
        "concerns you've been experiencing.\n"
        "(Type 'done' or 'no' when you're finished sharing.)"
    )

    conversation_history = []
    user_messages = []
    persona = {}
    turn_index = 1

    while True:
        try:
            user_input_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Conversation interrupted]")
            break

        if is_user_done(user_input_text):
            print("\nAssistant: Thank you for sharing. Let me analyze your symptoms now...")
            break

        conversation_history.append({"role": "user", "content": user_input_text})
        user_messages.append(user_input_text)

        print(f"  [Turn {turn_index}] Updating persona with new info...")
        persona = update_persona(persona, user_input_text, simplified_schema, llm_json)

        try:
            assistant_response = llm_conversational_turn(conversation_history, llm_chat)
        except Exception as e:
            assistant_response = f"(Could not generate response: {e}). Anything else?"
        conversation_history.append({"role": "assistant", "content": assistant_response})
        print(f"\nAssistant: {assistant_response}")
        turn_index += 1

    return user_messages, persona


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY environment variable is not set. Execution will fail.")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Initializing LLMs...")
    try:
        llm_chat = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.5,
        )
        llm_json = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
    except Exception as e:
        print(f"Could not initialize Groq LLMs: {e}")
        return

    print("Loading data files...")
    schema = load_json('user_attributes.json')
    simplified_schema = get_simplified_map(schema)
    problems_db = load_json('problems.json')
    tasks_db = load_json('tasks.json')

    # ============== STAGE 1: Turn-by-turn Conversation ==============
    user_messages, persona = run_conversation(llm_chat, llm_json, simplified_schema)

    if not user_messages:
        print("\nNo symptoms collected. Exiting.")
        return

    print("\n================ ANALYSIS ================")
    print("\n--- Final User Persona (built incrementally) ---")
    print(json.dumps(persona, indent=2))

    # ============== STAGE 2: Top-K Semantic Search ==============
    print("\n--- Top Detected Problems (Semantic Search) ---")
    top_problems = search_top_problems(user_messages, problems_db, embed_model, top_k=3)
    if not top_problems:
        print("No problems detected.")
        return
    for p in top_problems:
        print(
            f"  [{p['score']:.4f}] {p['problem']['primary_term']}  "
            f"(matched \"{p['best_synonym']}\" against \"{p['best_message']}\")"
        )

    # ============== STAGE 3: Personalized Plan Generation ==============
    print("\n--- Generating Personalized Plan ---")
    plan = recommend_plan(persona, top_problems, tasks_db, llm_json)
    print("\n================ PERSONALIZED HEALTH PLAN ================")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
