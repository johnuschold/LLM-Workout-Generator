import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import math
import re

import streamlit as st
from huggingface_hub import login


# LLM Configuration
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

  # finds .env even when cwd changes

hf_key = (
    st.secrets.get("HF_API_KEY")  # Streamlit Cloud / secrets.toml
    or os.getenv("HF_API_KEY")    # .env or real env var  # alt name supported by HF
)

if not hf_key:
    st.error("Missing HF_API_KEY (or HUGGINGFACE_HUB_TOKEN). Add it to .env or Streamlit secrets.")
    st.stop()

# Either login() explicitly...
login(token=hf_key)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {DEVICE}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_tokenizer = None
_model = None
_text_generation_pipeline = None


def get_llm_pipeline():
    """Initializes and returns the Hugging Face text generation pipeline, loaded only once."""
    global _tokenizer, _model, _text_generation_pipeline

    if _text_generation_pipeline is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
            _tokenizer.pad_token = _tokenizer.eos_token

            _model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto"
            )

            _text_generation_pipeline = pipeline(
                "text-generation",
                model=_model,
                tokenizer=_tokenizer,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.3,
                top_p=0.4,
                repetition_penalty=1.1,
            )
            print("LLM pipeline successfully initialized.")
        except Exception as e:
            print(f"Error during LLM pipeline initialization: {e}")
            _text_generation_pipeline = None
    return _text_generation_pipeline


def create_workout_prompt(user_goals, fitness_level, split, equipment, retrieved_exercises, workout_days,
                          target_muscles, special_instructions=None):
    """Generates a comprehensive prompt for the LLM to create a personalized workout plan."""
    WORKOUT_SPLITS = {
        "Full-Body Split": {
            "frequency": "2 or 4 days/week", "focus": "Whole body each session", "default_days": 3,
            "adjustable": True, "allowed_days": [2, 4], "target_muscles": ["full body"]
        },
        "Upper/Lower Split": {
            "frequency": "2 or 4 days/week", "focus": "Alternating upper and lower body days", "default_days": 4,
            "adjustable": True, "allowed_days": [2, 4], "target_muscles": ["upper body", "lower body"]
        },
        "Push/Pull/Legs (PPL) Split": {
            "frequency": "3 or 6 days/week", "focus": "Push = Chest, shoulders, triceps; Pull = Back, biceps; Legs = Quads, hamstrings, glutes, calves",
            "default_days": 6, "adjustable": True, "allowed_days": [3, 6],
            "target_muscles": ["chest", "shoulders", "triceps", "back", "biceps", "legs", "quads", "hamstrings", "glutes", "calves"]
        },
        "Bro Split (Body Part Split)": {
            "frequency": "5 days/week", "focus": "One major muscle group per day", "default_days": 5,
            "adjustable": False, "allowed_days": [5], "target_muscles": ["chest", "back", "shoulders", "arms", "legs"]
        },
        "Hybrid/PHUL (Power Hypertrophy Upper Lower)": {
            "frequency": "4 days/week", "focus": "Strength + hypertrophy", "default_days": 4,
            "adjustable": False, "allowed_days": [4], "target_muscles": ["upper body", "lower body", "full body"]
        },
        "Push/Pull Split (No Legs)": {
            "frequency": "4 days/week (upper body focused)", "focus": "Push and pull movements only", "default_days": 4,
            "adjustable": False, "allowed_days": [4], "target_muscles": ["chest", "shoulders", "triceps", "back", "biceps"]
        }
    }

    workout_info = WORKOUT_SPLITS.get(split)
    if not workout_info:
        raise ValueError(f"Unknown workout split: {split}")

    parsed_exercises = []
    for ex_str in retrieved_exercises:
        match = re.search(r"The '(.*?)' is a '.*?' level '(.*?)' exercise", ex_str)
        if match:
            exercise_name = match.group(1).strip()
            exercise_type = match.group(2).strip()
            parsed_exercises.append(f"{exercise_name} ({exercise_type})")
        else:
            parsed_exercises.append(ex_str.split(',')[0].strip())

    exercise_context = "\n".join([f"- {ex}" for ex in parsed_exercises])
    if not exercise_context:
        exercise_context = "No specific exercises were retrieved, please suggest general exercises."

    equipment_str = ", ".join(equipment) if equipment else "bodyweight only"

    min_exercises_per_day, max_exercises_per_day = 4, 7
    if fitness_level == "beginner":
        min_exercises_per_day, max_exercises_per_day = 3, 5
    elif fitness_level == "intermediate":
        min_exercises_per_day, max_exercises_per_day = 4, 6
    elif fitness_level == "advanced":
        min_exercises_per_day, max_exercises_per_day = 5, 7

    if user_goals == "weight loss":
        min_exercises_per_day += 1
    elif user_goals == "health":
        min_exercises_per_day += 2

    cardio_duration_minutes = "0 minutes"
    if user_goals == "weight loss":
        cardio_duration_minutes = {"beginner": "20-30 minutes", "intermediate": "30-45 minutes", "advanced": "45-60 minutes"}[fitness_level]
    elif user_goals == "health":
        cardio_duration_minutes = {"beginner": "10-15 minutes", "intermediate": "15-20 minutes", "advanced": "20-25 minutes"}[fitness_level]

    daily_workout_breakdown = ""
    if split == "Upper/Lower Split":
        days_sequence = ["Upper Body", "Lower Body"] * math.ceil(workout_days / 2)
        workout_day_types = days_sequence[:workout_days]
        for i, day_type in enumerate(workout_day_types):
            daily_workout_breakdown += f"    * Day {i + 1}: {day_type} (Focus: {day_type.lower()} muscles)\n"
    elif split == "Push/Pull/Legs (PPL) Split":
        days_sequence = ["Push", "Pull", "Legs"] * math.ceil(workout_days / 3)
        workout_day_types = days_sequence[:workout_days]
        day_mapping = {
            "Push": "Chest, Shoulders, Triceps",
            "Pull": "Back, Biceps",
            "Legs": "Quads, Hamstrings, Glutes, Calves"
        }
        for i, day_type in enumerate(workout_day_types):
            daily_workout_breakdown += f"    * Day {i + 1}: {day_type} (Focus: {day_mapping[day_type]})\n"
    elif split == "Bro Split (Body Part Split)":
        bro_split_muscles = ["Chest", "Back", "Shoulders", "Arms", "Legs"]
        workout_day_types = bro_split_muscles[:workout_days]
        for i, muscle_group in enumerate(workout_day_types):
            daily_workout_breakdown += f"    * Day {i + 1}: {muscle_group} (Focus: {muscle_group.lower()} muscles)\n"
    elif split == "Hybrid/PHUL (Power Hypertrophy Upper Lower)":
        phul_sequence = ["Upper Power", "Lower Power", "Upper Hypertrophy", "Lower Hypertrophy"]
        workout_day_types = phul_sequence[:workout_days]
        for i, day_type in enumerate(workout_day_types):
            daily_workout_breakdown += f"    * Day {i + 1}: {day_type} (Focus: {'upper body' if 'Upper' in day_type else 'lower body'} muscles, with a {'power' if 'Power' in day_type else 'hypertrophy'} emphasis)\n"
    elif split == "Push/Pull Split (No Legs)":
        push_pull_no_legs_sequence = ["Push", "Pull"] * math.ceil(workout_days / 2)
        workout_day_types = push_pull_no_legs_sequence[:workout_days]
        day_mapping = {
            "Push": "Chest, Shoulders, Triceps",
            "Pull": "Back, Biceps",
        }
        for i, day_type in enumerate(workout_day_types):
            daily_workout_breakdown += f"    * Day {i + 1}: {day_type} (Focus: {day_mapping[day_type]})\n"
    else:
        workout_day_types = ["Full Body"] * workout_days
        for i in range(workout_days):
            daily_workout_breakdown += f"    * Day {i + 1}: Full Body\n"

    workout_structure_guidance = ""
    if user_goals == "muscle gain":
        workout_structure_guidance = f"""
    * **General Structure:** Emphasize progressive overload. For each exercise, specify sets and reps typically within the 6-12 range for hypertrophy, with adequate rest (60-90 seconds).
    * **Prioritization:** Compound movements should be prioritized at the beginning of each workout session.
    * **Exercise Selection:** Aim for {min_exercises_per_day}-{max_exercises_per_day} exercises per day. Give preference to exercises from the "Context" that are tagged for 'muscle building'.
"""
    elif user_goals == "weight loss":
        workout_structure_guidance = f"""
    * **General Structure:** Integrate strength training with a clear cardio component at the end of each session. Strength exercises should typically be in the 8-15 rep range.
    * **Cardio:** Conclude each day's session with a 'cardio' exercise lasting **{cardio_duration_minutes}**. Specify the type of cardio (e.g., "Treadmill run", "Elliptical", "Jumping Jacks").
    * **Exercise Selection:** Aim for {min_exercises_per_day}-{max_exercises_per_day} exercises per day, including the cardio segment.
"""
    elif user_goals == "health":
        workout_structure_guidance = f"""
    * **General Structure:** Each workout day should commence with stretching, include at least 2-3 muscle-building exercises, be followed by additional stretching, and conclude with cardio.
    * **Stretching:** The initial exercise should be a 'stretching' routine (e.g., "Dynamic Warm-up Stretches") with a clear purpose (e.g., "prepare muscles for workout"). Another stretching component (e.g., "Cool-down Static Stretches") should precede the cardio.
    * **Cardio:** The final exercise of each day should be a 'cardio' activity lasting **{cardio_duration_minutes}**.
    * **Exercise Selection:** Aim for {min_exercises_per_day}-{max_exercises_per_day} exercises per day, encompassing stretching and cardio.
"""
    else:
        workout_structure_guidance = f"""
    * **General Structure:** Ensure a logical workout flow, typically including warm-up, main lifts, accessory work, and a cool-down where appropriate.
    * **Exercise Selection:** Target {min_exercises_per_day}-{max_exercises_per_day} exercises per day, chosen to be suitable for the user's level and available equipment.
"""

    special_instructions_section = ""
    if special_instructions and special_instructions.strip():
        special_instructions_section = f"**Special Instructions/Considerations:**\n- {special_instructions.strip()}\n\n"

    muscles_str = ", ".join(target_muscles) if target_muscles else "various muscle groups"

    prompt = f"""
You are an experienced fitness coach and personal trainer. Your primary objective is to craft a comprehensive, personalized workout plan.

**User Profile:**
- **Goal:** {user_goals}
- **Fitness Level:** {fitness_level}
- **Workout Split:** {split} ({workout_days} days per week)
- **Target Muscle Groups (Based on Split):** {muscles_str}
- **Available Equipment:** {equipment_str}

**Context (Relevant Exercises from Database):**
{exercise_context}

{special_instructions_section}

**Instructions:**
Leveraging the provided user profile and relevant exercise context, generate a {workout_days}-day workout plan.

For each day, clearly specify:
1.  **Workout Day:** (e.g., "Day 1: Full Body").
    * Include the structured breakdown of daily workout types as determined by the chosen split:
    {daily_workout_breakdown.strip()}
2.  **Exercises:**
    * For each exercise, detail its **Name**, suggested **Sets**, and **Reps** (e.g., "Squats: 3 sets of 8-12 reps").
    * Prioritize exercises from the "Context" section, but feel free to introduce other appropriate exercises to ensure a well-rounded and effective plan if the context is limited.
    * Avoid uncommon exercises unless specifically requested by the user.

{workout_structure_guidance}

**Example Workout Plan Format:**
Please adhere to the following Markdown format for the output:

### Day 1: Upper Body
**Exercises:**
* Bench Press: 3 sets of 8-12 reps
* Bent-Over Rows: 3 sets of 8-12 reps
* Overhead Press: 3 sets of 10-15 reps
* Bicep Curls: 3 sets of 10-15 reps
* Triceps Pushdowns: 3 sets of 10-15 reps
* Plank: 3 sets, 45-60 seconds hold

### Day 2: Lower Body
**Exercises:**
* Barbell Squats: 3 sets of 6-10 reps
* Romanian Deadlifts: 3 sets of 8-12 reps
* Leg Press: 3 sets of 10-15 reps
* Calf Raises: 3 sets of 15-20 reps
* Lunges: 3 sets of 10-12 reps per leg

**Workout Plan:**


"""
    return prompt


def get_improved_rag_query(original_query: str, muscle_groups: list[str], goal: str, difficulty: str,
                           equipment: str) -> str:
    """Constructs an optimized RAG query, incorporating user goals and respecting negative constraints."""

    muscle_groups_str = ", ".join(muscle_groups) if muscle_groups else "any muscle group"

    llm_prompt = f"""
    You are an expert query refiner for a Retrieval-Augmented Generation (RAG) system that ranks documents by lexical/keyword similarity (bag-of-words). Your job is to produce a single, compact search string of ONLY positive keywords to retrieve exercise documents.
    
    Critical: The system does not understand semantics or negation. If a word appears in the query, results will likely include that word. Therefore:
    - Do NOT include any excluded items from the user's 'original_query'—not even as negations (e.g., "no barbell", "without legs").
    - Do NOT include stems, plural/singular variants, abbreviations, or common synonyms of excluded items.
    - Do NOT include generic negation words (no/avoid/exclude/without) or any “do not include” phrasing at all.
    
    Inputs:
    - User's original request/special instructions: '{original_query}'
    - User's Goal: '{goal}'
    - User's Fitness Level: '{difficulty}'
    - User's Equipment: '{equipment}'
    - Target Muscle Groups: '{muscle_groups_str}'
    
    Instructions:
    1) Extract explicit and implicit negative constraints from 'original_query' (e.g., injuries, equipment to avoid, disliked movements). Internally note them, but NEVER echo them in the output.
    2) Build a positive-only keyword string that emphasizes:
       • Allowed exercise types and movement patterns
       • Target muscle groups (and safe neighboring muscles if needed)
       • Allowed/available equipment only
       • Intended difficulty level / progression
       • Programming tags commonly used in exercise databases (e.g., bodyweight, dumbbell, unilateral, low-impact, beginner, hypertrophy, mobility, warm-up, EMOM, AMRAP) when relevant
    3) Canonicalize terms to how they’re likely indexed (e.g., “glutes” and “gluteus” → “glute”, “hamstrings” → “hamstring”), and include a few high-value synonyms for INCLUDED items only.
    4) Prefer safe alternatives when a region is excluded (e.g., if legs are excluded, favor core, upper-body, low-impact, mobility, etc.), but NEVER mention the excluded region by name.
    5) Keep it concise: ~8–20 tokens, comma- or space-separated, no sentences, no punctuation other than commas/spaces.
    
    Output format:
    Return ONLY the refined keyword query string (no preface or trailing text).
    
    Improved prompt:
    """

    improved_query = generate_text_with_llm(llm_prompt)
    return improved_query


def generate_text_with_llm(prompt: str) -> str:
    """A wrapper function to interact with the pre-loaded LLM pipeline and generate text."""
    llm_pipeline = get_llm_pipeline()
    try:
        result = llm_pipeline(prompt)
        generated_text = result[0]['generated_text']

        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text.strip()
    except Exception as e:
        print(f"Error during LLM text generation: {e}")
        return "An error occurred while generating the plan. Please try again or adjust your inputs."