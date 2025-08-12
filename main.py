import streamlit as st
import re
import pandas as pd

from LLMgen import get_llm_pipeline, create_workout_prompt, generate_text_with_llm, \
    get_improved_rag_query
from rag_retriever import load_vectorstore, retrieve_similar_documents

# Import thefuzz for fuzzy string matching
from thefuzz import process
from thefuzz import fuzz


# Initialize the LLM pipeline once
@st.cache_resource
def load_llm():
    return get_llm_pipeline()


llm_pipeline = load_llm()
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_PATH_EXERCISE = "./faiss_index_exercises"

st.set_page_config(page_title="AI Fitness Coach", layout="centered")


# Load the FAISS vector store for exercises
@st.cache_resource
def load_vector_stores():
    return load_vectorstore(SAVE_PATH_EXERCISE)


loaded_vectorstore_exercise = load_vector_stores()


# Load exercise data from CSV
@st.cache_resource
def load_exercises_data():
    try:
        df = pd.read_csv('data/exercises_data_final.csv')
        return df
    except FileNotFoundError:
        st.error(
            "Error: 'exercises_data_final.csv' not found. Please ensure the dataset is in the correct directory.")
        return pd.DataFrame()


exercises_df = load_exercises_data()

st.title("💪 AI Fitness Coach")
st.markdown("""
Welcome to your personalized AI fitness coach!
Just tell me your goals and preferences, and I'll generate a custom workout plan for you.
""")

st.sidebar.header("About")
st.sidebar.info(
    "This application leverages an AI model to generate tailored workout plans "
    "based on user input. It serves as a proof-of-concept demonstrating the capabilities of the `llmgen` library and RAG principles."
)
st.sidebar.markdown("---")
st.sidebar.header("How it works")
st.sidebar.markdown("""
- **Workout Plan:** Crafts personalized exercise routines by considering your objectives, current fitness level, and available equipment.
""")

# Define workout split configurations
WORKOUT_SPLITS = {
    "Full-Body Split": {
        "frequency": "2 or 4 days/week",
        "focus": "Whole body each session",
        "default_days": 3,
        "adjustable": True,
        "allowed_days": [2, 4],
        "target_muscles": ["full body"]
    },
    "Upper/Lower Split": {
        "frequency": "2 or 4 days/week",
        "focus": "Alternating upper and lower body days",
        "default_days": 4,
        "adjustable": True,
        "allowed_days": [2, 4],
        "target_muscles": ["upper body", "lower body"]
    },
    "Push/Pull/Legs (PPL) Split": {
        "frequency": "3 or 6 days/week",
        "focus": "Push = Chest, shoulders, triceps; Pull = Back, biceps; Legs = Quads, hamstrings, glutes, calves",
        "default_days": 6,
        "adjustable": True,
        "allowed_days": [3, 6],
        "target_muscles": ["chest", "shoulders", "triceps", "back", "biceps", "legs", "quads", "hamstrings", "glutes",
                           "calves"]
    },
    "Bro Split (Body Part Split)": {
        "frequency": "5 days/week",
        "focus": "One major muscle group per day",
        "default_days": 5,
        "adjustable": False,
        "allowed_days": [5],
        "target_muscles": ["chest", "back", "shoulders", "arms", "legs"]
    },
    "Hybrid/PHUL (Power Hypertrophy Upper Lower)": {
        "frequency": "4 days/week",
        "focus": "Strength + hypertrophy",
        "default_days": 4,
        "adjustable": False,
        "allowed_days": [4],
        "target_muscles": ["upper body", "lower body", "full body"]
    },
    "Push/Pull Split (No Legs)": {
        "frequency": "4 days/week (upper body focused)",
        "focus": "Push and pull movements only",
        "default_days": 4,
        "adjustable": False,
        "allowed_days": [4],
        "target_muscles": ["chest", "shoulders", "triceps", "back", "biceps"]
    }
}


# Utility function to parse generated workout text and extract individual exercise names
def extract_exercise_names(workout_text):
    exercise_names = []
    matches = re.findall(r"^\* (.+?)(?::|\n)", workout_text, re.MULTILINE)
    for match in matches:
        name = match.split(':')[0].strip()
        name = re.sub(r'\s*\d+\s*sets.*$', '', name).strip()
        # Explicitly remove any remaining asterisks from the exercise name
        name = name.replace('**', '').strip()
        exercise_names.append(name)
    return exercise_names


# Function to process the 'Combined_Text' field for clean, bulleted instructions
def get_clean_instructions(combined_text):
    if pd.isna(combined_text) or not combined_text.strip():
        return "No instructions available."

    instructions_block = combined_text.split('\n\n')[0].strip()
    matches = re.findall(r'(\d+)\n(.+?)(?=\n\d+|\Z)', instructions_block, re.DOTALL)

    formatted_instructions = []
    if matches:
        for num, instruction_text in matches:
            formatted_instructions.append(f"* {instruction_text.strip()}")
        return "\n".join(formatted_instructions)
    else:
        lines = instructions_block.split('\n')
        if len(lines) >= 2 and lines[0].strip().isdigit():
            return f"* {lines[1].strip()}"
        else:
            if instructions_block:
                return f"* {instructions_block}"
            return instructions_block


st.header("🏋️‍♀️ Generate Your Workout Plan")
st.markdown("Tell us about your fitness goals and what you have available.")

with st.expander("Workout Plan Inputs", expanded=True):
    user_goals = st.selectbox(
        "What are your primary fitness goals?",
        ["muscle gain", "weight loss", "health"],
        index=0
    )

    fitness_level = st.selectbox(
        "What is your current fitness level?",
        ["beginner", "intermediate", "advanced"],
        index=1
    )

    selected_split_name = st.selectbox(
        "Choose a workout split that fits your preferences:",
        list(WORKOUT_SPLITS.keys())
    )

    selected_split = WORKOUT_SPLITS[selected_split_name]
    st.markdown(f"**Frequency:** {selected_split['frequency']}")
    st.markdown(f"**Focus:** {selected_split['focus']}")

    if selected_split["adjustable"] and len(selected_split["allowed_days"]) > 1:
        workout_days = st.radio(
            "How many days per week do you want to work out?",
            options=selected_split["allowed_days"],
            index=selected_split["allowed_days"].index(selected_split["default_days"]) if selected_split[
                                                                                              "default_days"] in \
                                                                                          selected_split[
                                                                                              "allowed_days"] else 0,
            key=f"workout_days_radio_{selected_split_name}"
        )
    else:
        workout_days = selected_split["default_days"]
        st.radio(
            "How many days per week do you want to work out?",
            options=[workout_days],
            index=0,
            key=f"workout_days_fixed_radio_{selected_split_name}",
            disabled=True
        )
        if selected_split_name != "Push/Pull Split (No Legs)":
            st.info(
                f"The number of workout days for **{selected_split_name}** is fixed at {workout_days} days/week.")

    all_equipment = ['bodyweight', 'cables', 'dumbbells', 'barbell', 'kettlebells', 'machine',
                     'band', 'plate', 'trx', 'smith-machine', 'bosu-ball', 'vitruvian',
                     'medicine-ball']

    equipment = st.multiselect(
        "What equipment do you have available?",
        options=['All'] + all_equipment,
        default=['All']
    )

    if 'All' in equipment:
        equipment = all_equipment

    special_instructions = st.text_area(
        "Any special instructions or considerations for your workout plan? (e.g., injuries, specific exercises to include/exclude, time limits per session)",
        height=100
    )

    if 'workout_plan_generated' not in st.session_state:
        st.session_state.workout_plan_generated = False
        st.session_state.workout_plan_output = ""
        st.session_state.extracted_exercises = []

    if st.button("Generate Workout Plan", key="generate_workout_tab"):
        if not equipment:
            st.warning("Please select at least one type of equipment you have.")
        else:
            with st.spinner("Generating your workout plan..."):
                rag_base_query = special_instructions if special_instructions.strip() else ""

                improved_query_for_rag = get_improved_rag_query(
                    original_query=rag_base_query,
                    muscle_groups=selected_split["target_muscles"],
                    goal=user_goals,
                    equipment=equipment,
                    difficulty=fitness_level
                )
                st.info(
                    f"Leveraging an AI-enhanced search query for more relevant exercise suggestions: `{improved_query_for_rag}`")

                rag_retrieved_exercises = retrieve_similar_documents(
                    query=improved_query_for_rag,
                    vectorstore=loaded_vectorstore_exercise,
                    k=20
                )
                workout_prompt = create_workout_prompt(
                    user_goals=user_goals,
                    fitness_level=fitness_level,
                    split=selected_split_name,
                    workout_days=int(workout_days),
                    target_muscles=selected_split["target_muscles"],
                    equipment=equipment,
                    special_instructions=special_instructions,
                    retrieved_exercises=rag_retrieved_exercises
                )
                workout_plan_output = generate_text_with_llm(workout_prompt)

                st.session_state.workout_plan_output = workout_plan_output
                st.session_state.extracted_exercises = extract_exercise_names(workout_plan_output)
                st.session_state.workout_plan_generated = True

    if st.session_state.workout_plan_generated:
        st.success("Workout plan successfully generated!")
        with st.expander("View Your Personalized Workout Plan", expanded=True):
            st.markdown(st.session_state.workout_plan_output)

        extracted_exercises = st.session_state.extracted_exercises

        if extracted_exercises:
            st.subheader("Explore Exercises:")
            display_options = ["Select an exercise..."] + sorted(list(set(extracted_exercises)))
            selected_exercise_name = st.selectbox(
                "Choose an exercise to see details:",
                options=display_options,
                key="exercise_dropdown"
            )

            if selected_exercise_name and selected_exercise_name != "Select an exercise...":
                all_db_exercise_names = exercises_df['Exercise Name'].dropna().unique().tolist()

                # Prioritize a stricter match first, then fall back to a more lenient one
                # Attempt to find a match using token_sort_ratio
                best_match = process.extractOne(
                    selected_exercise_name,
                    all_db_exercise_names,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=75  # Still a good starting point for general matches
                )

                if not best_match or best_match[1] < 75:  # If no good token_sort_ratio match, try partial_ratio
                    # If the first attempt is not strong enough, try partial_ratio for sub-string matches
                    # Adjust cutoff for partial_ratio, it often gives higher scores
                    best_match_partial = process.extractOne(
                        selected_exercise_name,
                        all_db_exercise_names,
                        scorer=fuzz.partial_ratio,
                        score_cutoff=85  # A higher cutoff for partial_ratio
                    )
                    # Use the partial match if it's better or if the first one failed to meet the threshold
                    if best_match_partial and (not best_match or best_match_partial[1] > best_match[1]):
                        best_match = best_match_partial

                if best_match:
                    matched_db_exercise_name, score = best_match
                    st.info(
                        f"Found best match for '{selected_exercise_name}': '{matched_db_exercise_name}' with a similarity score of {score}.")

                    matching_rows = exercises_df[exercises_df['Exercise Name'] == matched_db_exercise_name]

                    if not matching_rows.empty:
                        exercise_data = matching_rows.iloc[0]
                        st.markdown(f"### {exercise_data['Exercise Name']}")

                        st.markdown("**Instructions:**")
                        instructions = get_clean_instructions(exercise_data['Combined_Text'])
                        st.markdown(instructions)

                        if pd.notna(exercise_data['Video_URL']) and exercise_data['Video_URL'].strip():
                            st.video(exercise_data['Video_URL'])
                        else:
                            st.info("No video available for this exercise.")
                    else:
                        st.warning(
                            f"Details for '{matched_db_exercise_name}' were found via fuzzy matching, but the exact entry could not be retrieved. This is unexpected. Please check your data or the matching logic.")
                else:
                    st.warning(
                        f"Details for '{selected_exercise_name}' not found in the exercises database even with fuzzy matching. Consider adjusting the matching threshold or adding the exercise to your database.")

st.markdown("---")
st.info(
    "Disclaimer: This tool provides AI-generated suggestions and should not replace professional advice from a certified fitness trainer. Always consult with a healthcare professional before making significant changes to your exercise routine.")