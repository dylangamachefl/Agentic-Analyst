import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re
import traceback
import os  # ## NEW ## For file cleanup

# ## NEW ## Import prompts from our new file
from prompts import (
    ANALYSIS_GENERATOR_PROMPT_TEMPLATE,
    VISUALIZATION_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    DOMAIN_DETECTOR_PROMPT_TEMPLATE,
    CONCEPT_EXTRACTOR_PROMPT_TEMPLATE,
    EVALUATOR_PROMPT_TEMPLATE,
)

# --- HELPER & AGENT FUNCTIONS (MODIFIED TO USE PROMPT TEMPLATES) ---


@st.cache_data
def data_profiler_agent(df: pd.DataFrame) -> str:
    # ... (This function is unchanged)
    profile = {
        "file_info": {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": list(df.columns),
        },
        "column_details": {},
        "correlation_matrix": None,
    }
    for col in df.columns:
        col_data = df[col]
        col_profile = {}
        if pd.api.types.is_numeric_dtype(col_data):
            if col_data.nunique() == 2 and col_data.min() == 0 and col_data.max() == 1:
                semantic_type = "boolean"
            else:
                semantic_type = "numeric"
        else:
            try:
                pd.to_datetime(col_data)
                semantic_type = "datetime"
            except (ValueError, TypeError):
                if col_data.nunique() / len(col_data) < 0.5:
                    semantic_type = "categorical"
                else:
                    semantic_type = "text"
        col_profile["semantic_type"] = semantic_type
        col_profile["null_values"] = int(col_data.isnull().sum())
        if semantic_type == "numeric":
            desc = col_data.describe()
            col_profile["stats"] = {
                "mean": desc.get("mean", 0),
                "std": desc.get("std", 0),
                "min": desc.get("min", 0),
                "max": desc.get("max", 0),
                "25%": desc.get("25%", 0),
                "50%": desc.get("50%", 0),
                "75%": desc.get("75%", 0),
            }
        elif semantic_type == "categorical" or semantic_type == "boolean":
            col_profile["value_counts"] = col_data.value_counts().to_dict()
            col_profile["num_unique"] = len(col_profile["value_counts"])
        elif semantic_type == "datetime":
            dt_series = pd.to_datetime(col_data)
            col_profile["range"] = {
                "earliest": str(dt_series.min()),
                "latest": str(dt_series.max()),
            }
        elif semantic_type == "text":
            col_profile["num_unique"] = col_data.nunique()
        profile["column_details"][col] = col_profile
    numeric_cols = [
        col
        for col, details in profile["column_details"].items()
        if details["semantic_type"] == "numeric"
    ]
    if len(numeric_cols) > 1:
        profile["correlation_matrix"] = json.loads(df[numeric_cols].corr().to_json())
    return json.dumps(profile, indent=2)


# ## NEW ##: Token counting and tracking logic
def initialize_token_counter():
    st.session_state.token_usage = {"input": 0, "output": 0, "total": 0}


def update_token_count(response):
    """Updates the token count in session_state from an API response."""
    if "token_usage" not in st.session_state:
        initialize_token_counter()

    try:
        # Accessing token metadata if available
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        st.session_state.token_usage["input"] += input_tokens
        st.session_state.token_usage["output"] += output_tokens
        st.session_state.token_usage["total"] += input_tokens + output_tokens
    except (AttributeError, KeyError):
        # Fallback if usage_metadata is not available
        pass


# ## MODIFIED ##: get_model now includes enhanced error handling for API keys
def get_model(api_key, model_name):
    """Initializes and returns a Gemini GenerativeModel with better error handling."""
    if not api_key:
        st.error("API Key is missing. Please enter your Google AI API Key.")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        # Test the key with a small, inexpensive call
        model.count_tokens("test")
        return model
    except Exception as e:
        error_message = str(e)
        if "API key not valid" in error_message:
            st.error(
                "Your Google AI API Key is not valid. Please check the key and try again."
            )
        elif "quota" in error_message.lower():
            st.error(
                "API Quota Exceeded. Please check your Google AI account or try again later."
            )
        else:
            st.error(f"An error occurred configuring the Gemini API: {error_message}")
        return None


# ## MODIFIED ##: Agents now use imported prompts
def domain_detector_agent(model, data_profile: str) -> dict:
    prompt = DOMAIN_DETECTOR_PROMPT_TEMPLATE.format(data_profile=data_profile)
    response = model.generate_content(
        prompt, generation_config={"response_mime_type": "application/json"}
    )
    update_token_count(response)
    return json.loads(response.text)


def concept_extractor_agent(model, data_profile: str, domain_info: dict) -> list:
    prompt = CONCEPT_EXTRACTOR_PROMPT_TEMPLATE.format(
        domain=domain_info["domain"], data_profile=data_profile
    )
    response = model.generate_content(
        prompt, generation_config={"response_mime_type": "application/json"}
    )
    update_token_count(response)
    return json.loads(response.text)


# ## MODIFIED ##: analysis_generator_agent now accepts user_focus
def analysis_generator_agent(
    model,
    data_profile: str,
    domain_info: dict,
    concepts: list,
    history: list,
    user_focus_str: str,
) -> dict:
    history_str = json.dumps(history, indent=2) if history else "No previous attempts."
    prompt = ANALYSIS_GENERATOR_PROMPT_TEMPLATE.format(
        data_profile=data_profile,
        domain_info=domain_info,
        concepts=concepts,
        history_str=history_str,
        user_focus=user_focus_str,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt, generation_config={"response_mime_type": "application/json"}
            )
            update_token_count(response)
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            st.warning(
                f"Attempt {attempt + 1}: Analysis agent produced invalid JSON. Asking for a correction..."
            )
            error_feedback = f"The JSON you previously provided was invalid... Please correct the syntax and provide a complete, valid JSON object now."
            prompt = error_feedback
    raise Exception(f"Failed to generate valid JSON after {max_retries} attempts.")


def evaluator_agent(model, analysis: dict) -> dict:
    prompt = EVALUATOR_PROMPT_TEMPLATE.format(
        analysis_str=json.dumps(analysis, indent=2)
    )
    response = model.generate_content(
        prompt, generation_config={"response_mime_type": "application/json"}
    )
    update_token_count(response)
    return json.loads(response.text)


def insight_to_chart_agent(
    model, final_analysis: dict, df_columns: list, file_path_for_code: str
) -> str:
    prompt = VISUALIZATION_PROMPT_TEMPLATE.format(
        analysis_str=json.dumps(final_analysis, indent=2),
        columns_str=", ".join([f"'{col}'" for col in df_columns]),
    )

    max_retries = 3
    for attempt in range(max_retries):
        st.write(f"↪️ Visualization attempt {attempt + 1}...")
        response = model.generate_content(prompt)
        update_token_count(response)
        # ... (rest of the self-correction logic is unchanged)
        content = response.text
        thoughts_match = re.search(r"```thoughts(.*?)```", content, re.S)
        code_match = re.search(r"```python(.*?)```", content, re.S)
        if not (thoughts_match and code_match):
            st.warning(
                "Agent did not follow format. Re-prompting with stricter instructions."
            )
            prompt = (
                "Your previous response was not formatted correctly. Please return exactly two blocks: ```thoughts and ```python. "
                + prompt
            )
            continue
        thoughts = thoughts_match.group(1).strip()
        code_text = code_match.group(1).strip()
        try:
            if "matplotlib" in code_text:
                raise ValueError("Forbidden library 'matplotlib' was found.")
            local_namespace = {}
            exec(code_text, globals(), local_namespace)
            if "fig" in local_namespace:
                st.success(
                    f"   - Code validated successfully on attempt {attempt + 1}."
                )
                st.session_state.viz_thoughts = thoughts
                return code_text
            else:
                raise ValueError(
                    "Generated code ran without errors but did not create a 'fig' variable."
                )
        except Exception as e:
            error_traceback = traceback.format_exc()
            st.warning(
                f"   - Attempt {attempt + 1} failed. Asking agent to fix the code..."
            )
            prompt = f"The Python code you previously generated resulted in an error... THE ERROR TRACEBACK:\n```{error_traceback}```\n\nTHE FAULTY CODE:\n```python\n{code_text}```\n\nPlease provide a new, corrected Python script..."
    raise Exception(
        f"The visualization agent failed to generate valid code after {max_retries} attempts."
    )


def summary_agent(model, final_analysis: dict) -> str:
    prompt = SUMMARY_PROMPT_TEMPLATE.format(analysis_str=json.dumps(final_analysis))
    response = model.generate_content(prompt)
    update_token_count(response)
    return response.text


# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Data-to-Analysis Agent")
st.title("📊 Data-to-Analysis Agent")
st.markdown(
    "Upload a CSV file, and this multi-agent system will perform an analysis and generate a visualization."
)

# --- Sidebar for Inputs (MODIFIED) ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google AI API Key", type="password")
    model_options = [
        "gemini-1.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.5-flash",
    ]
    selected_model = st.selectbox("Choose a Gemini Model", options=model_options)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    max_iterations = st.slider("Max Analysis Iterations", 1, 5, 2)

    # ## NEW ##: Improvement #5 - Allow User Focus
    with st.expander("Advanced Settings"):
        user_focus = st.text_area(
            "Optional: Add a specific focus for the analysis",
            help="e.g., 'Focus on the relationship between marketing spend and revenue for the North region.'",
        )

# ## MODIFIED ##: No token estimation, just a placeholder for the actual count
token_usage_placeholder = st.empty()

if uploaded_file and api_key:
    # We still need to load the data to enable the button
    st.session_state.df = pd.read_csv(uploaded_file)

analyze_button = st.button(
    "Analyze Data", type="primary", disabled=(not uploaded_file or not api_key)
)

# --- Main App Logic (MODIFIED) ---
if analyze_button:
    st.session_state.analysis_complete = False
    initialize_token_counter()  # ## NEW ##

    model = get_model(api_key, selected_model)

    if model and "df" in st.session_state:
        df = st.session_state.df
        df_profile = data_profiler_agent(df)
        st.session_state.df_profile = df_profile

        # ## NEW ##: Format user focus for the prompt
        user_focus_str = (
            f"\n\n--- USER-DIRECTED FOCUS ---\nThe user has requested a specific focus for this analysis:\n'{user_focus}'\nPlease prioritize this in your 'domain_related' insights."
            if user_focus
            else ""
        )

        temp_file_path = "temp_data.csv"

        try:
            with st.spinner("The agents are thinking... This may take a moment."):
                # Agentic loop...
                with st.status("Step 1: Profiling Data...", expanded=True):
                    st.write("Data profile created successfully.")

                with st.status(
                    "Step 2: Performing Iterative Analysis...", expanded=True
                ) as status:
                    analysis_history = []
                    for i in range(max_iterations):
                        # ... (analysis loop is the same, but now passes user_focus_str)
                        if not analysis_history:
                            domain_info = domain_detector_agent(model, df_profile)
                            concepts = concept_extractor_agent(
                                model, df_profile, domain_info
                            )
                        analysis = analysis_generator_agent(
                            model,
                            df_profile,
                            domain_info,
                            concepts,
                            analysis_history,
                            user_focus_str,
                        )
                        evaluation = evaluator_agent(model, analysis)
                        if all(score == 4 for score in evaluation["scores"].values()):
                            break
                        analysis_history.append(
                            {"analysis": analysis, "evaluation": evaluation}
                        )
                    st.session_state.final_analysis = analysis
                    status.update(label="Step 2: Analysis Complete!", state="complete")

                with st.status(
                    "Step 3: Generating Visualization...", expanded=True
                ) as status:
                    df.to_csv(temp_file_path, index=False)  # Create temp file
                    visualization_code = insight_to_chart_agent(
                        model,
                        st.session_state.final_analysis,
                        df.columns.tolist(),
                        temp_file_path,
                    )
                    st.session_state.visualization_code = visualization_code
                    status.update(
                        label="Step 3: Visualization Complete!", state="complete"
                    )

                with st.status(
                    "Step 4: Generating Summary...", expanded=True
                ) as status:
                    analysis_summary = summary_agent(
                        model, st.session_state.final_analysis
                    )
                    st.session_state.analysis_summary = analysis_summary
                    status.update(label="Step 4: Summary Complete!", state="complete")

                st.session_state.analysis_complete = True
                st.success("Analysis and Visualization Pipeline Completed!")

        # ## NEW ##: Improvement #1 - Enhanced Error Handling
        except genai.types.StopCandidateException as e:
            st.error(
                f"Analysis Stopped by API: The model stopped generating content, possibly due to safety settings. Details: {e}"
            )
        except Exception as e:
            st.error(f"An unexpected error occurred during the analysis: {e}")
            st.exception(e)  # Show full traceback for debugging

        # ## NEW ##: Improvement #2 - Post-Analysis Cleanup
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Cleaned up {temp_file_path}")
            # ## NEW ##: Improvement #3 - Display Actual Token Usage
            if st.session_state.get("token_usage", {}).get("total", 0) > 0:
                token_usage = st.session_state.token_usage
                token_usage_placeholder.info(
                    f"**Actual Token Usage:** {token_usage['total']:,} tokens (Input: {token_usage['input']:,}, Output: {token_usage['output']:,})"
                )

    else:
        st.error(
            "Could not proceed. Please ensure a file is uploaded and your API key is valid."
        )

# --- Display Results (Unchanged from before) ---
if st.session_state.get("analysis_complete", False):
    st.header("Results")
    # ... (The rest of the display logic is the same)
    st.subheader("Textual Analysis & Insights")
    if "analysis_summary" in st.session_state:
        st.markdown("#### Executive Summary")
        st.markdown(st.session_state.analysis_summary)
    analysis_json = st.session_state.final_analysis
    if "analysis" in analysis_json and all(
        k in analysis_json["analysis"]
        for k in ["Descriptive", "Predictive", "Domain-Related"]
    ):
        st.markdown("#### Detailed Analysis")
        tab1, tab2, tab3 = st.tabs(
            ["📊 Descriptive", "📈 Predictive", "💡 Strategic (Domain)"]
        )
        with tab1:
            st.write(analysis_json["analysis"]["Descriptive"])
        with tab2:
            st.write(analysis_json["analysis"]["Predictive"])
        with tab3:
            st.write(analysis_json["analysis"]["Domain-Related"])
    st.divider()
    st.subheader("Generated Visualization")
    try:
        local_namespace = {}
        exec(st.session_state.visualization_code, globals(), local_namespace)
        fig = local_namespace.get("fig")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Generated code did not produce a Plotly figure named 'fig'.")
    except Exception as e:
        st.error(f"Error executing visualization code: {e}")
    with st.expander("View Agent's Visualization Reasoning"):
        if "viz_thoughts" in st.session_state:
            st.markdown(st.session_state.viz_thoughts)
        else:
            st.info("The visualization agent did not provide its reasoning.")
    with st.expander("View Generated Visualization Code"):
        st.code(st.session_state.visualization_code, language="python")
    with st.expander("View Full Raw JSON Analysis"):
        st.json(analysis_json)
