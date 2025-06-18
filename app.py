import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re

# --- HELPER & AGENT FUNCTIONS ---


def estimate_token_usage(model, df_profile, max_iterations):
    """
    Estimates the number of input tokens for the analysis pipeline by sending
    mock prompts to the model's count_tokens method.
    """
    AVG_OUTPUT_PER_CALL = 250
    total_tokens = 0
    try:
        # Construct the mock prompts
        domain_prompt = f"Based on the following statistical data profile, identify the most likely business domain... Profile: {df_profile} ...Output your response in a valid JSON format..."
        concept_prompt = f"You are an expert in a domain. Based on the provided data profile, extract key analytical concepts... Profile: {df_profile} ...Output a JSON list of strings."

        # KEY FIX: Wrap prompts in lists [] for the count_tokens API call
        total_tokens += model.count_tokens([domain_prompt]).total_tokens
        total_tokens += model.count_tokens([concept_prompt]).total_tokens

        # Simulate the iterative analysis loop
        history_tokens = 0
        for i in range(max_iterations):
            analysis_prompt = f"As a senior data analyst, generate structured insights... Profile: {df_profile} ...Previous Attempts: {'.'*history_tokens} ...Output a unified JSON..."
            total_tokens += model.count_tokens([analysis_prompt]).total_tokens
            history_tokens += AVG_OUTPUT_PER_CALL

            evaluator_prompt = f"You are a meticulous lead analyst. Evaluate the following analysis... Analysis to Evaluate: {'.'*history_tokens} ...Output a valid JSON..."
            total_tokens += model.count_tokens([evaluator_prompt]).total_tokens
            history_tokens += AVG_OUTPUT_PER_CALL

        # Final chart agent
        chart_prompt = f"You are a data visualization expert... Profile: {df_profile} ...Final Analysis: {'.'*history_tokens} ...Output only the final, complete Python code..."
        total_tokens += model.count_tokens([chart_prompt]).total_tokens

        return f"{total_tokens:,}"  # Return as a formatted string

    except Exception as e:
        print(f"Token estimation failed: {e}")
        return "Not available"


@st.cache_data
def data_profiler_agent(df: pd.DataFrame) -> str:
    """Creates a statistical JSON summary of a DataFrame without using an LLM."""
    profile = {
        "file_info": {
            "num_rows": len(df),
            "num_cols": len(df.columns),
            "columns": list(df.columns),
        },
        "column_summaries": json.loads(df.describe(include="all").to_json()),
        "null_values": json.loads(df.isnull().sum().to_json()),
    }
    return json.dumps(profile, indent=2)


def get_model(api_key, model_name):
    """Initializes and returns a Gemini GenerativeModel."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        return None


def domain_detector_agent(model, data_profile: str) -> dict:
    """Identifies the business domain of the dataset."""
    prompt = f"""Based on the following statistical data profile, identify the most likely business domain (e.g., Finance, Marketing, Operations, HR). Provide a concise, one-sentence definition for the identified domain. Data Profile: {data_profile}. Output your response in a valid JSON format with keys "domain" and "definition"."""
    response = model.generate_content(
        prompt, generation_config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)


def concept_extractor_agent(model, data_profile: str, domain_info: dict) -> list:
    """Extracts key business concepts and KPIs from the data profile."""
    domain = domain_info["domain"]
    prompt = f"""You are an expert in the {domain} domain. Based on the provided data profile, extract key analytical concepts or KPIs relevant to this domain. Formulate them as natural language phrases (e.g., "customer lifetime value", "churn rate"). Data Profile: {data_profile}. Output a JSON list of strings."""
    response = model.generate_content(
        prompt, generation_config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)


def analysis_generator_agent(
    model, data_profile: str, domain_info: dict, concepts: list, history: list
) -> dict:
    """Generates the core analysis with a self-correction retry loop and an expert persona."""
    domain = domain_info["domain"]
    history_str = json.dumps(history, indent=2) if history else "No previous attempts."

    # --- NEW, MORE ADVANCED PROMPT ---
    prompt = f"""
    You are "InsightWriter-Advanced", a world-class business analyst consultant. Your analysis is always sharp, insightful, and actionable.
    You will turn raw data into high-value insights for a business leader.

    Your core methodology is to apply established business analysis techniques:
    1.  **Business-Lens Taxonomy:** Frame your findings using concepts like Trend, Variance, Benchmark, Correlation, and Efficiency.
    2.  **Business Frameworks:** Where applicable, connect your insights to strategic frameworks like SWOT (Strengths, Weaknesses, Opportunities, Threats), PESTEL, or Porter's Five Forces.

    Here is the data context:
    - Profile: {data_profile}
    - Domain: {domain_info}
    - Core concepts to investigate: {concepts}
    - Previous attempts and feedback to improve upon: {history_str}

    Produce a JSON response with three distinct paragraphs for the 'analysis' key:
    1.  'descriptive': What the data says. (e.g., "A downward trend in sales was observed in Q3...")
    2.  'predictive': What the data suggests might happen next. (e.g., "This trend suggests Q4 sales may also underperform without intervention...")
    3.  'domain_related': The strategic "so what" for the business, referencing a business framework. (e.g., "From a SWOT perspective, this sales dip is a clear Weakness that could be exploited by competitors...")

    You MUST output a unified and syntactically correct JSON object with the keys: 'domain', 'core_concepts', and 'analysis'.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt, generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            st.warning(
                f"Attempt {attempt + 1}: Analysis agent produced invalid JSON. Asking for a correction..."
            )
            error_feedback = (
                f"The JSON you previously provided was invalid. I received this error:\n```\n{e}\n```\n"
                f"Here is the invalid JSON text you sent:\n```json\n{response.text}\n```\n"
                f"Please correct the syntax and provide a complete, valid JSON object now."
            )
            prompt = error_feedback
    raise Exception(f"Failed to generate valid JSON after {max_retries} attempts.")


def evaluator_agent(model, analysis: dict) -> dict:
    """Scores the generated analysis based on several criteria."""
    analysis_str = json.dumps(analysis, indent=2)
    prompt = f"""You are a meticulous lead analyst. Evaluate the following analysis based on a strict rubric. For each dimension, provide a score from 1 (poor) to 4 (excellent) and a brief justification. Dimensions: Insightfulness, Novelty, Analytical Depth, Domain Relevance. Analysis to Evaluate: {analysis_str}. Output a valid JSON object with two keys: "scores" and "justifications"."""
    response = model.generate_content(
        prompt, generation_config={"response_mime_type": "application/json"}
    )
    return json.loads(response.text)


def insight_to_chart_agent(
    model, final_analysis: dict, df_columns: list, file_path_for_code: str
) -> str:
    """Generates runnable Python code for a Plotly chart using a Tree-of-Thought process."""
    analysis_str = json.dumps(final_analysis, indent=2)
    columns_str = ", ".join([f"'{col}'" for col in df_columns])

    # --- NEW, TREE-OF-THOUGHT PROMPT ---
    prompt = f"""
    You are an elite data-visualisation consultant who writes flawless Python code for Plotly.
    Your task is to create the single best visualization to represent the core insight from the provided analysis.

    --- CONTEXT ---
    1.  **Final Analysis JSON:** {analysis_str}
    2.  **Available Data Columns:** You MUST use column names from this list only: [{columns_str}]

    --- TREE-OF-THOUGHT PROCESS ---
    Follow these steps to decide on the visualization. I want you to output this thinking process later.

    **Step I – Identify Core Insight:** What is the single most important business question or finding in the 'domain_related' analysis that a chart can answer?

    **Step II – Propose Chart Types:** Based on the core insight and available data, propose 2-3 suitable chart types (e.g., scatter plot, bar chart, line chart, heatmap). For each, explain its pros and cons in one sentence.

    **Step III – Consolidate and Decide:** Choose the single best chart type. Provide a final, clear reason for your choice, explaining how it will effectively communicate the core business insight.

    --- VISUALIZATION BEST PRACTICES ---
    When you write the code, you must adhere to these rules:
    - **Clarity:** Use clear, descriptive titles and axis labels. The title should summarize the chart's main finding.
    - **Color:** Use a colorblind-friendly and professional palette, like Plotly's default or `px.colors.qualitative.Plotly`.
    - **Data Aggregation:** If necessary, perform data aggregation (like groupby, pivot_table) in pandas *before* passing the data to Plotly.
    - **Interactivity:** Make good use of `hover_data` to show relevant details on mouse-over.

    --- FINAL INSTRUCTIONS ---
    Return **exactly two fenced blocks** in order:
    1.  **Thoughts block (label it ```thoughts)**: Your full Step I-III Tree-of-Thought reasoning.
    2.  **Python block (label it ```python)**: A script that:
        - Imports pandas and plotly.express.
        - Loads the data from '{file_path_for_code}'.
        - Implements your final chart decision.
        - Is self-contained and ready to execute.
        - Assigns the final chart to a variable named `fig`.
    """

    response = model.generate_content(prompt)

    # --- NEW: Extract both thoughts and code ---
    content = response.text
    thoughts_match = re.search(r"```thoughts(.*?)```", content, re.S)
    code_match = re.search(r"```python(.*?)```", content, re.S)

    if not (thoughts_match and code_match):
        # Fallback for models that might ignore the dual-block instruction
        st.warning(
            "Agent did not follow ToT format. Falling back to code-only extraction."
        )
        st.session_state.viz_thoughts = "The agent did not provide its thought process."
        code_text = content  # Assume the whole response is code
    else:
        st.session_state.viz_thoughts = thoughts_match.group(1).strip()
        code_text = code_match.group(1).strip()

    # Clean up the code block just in case
    if code_text.startswith("```python"):
        code_text = code_text[9:]
    if code_text.endswith("```"):
        code_text = code_text[:-3]

    return code_text.strip()


def summary_agent(model, final_analysis: dict) -> str:
    """Creates a high-level executive summary from the detailed analysis JSON."""
    analysis_str = json.dumps(final_analysis)

    # --- TWEAKED PROMPT ---
    prompt = f"""
    You are a senior business communication expert, skilled at distilling complex analyses into clear, concise takeaways for C-level executives.
    Based on the following detailed JSON data analysis, write a high-level executive summary.
    Focus on the most critical insights and actionable recommendations.
    Present the summary as 3-4 key bullet points using markdown.

    Detailed Analysis:
    {analysis_str}
    """

    response = model.generate_content(prompt)
    return response.text


# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Data-to-Dashboard Agent")
st.title("📊 Data-to-Dashboard Agent")
st.markdown(
    "Upload a CSV file, and this multi-agent system will perform an analysis and generate a visualization."
)

# --- Sidebar for Inputs ---
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

# --- Live Token Estimation ---
token_estimation_placeholder = st.empty()

if uploaded_file and api_key:
    model = get_model(api_key, selected_model)
    if model:
        df = pd.read_csv(uploaded_file)
        df_profile = data_profiler_agent(df)

        # Store df and profile in session_state for the main analysis step
        st.session_state.df = df
        st.session_state.df_profile = df_profile

        estimated_tokens = estimate_token_usage(model, df_profile, max_iterations)
        token_estimation_placeholder.info(
            f"**Estimated Input Tokens for this run:** ~{estimated_tokens} tokens.\n\n*(This is an approximation of input tokens only. Actual usage will vary.)*"
        )

elif not uploaded_file:
    token_estimation_placeholder.info(
        "Upload a CSV file to see a token usage estimate."
    )

# --- Button to start the main analysis ---
analyze_button = st.button(
    "Analyze Data", type="primary", disabled=(not uploaded_file or not api_key)
)

# --- Main App Logic ---
if analyze_button:
    token_estimation_placeholder.empty()
    model = get_model(api_key, selected_model)

    # Check for data in session_state to prevent errors
    if model and "df" in st.session_state and "df_profile" in st.session_state:
        df = st.session_state.df
        df_profile = st.session_state.df_profile

        with st.spinner("The agents are thinking... This may take a moment."):
            try:
                # The agentic loop
                with st.status("Step 1: Profiling Complete!", expanded=True) as status:
                    st.write("Using pre-computed data profile.")

                with st.status(
                    "Step 2: Performing iterative analysis...", expanded=True
                ) as status:
                    analysis_history = []
                    for i in range(max_iterations):
                        st.write(f"↪️ Iteration {i+1}...")
                        if not analysis_history:
                            domain_info = domain_detector_agent(model, df_profile)
                            concepts = concept_extractor_agent(
                                model, df_profile, domain_info
                            )
                        analysis = analysis_generator_agent(
                            model, df_profile, domain_info, concepts, analysis_history
                        )
                        evaluation = evaluator_agent(model, analysis)
                        if all(score == 4 for score in evaluation["scores"].values()):
                            st.write("✅ Achieved optimal analysis.")
                            break
                        reflection_data = {
                            "analysis": analysis,
                            "evaluation": evaluation,
                        }
                        analysis_history.append(reflection_data)
                    st.session_state.final_analysis = analysis
                    status.update(label="Step 2: Analysis Complete!", state="complete")

                with st.status(
                    "Step 3: Generating visualization code...", expanded=True
                ) as status:
                    with open("temp_data.csv", "w") as f:
                        f.write(df.to_csv(index=False))
                    visualization_code = insight_to_chart_agent(
                        model,
                        st.session_state.final_analysis,
                        df.columns.tolist(),
                        "temp_data.csv",
                    )
                    st.session_state.visualization_code = visualization_code
                    status.update(
                        label="Step 3: Visualization Complete!", state="complete"
                    )

                # --- NEW: Final Summary Step ---
                with st.status(
                    "Step 4: Generating executive summary...", expanded=True
                ) as status:
                    analysis_summary = summary_agent(
                        model, st.session_state.final_analysis
                    )
                    st.session_state.analysis_summary = analysis_summary
                    status.update(label="Step 4: Summary Complete!", state="complete")

                st.session_state.analysis_complete = True
                st.success("Analysis and Visualization Pipeline Completed!")

            except Exception as e:
                st.error(f"An error occurred during the analysis: {e}")
                st.exception(e)
    else:
        st.error(
            "Data not loaded correctly. Please ensure a file is uploaded and API key is set."
        )

# --- Display Results (FINAL VERSION) ---
if st.session_state.get("analysis_complete", False):
    st.header("Results")

    # --- Part 1: The Generated Chart and its Rationale ---
    st.subheader("Generated Visualization")

    col1, col2 = st.columns([3, 2])  # Give more space to the chart
    with col1:
        try:
            local_namespace = {}
            exec(st.session_state.visualization_code, globals(), local_namespace)
            fig = local_namespace.get("fig")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(
                    "Generated code did not produce a Plotly figure named 'fig'."
                )
        except Exception as e:
            st.error(f"Error executing visualization code: {e}")

    with col2:
        st.markdown("##### 💡 Agent's Reasoning")
        # Display the thoughts captured from the ToT agent
        if "viz_thoughts" in st.session_state:
            st.markdown(st.session_state.viz_thoughts)
        else:
            st.info("The visualization agent did not provide its reasoning.")

    # Show the generated code in an expander
    with st.expander("View Generated Visualization Code"):
        st.code(st.session_state.visualization_code, language="python")

    st.divider()

    # --- Part 2: The Formatted Textual Analysis ---
    st.header("Textual Analysis & Insights")

    if "analysis_summary" in st.session_state:
        st.subheader("Executive Summary")
        st.markdown(st.session_state.analysis_summary)

    analysis_json = st.session_state.final_analysis
    if (
        "Descriptive" in analysis_json.get("analysis", {})
        and "Predictive" in analysis_json.get("analysis", {})
        and "Domain-Related" in analysis_json.get("analysis", {})
    ):
        st.subheader("Detailed Analysis")
        tab1, tab2, tab3 = st.tabs(
            ["📊 Descriptive", "📈 Predictive", "💡 Strategic (Domain)"]
        )
        with tab1:
            st.write(analysis_json["analysis"]["Descriptive"])
        with tab2:
            st.write(analysis_json["analysis"]["Predictive"])
        with tab3:
            st.write(analysis_json["analysis"]["Domain-Related"])

    with st.expander("View Full Raw JSON Analysis"):
        st.json(analysis_json)
