# Agentic Analyst: A Multi-Agent AI Analyst

This project is an interactive web application that leverages a sophisticated multi-agent system, powered by the Google Gemini API, to automatically analyze a CSV dataset, derive strategic business insights, and generate a final, insightful data visualization.

The application is a functional implementation inspired by the concepts presented in the research paper **"Data-to-Dashboard: Multi-Agent LLM Framework for Insightful Visualization in Enterprise Analytics"**. It showcases how modern LLMs can simulate the reasoning process of a team of expert analysts.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini_API-4285F4?logo=google)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Live Demo

**[Link to your deployed Streamlit Community Cloud App]**  
*(Note: You will replace this with your actual URL after deploying the app.)*

## ✨ Key Features

*   **Multi-Agent System:** Utilizes a pipeline of specialized AI agents (Profiler, Analyst, Charting Expert, Summarizer) that collaborate to produce the final output.
*   **Tree-of-Thought (ToT) Reasoning:** The visualization agent uses a ToT prompt to debate and select the most effective chart type, explaining its rationale.
*   **Expert Persona Prompting:** The analysis agent adopts the persona of a senior business consultant, using established business frameworks (e.g., SWOT, PESTEL) to generate high-quality, strategic insights.
*   **Self-Correcting JSON Output:** The system automatically detects when an agent produces invalid JSON, re-prompts it with the error, and asks for a correction, making the pipeline more robust.
*   **Live Token Estimation:** Provides users with a real-time estimate of the token usage for their analysis *before* they run it, promoting transparent and cost-conscious use.
*   **Interactive UI:** Built with Streamlit for a clean, user-friendly experience, including model selection, file uploads, and a polished results display with tabs and summaries.

## 📸 Application Screenshot

*(It's highly recommended to add a screenshot of your working app here. Replace the placeholder below.)*


## 🛠️ Technology Stack

*   **Language:** Python 3.9+
*   **AI Model:** Google Gemini API
*   **Web Framework:** Streamlit
*   **Data Manipulation:** Pandas
*   **Plotting:** Plotly

## 🏛️ System Architecture

This application simulates a team of analysts by chaining together several specialized agents. Each agent is a Python function that calls the Gemini API with a highly specific, role-oriented prompt.

1.  **`Data Profiler Agent` (Local):**
    *   **Role:** The Data Clerk.
    *   **Action:** Ingests the raw CSV and creates a compact, statistical JSON summary using Pandas. This is a crucial cost-saving step that avoids sending the entire file to the LLM.

2.  **`Analysis Generator Agent`:**
    *   **Role:** The Senior Business Consultant.
    *   **Action:** Takes the data profile and, using its "InsightWriter-Advanced" persona, generates a deep textual analysis covering descriptive, predictive, and strategic domain-related insights. It has a self-correction loop to ensure its JSON output is valid.

3.  **`Insight-to-Chart Agent`:**
    *   **Role:** The Visualization Expert.
    *   **Action:** Implements a Tree-of-Thought process. It analyzes the final textual analysis, debates chart types, selects the best one to communicate the core insight, and then generates the Python code (using Plotly) to create the visualization. It also outputs its reasoning.

4.  **`Summary Agent`:**
    *   **Role:** The Communications Expert.
    *   **Action:** Reads the final, detailed JSON analysis and writes a concise, high-level executive summary in markdown for easy digestion by the user.

## ⚙️ Local Setup and Installation

To run this application on your local machine, follow these steps:

### 1. Clone the Repository```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Create a `requirements.txt` file
Create a file named `requirements.txt` in the root of your project and add the following lines:
```
streamlit
pandas
google-generativeai
plotly
python-dotenv
```

### 4. Install Dependencies```bash
pip install -r requirements.txt
```

### 5. Set Up Your API Key
For security, we will use an environment file.
1.  Create a file named `.env` in the root of your project.
2.  Add your Google AI API Key to this file:
    ```
    GOOGLE_API_KEY="your-api-key-here"
    ```
3.  **Important:** Your `app.py` must be modified to load this key. The provided `app.py` uses `st.text_input` for simplicity, but for local development, you could modify `get_model` to use `os.getenv("GOOGLE_API_KEY")`.

### 6. Run the Application
```bash
streamlit run app.py
```Your browser will open a new tab with the running application.

## ☁️ Deployment on Streamlit Community Cloud (Free)

You can deploy this app for free and share it with the world.

1.  **Push to GitHub:** Make sure your project, including the `app.py` and `requirements.txt` files, is pushed to a public GitHub repository.

2.  **Sign Up for Streamlit Community Cloud:** Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign up using your GitHub account.

3.  **Deploy New App:**
    *   From your workspace, click "New app".
    *   Select your repository, the branch (usually `main`), and the path to your `app.py` file.
    *   Click "Deploy!".

4.  **Add Your API Key as a Secret:** Your app will not work until you provide the API key securely.
    *   In your app's dashboard on Streamlit Cloud, go to the "Settings" menu (the three dots).
    *   Go to the "Secrets" section.
    *   Add your API key in the following format and save it:
        ```toml
        # .streamlit/secrets.toml
        API_KEY="your-google-api-key-here"
        ```
    *   You will need to modify the `app.py` one last time to read this secret. Change the `api_key` line to:
        ```python
        # In the sidebar section
        api_key = st.secrets.get("API_KEY", "")
        ```
        (You can remove the `st.text_input` for the API key in your deployed version).

## 🎓 Inspiration and Credit

This project is a practical implementation and extension of the ideas presented in the following academic paper. The core concepts of the multi-agent pipeline and the Tree-of-Thought approach for visualization are heavily inspired by their work.

*   **Paper:** Data-to-Dashboard: Multi-Agent LLM Framework for Insightful Visualization in Enterprise Analytics
*   **Authors:** Ran Zhang, Mohannad Elhamod
*   **arXiv Link:** [https://arxiv.org/abs/2405.16395](https://arxiv.org/abs/2405.16395)

## 🚀 Future Improvements

*   **Support More File Types:** Extend the app to handle Excel files (`.xlsx`) or connect to databases.
*   **Chart Customization:** Allow the user to give feedback on a generated chart and ask the agent to regenerate it with modifications.
*   **Caching Results:** Implement a more robust caching mechanism to save the results of a full analysis for a given file, reducing costs on subsequent runs.
*   **Enhanced Error Handling:** Provide more specific feedback to the user if a particular agent fails its task repeatedly.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.