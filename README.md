# Data-to-Analysis: A Multi-Agent AI Analyst

This project is an interactive web application that leverages a sophisticated multi-agent system, powered by the Google Gemini API, to automatically analyze a CSV dataset, derive strategic business insights, and generate a final, insightful data visualization.

The application is a functional implementation inspired by the concepts presented in the research paper **"Data-to-Dashboard: Multi-Agent LLM Framework for Insightful Visualization in Enterprise Analytics"**. It showcases how modern LLMs can simulate the reasoning process of a team of expert analysts.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini_API-4285F4?logo=google)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Live Demo

[Streamlit Link](https://data-to-analysis-app.streamlit.app/) 

## ✨ Key Features

*   **Advanced Multi-Agent System:** Utilizes a pipeline of specialized AI agents (Profiler, Domain Detector, Concept Extractor, Analyst, Visualizer, Summarizer) that collaborate in a self-correcting, iterative loop.
*   **Modular & Maintainable Code:** Agent prompts are modularized into a separate `prompts.py` file, making the core application logic cleaner and easier to manage.
*   **Rich Data Profiling:** The initial `Data Profiler` (which uses no LLM calls) performs deep analysis, including semantic type detection (identifying categories, booleans, datetimes), detailed value counts, and a numerical correlation matrix to ground the AI agents with high-quality data.
*   **Expert Persona & ToT Reasoning:** The `Analysis Agent` adopts the persona of a senior business consultant to generate strategic insights, while the `Visualization Agent` uses a Tree-of-Thought (ToT) process to debate and select the most effective chart type.
*   **Robust Self-Correction:** Both the analysis and visualization agents feature a retry mechanism. If an agent produces invalid JSON or buggy Python code, it automatically re-prompts itself with the error, enabling it to fix its own mistakes.
*   **User-Directed Analysis:** An "Advanced Settings" option allows users to provide a specific focus or question, guiding the AI's analysis toward their primary areas of interest.
*   **Transparent & Secure:** Provides users with the *actual* token count post-analysis for transparent cost tracking. It also includes enhanced error handling for API issues and automatically cleans up temporary data files for better security.

## 📸 Application Screenshot

![Screenshot 2025-06-25 at 13-33-45 Data-to-Analysis Agent · Streamlit](https://github.com/user-attachments/assets/61af6b6b-116d-41fb-89b7-24a7e86b6ffa)

## 🛠️ Technology Stack

*   **Language:** Python 3.9+
*   **AI Model:** Google Gemini API
*   **Web Framework:** Streamlit
*   **Data Manipulation:** Pandas
*   **Plotting:** Plotly

## 🏛️ System Architecture

This application simulates a team of expert analysts by chaining together several specialized agents in an iterative, self-reflecting loop inspired by the original research paper.

The core process is: **Profile -> Detect -> Extract -> Analyze -> Evaluate -> Reflect**.

1.  **`Data Profiler Agent` (Local):**
    *   **Role:** The Data Clerk.
    *   **Action:** Ingests the raw CSV and creates a compact, statistical JSON summary using Pandas. This crucial first step avoids sending the entire (and potentially large) file to the LLM, saving costs and improving performance.

2.  **`Domain Detector Agent`:**
    *   **Role:** The Domain Specialist.
    *   **Action:** Examines the data profile (column names, data types) to determine the specific business domain (e.g., "Marketing Analytics," "Financial Operations"). This provides essential context for all subsequent steps.

3.  **`Concept Extractor Agent`:**
    *   **Role:** The KPI Strategist.
    *   **Action:** Takes the identified domain and data profile, and extracts a list of relevant key performance indicators (KPIs) and business concepts to focus on (e.g., "Customer Acquisition Cost," "Return on Investment"). This ensures the analysis is focused and relevant.

4.  **`Analysis Generator Agent`:**
    *   **Role:** The Senior Business Consultant.
    *   **Action:** This is the core reasoning engine. It takes the profile, domain, and concepts, and adopts the "InsightWriter-Advanced" persona to generate a deep textual analysis covering descriptive, predictive, and strategic insights, often referencing established business frameworks (like SWOT).

5.  **`Evaluator & Reflector Agents`:**
    *   **Role:** The Quality Assurance Team.
    *   **Action:** The `evaluator_agent` scores the analysis on metrics like depth, novelty, and relevance. If the scores are not perfect, the results are passed to a `reflector_agent` (implicitly handled in our code by feeding the evaluation back into the main analysis prompt on the next loop). This iterative feedback loop forces the system to improve its own output.

6.  **`Insight-to-Chart Agent`:**
    *   **Role:** The Visualization Expert.
    *   **Action:** Implements a Tree-of-Thought process. It analyzes the *final, polished* textual analysis, debates chart types, and generates the Python code for the single most effective visualization, complete with its own reasoning.

7.  **`Summary Agent`:**
    *   **Role:** The Communications Expert.
    *   **Action:** Distills the final, complex analysis into a concise, bullet-point executive summary for the user.
    
## ⚙️ Local Setup and Installation

To run this application on your local machine, follow these steps:

### 1. Clone the Repository
```bash
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

### 4. Install Dependencies
```bash
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
```
Your browser will open a new tab with the running application.

## 🎓 Inspiration and Credit

This project is a practical implementation and extension of the ideas presented in the following academic paper. The core concepts of the multi-agent pipeline and the Tree-of-Thought approach for visualization are heavily inspired by their work.

*   **Paper:** Data-to-Dashboard: Multi-Agent LLM Framework for Insightful Visualization in Enterprise Analytics
*   **Authors:** Ran Zhang, Mohannad Elhamod
*   **arXiv Link:** [https://arxiv.org/abs/2505.26395](https://arxiv.org/abs/2505.23695)

## 🚀 Future Improvements

*   **Support More File Types:** Extend the app to handle Excel files (`.xlsx`) or connect to databases.
*   **Chart Customization:** Allow the user to give feedback on a generated chart and ask the agent to regenerate it with modifications.
*   **Caching Results:** Implement a more robust caching mechanism to save the results of a full analysis for a given file, reducing costs on subsequent runs.
*   **Enhanced Error Handling:** Provide more specific feedback to the user if a particular agent fails its task repeatedly.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
