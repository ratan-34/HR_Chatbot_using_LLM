# HR_Chatbot_using_LLM

🤖 HR Resource Query Chatbot
An intelligent AI-powered HR assistant that helps HR teams find the most relevant employees for projects and tasks using natural language queries.

📌 Overview
This project implements a Retrieval-Augmented Generation (RAG)-based chatbot to streamline HR resource allocation. Given a user query like "Find a Python developer with healthcare experience," the system retrieves matching employee profiles, augments context, and generates natural-sounding responses using Azure OpenAI (GPT-4o).

🚀 Features
✅ Employee search with natural language queries

✅ RAG pipeline: Retrieval ➝ Augmentation ➝ Generation

✅ Azure OpenAI (GPT-4o) integration for response generation

✅ Semantic search using TF-IDF + Cosine Similarity

✅ Flask backend API with endpoints for chat and employee search

✅ Clean HTML-based frontend chat interface

✅ Error handling and structured JSON API responses

🏗️ Architecture
text
Copy
Edit
User ↔ Frontend (HTML/JS)
          ↕
    Flask Backend (Python)
          ↕
Retrieval: TF-IDF + Cosine Similarity
          ↕
Augmentation: Match Data + Query
          ↕
Generation: Azure OpenAI (GPT-4o)
🧰 Setup & Installation
🔧 Prerequisites
Python 3.8+

Azure OpenAI account with API key and endpoint

📦 Installation
bash
Copy
Edit
# Clone the repository
git clone https://github.com/your-username/hr-resource-chatbot.git
cd hr-resource-chatbot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
📡 API Documentation
🔹 POST /chat
Description: Submit a user query to the chatbot.

Request Payload:

json
Copy
Edit
{
  "query": "Find React developers with 3+ years experience",
  "session_id": "abc123"
}
Response:

json
Copy
Edit
{
  "response": "Based on your query...",
  "query_type": "hr_related",
  "relevant_employees": [...],
  "timestamp": "..."
}
🔹 GET /employees/search?q=python
Returns employees matching the keyword.

🔹 GET /employees
Returns all available employees in the dataset.

🧠 AI Development Process
🛠️ AI Tools Used
ChatGPT (GPT-4o) – for planning, debugging, and writing the code logic

GitHub Copilot – for code suggestions and speedup

ChatGPT Plugins – for architecture decisions and problem solving

📊 AI Usage Breakdown
~70% AI-assisted code

~30% hand-written and manually debugged code

✅ AI Helped With:
Designing the retrieval and generation pipeline

Creating TF-IDF vectorization and similarity logic

Prompting structure for GPT-4o

Structuring RESTful API in Flask

❌ AI Couldn't Help With:
HTML interface bug fixes

Embedding test data manually

Optimizing for Render deployment

🧪 Technical Decisions
Component	Choice	Justification
Backend	Flask	Quick prototyping, familiar syntax
LLM	Azure OpenAI GPT-4o	Best-in-class generation with minimal setup
Embeddings	TF-IDF	Lightweight and fast for 15-employee dataset
UI	HTML + Jinja2	Simple and deployable on Render
Deployment	Render	Free and easy public hosting

🔮 Future Improvements
Integrate OpenAI embeddings + FAISS for better semantic matching

Expand employee dataset with real-world resume samples

Convert backend to FastAPI + async for scalability

Build React or Streamlit frontend

Add user authentication and admin dashboard

Enable interview scheduling and team assignment automation

🌐 Demo
🔗 GitHub Repository: [https://github.com/your-username/hr-resource-chatbot
](https://github.com/ratan-34/HR_Chatbot_using_LLM)
🚀 Live Demo: https:[//your-app-name.onrender.com](https://hr-chatbot-using-llm-2.onrender.com)

🧾 Sample Employee Data
json
Copy
Edit
{
  "id": 1,
  "name": "Alice Johnson",
  "skills": ["Python", "React", "AWS"],
  "experience_years": 5,
  "projects": ["E-commerce Platform", "Healthcare Dashboard"],
  "availability": "available"
}
✅ Deliverables
 GitHub repository with all source code

 Live deployed demo

 README with setup, API, and AI documentation

 Realistic sample data for employees

 RAG pipeline for intelligent employee search

💬 Conclusion
This project demonstrates an intelligent HR chatbot capable of understanding natural queries, retrieving relevant employee data, and generating professional recommendations. It blends traditional semantic search with cutting-edge LLMs using a lightweight architecture — ready for rapid deployment and further expansion.

Built with ❤️ using Python, Flask, Azure OpenAI, and modern AI development tools.

yaml
Copy
Edit

---

✅ You can now **copy and paste** this entire section directly into your `README.md`. Let me know if you'd like to insert screenshots, badges, or contributor credits too.
