import os
from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import re

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = '7c0d2e2f35e020ec485f18271ca26451'

# Azure OpenAI Configuration
client = AzureOpenAI(
    api_key="99pNnIIEnYGr7klx9lre5slwp1AJ2WvjJJrtQsAHlvTBpQF7vZBJQQJ99BFACHYHv6XJ3w3AAAAACOG6WvB",
    api_version="2024-06-01",
    azure_endpoint="https://deepi-mbmwweg-eastus2.cognitiveservices.azure.com"
)

# Sample Employee Dataset
EMPLOYEES_DATA = {
    "employees": [
        {
            "id": 1,
            "name": "Alice Johnson",
            "skills": ["Python", "React", "AWS", "Docker", "PostgreSQL"],
            "experience_years": 5,
            "projects": ["E-commerce Platform", "Healthcare Dashboard", "Supply Chain Analytics"],
            "availability": "available",
            "department": "Engineering",
            "specialization": "Full Stack Development"
        },
        {
            "id": 2,
            "name": "Dr. Sarah Chen",
            "skills": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Computer Vision"],
            "experience_years": 6,
            "projects": ["Medical Diagnosis Platform", "X-ray Analysis System", "Patient Risk Prediction"],
            "availability": "available",
            "department": "AI/ML",
            "specialization": "Healthcare AI"
        },
        {
            "id": 3,
            "name": "Michael Rodriguez",
            "skills": ["Python", "scikit-learn", "pandas", "SQL", "Machine Learning"],
            "experience_years": 4,
            "projects": ["Patient Risk Prediction System", "Healthcare Analytics", "Medical Data Processing"],
            "availability": "busy",
            "department": "Data Science",
            "specialization": "Healthcare Analytics"
        },
        {
            "id": 4,
            "name": "Emily Davis",
            "skills": ["React Native", "JavaScript", "TypeScript", "Redux", "Firebase"],
            "experience_years": 3,
            "projects": ["Mobile Banking App", "Social Media Platform", "Food Delivery App"],
            "availability": "available",
            "department": "Mobile Development",
            "specialization": "Mobile App Development"
        },
        {
            "id": 5,
            "name": "James Wilson",
            "skills": ["AWS", "Docker", "Kubernetes", "Jenkins", "Terraform"],
            "experience_years": 7,
            "projects": ["Cloud Migration", "DevOps Pipeline", "Infrastructure Automation"],
            "availability": "available",
            "department": "DevOps",
            "specialization": "Cloud Infrastructure"
        },
        {
            "id": 6,
            "name": "Lisa Thompson",
            "skills": ["Java", "Spring Boot", "Microservices", "Apache Kafka", "Redis"],
            "experience_years": 6,
            "projects": ["Banking System", "Payment Gateway", "Real-time Analytics"],
            "availability": "available",
            "department": "Backend Development",
            "specialization": "Enterprise Systems"
        },
        {
            "id": 7,
            "name": "Robert Kim",
            "skills": ["React", "Node.js", "GraphQL", "MongoDB", "Express.js"],
            "experience_years": 4,
            "projects": ["Social Platform", "Content Management System", "API Gateway"],
            "availability": "busy",
            "department": "Full Stack Development",
            "specialization": "Modern Web Development"
        },
        {
            "id": 8,
            "name": "Jennifer Lee",
            "skills": ["Python", "Django", "REST API", "PostgreSQL", "Celery"],
            "experience_years": 5,
            "projects": ["HR Management System", "Project Management Tool", "Inventory System"],
            "availability": "available",
            "department": "Backend Development",
            "specialization": "Enterprise Applications"
        },
        {
            "id": 9,
            "name": "David Brown",
            "skills": ["Angular", "TypeScript", "RxJS", "NgRx", "Jest"],
            "experience_years": 4,
            "projects": ["Enterprise Dashboard", "Financial Planning Tool", "Reporting System"],
            "availability": "available",
            "department": "Frontend Development",
            "specialization": "Enterprise Frontend"
        },
        {
            "id": 10,
            "name": "Maria Garcia",
            "skills": ["Vue.js", "Nuxt.js", "JavaScript", "CSS3", "SASS"],
            "experience_years": 3,
            "projects": ["E-commerce Frontend", "Portfolio Website", "Admin Dashboard"],
            "availability": "available",
            "department": "Frontend Development",
            "specialization": "Modern Frontend"
        },
        {
            "id": 11,
            "name": "Alex Turner",
            "skills": ["Python", "FastAPI", "SQLAlchemy", "Redis", "Docker"],
            "experience_years": 3,
            "projects": ["API Development", "Microservices Architecture", "Performance Optimization"],
            "availability": "available",
            "department": "Backend Development",
            "specialization": "API Development"
        },
        {
            "id": 12,
            "name": "Sophie Martin",
            "skills": ["Data Science", "Python", "R", "Tableau", "Power BI"],
            "experience_years": 4,
            "projects": ["Sales Analytics", "Customer Segmentation", "Predictive Analytics"],
            "availability": "busy",
            "department": "Data Science",
            "specialization": "Business Analytics"
        },
        {
            "id": 13,
            "name": "Carlos Mendez",
            "skills": ["React", "Next.js", "TypeScript", "Tailwind CSS", "Vercel"],
            "experience_years": 2,
            "projects": ["Company Website", "Marketing Dashboard", "Blog Platform"],
            "availability": "available",
            "department": "Frontend Development",
            "specialization": "Modern Web Technologies"
        },
        {
            "id": 14,
            "name": "Rachel Green",
            "skills": ["Python", "Flask", "SQLAlchemy", "JWT", "Unit Testing"],
            "experience_years": 3,
            "projects": ["Authentication System", "User Management", "API Security"],
            "availability": "available",
            "department": "Backend Development",
            "specialization": "Security & Authentication"
        },
        {
            "id": 15,
            "name": "Kevin O'Connor",
            "skills": ["Go", "gRPC", "Protocol Buffers", "Microservices", "Docker"],
            "experience_years": 5,
            "projects": ["High-performance APIs", "Distributed Systems", "Load Balancing"],
            "availability": "available",
            "department": "Backend Development",
            "specialization": "High-performance Systems"
        }
    ]
}

class UniversalHRChatbot:
    def __init__(self):
        self.employees = EMPLOYEES_DATA["employees"]
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self._prepare_employee_vectors()
        self.conversation_history = []
    
    def _prepare_employee_vectors(self):
        """Prepare employee text representations for similarity search"""
        self.employee_texts = []
        for emp in self.employees:
            # Create comprehensive text representation
            text = f"{emp['name']} {' '.join(emp['skills'])} {' '.join(emp['projects'])} {emp['specialization']} {emp['department']}"
            text += f" {emp['experience_years']} years experience {emp['availability']}"
            self.employee_texts.append(text)
        
        # Fit vectorizer on employee texts
        self.employee_vectors = self.vectorizer.fit_transform(self.employee_texts)
    
    def is_hr_related_query(self, query):
        """Determine if query is HR/employee related"""
        hr_keywords = [
            'employee', 'staff', 'team', 'developer', 'engineer', 'programmer',
            'skills', 'experience', 'project', 'available', 'busy', 'hire',
            'recruit', 'talent', 'expertise', 'department', 'backend', 'frontend',
            'fullstack', 'devops', 'data science', 'machine learning', 'python',
            'react', 'javascript', 'java', 'aws', 'docker', 'find someone',
            'who can', 'looking for', 'need someone', 'team member', 'specialist',
            'expert', 'professional', 'work on', 'build', 'develop', 'create'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in hr_keywords)
    
    def retrieve_relevant_employees(self, query, top_k=5):
        """Retrieve most relevant employees based on query similarity"""
        try:
            # Transform query using fitted vectorizer
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.employee_vectors).flatten()
            
            # Get top-k most similar employees
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            relevant_employees = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    employee = self.employees[idx].copy()
                    employee['similarity_score'] = float(similarities[idx])
                    relevant_employees.append(employee)
            
            return relevant_employees
        except Exception as e:
            print(f"Error in retrieve_relevant_employees: {e}")
            return []
    
    def generate_hr_response(self, query, relevant_employees):
        """Generate HR-specific response using Azure OpenAI"""
        try:
            # Prepare context for the LLM
            context = ""
            if relevant_employees:
                context = "Employee Database Information:\n"
                for emp in relevant_employees:
                    context += f"\n**{emp['name']}**\n"
                    context += f"- Skills: {', '.join(emp['skills'])}\n"
                    context += f"- Experience: {emp['experience_years']} years\n"
                    context += f"- Projects: {', '.join(emp['projects'])}\n"
                    context += f"- Department: {emp['department']}\n"
                    context += f"- Specialization: {emp['specialization']}\n"
                    context += f"- Availability: {emp['availability']}\n"
                    context += f"- Match Score: {emp['similarity_score']:.2f}\n"
            
            # Create system prompt
            system_prompt = """You are an intelligent HR assistant chatbot. Your role is to help HR teams find the right employees for projects and tasks.

Given a user query and relevant employee information, provide a helpful, professional response that:
1. Directly addresses the user's query
2. Recommends the most suitable employees (if applicable)
3. Explains why they're good fits
4. Highlights relevant skills and experience
5. Mentions availability status
6. Uses a friendly, professional tone

Format your response with clear structure using markdown when appropriate."""

            # Create user prompt
            user_prompt = f"User Query: {query}\n\n{context}\n\nPlease provide a comprehensive recommendation based on the query and employee information."
            
            # Call Azure OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content, relevant_employees
            
        except Exception as e:
            print(f"Error in generate_hr_response: {e}")
            return f"I found {len(relevant_employees)} relevant employees, but I'm having trouble generating a detailed response right now. Please try again.", relevant_employees
    
    def generate_general_response(self, query, conversation_context=""):
        """Generate general response for non-HR queries"""
        try:
            # Create system prompt for general conversation
            system_prompt = """You are a helpful, knowledgeable, and friendly AI assistant. You can help with a wide variety of topics including:

- General knowledge and information
- Technical questions and programming help
- Creative writing and brainstorming
- Problem-solving and analysis
- Educational explanations
- Casual conversation
- Professional advice
- And much more!

You should provide accurate, helpful, and engaging responses. When you don't know something, be honest about it. Always maintain a professional yet friendly tone.

Note: You also have access to HR/employee data when needed, but for general queries, focus on being a comprehensive AI assistant."""

            # Prepare conversation history for context
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add recent conversation history if available
            if conversation_context:
                messages.append({"role": "assistant", "content": f"Previous context: {conversation_context}"})
            
            messages.append({"role": "user", "content": query})
            
            # Call Azure OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.8,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in generate_general_response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or rephrase your question."
    
    def process_query(self, query, session_id=None):
        """Main processing pipeline: Handle both HR and general queries"""
        try:
            # Store conversation history (simplified - in production use proper session management)
            self.conversation_history.append({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            })
            
            # Keep only last 5 conversations for context
            if len(self.conversation_history) > 5:
                self.conversation_history = self.conversation_history[-5:]
            
            # Determine query type
            is_hr_query = self.is_hr_related_query(query)
            
            if is_hr_query:
                # Handle HR-related queries with RAG
                relevant_employees = self.retrieve_relevant_employees(query)
                response, employees_data = self.generate_hr_response(query, relevant_employees)
                
                return {
                    "response": response,
                    "query_type": "hr_related",
                    "relevant_employees": employees_data,
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Handle general queries
                # Create conversation context from recent history
                context = ""
                if len(self.conversation_history) > 1:
                    recent_queries = [item["query"] for item in self.conversation_history[-3:-1]]
                    context = " | ".join(recent_queries)
                
                response = self.generate_general_response(query, context)
                
                return {
                    "response": response,
                    "query_type": "general",
                    "relevant_employees": [],
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            print(f"Error in process_query: {e}")
            return {
                "response": "I'm sorry, I encountered an error processing your query. Please try again with a different phrasing.",
                "query_type": "error",
                "relevant_employees": [],
                "query": query,
                "timestamp": datetime.now().isoformat()
            }

# Initialize chatbot
chatbot = UniversalHRChatbot()

@app.route('/')
def index():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries - both HR and general"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Process query through enhanced pipeline
        result = chatbot.process_query(query, session_id)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/employees/search', methods=['GET'])
def search_employees():
    """Search employees endpoint"""
    try:
        query = request.args.get('q', '').strip()
        
        if not query:
            return jsonify({"employees": EMPLOYEES_DATA["employees"]})
        
        relevant_employees = chatbot.retrieve_relevant_employees(query, top_k=10)
        
        return jsonify({"employees": relevant_employees})
    
    except Exception as e:
        print(f"Error in search_employees endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/employees', methods=['GET'])
def get_all_employees():
    """Get all employees"""
    return jsonify(EMPLOYEES_DATA)

@app.route('/chat/types', methods=['GET'])
def get_supported_query_types():
    """Get information about supported query types"""
    return jsonify({
        "supported_types": [
            {
                "type": "hr_related",
                "description": "Employee search, team building, skill matching, project staffing",
                "examples": [
                    "Find a Python developer for a healthcare project",
                    "Who is available for frontend development?",
                    "I need someone with React and Node.js skills"
                ]
            },
            {
                "type": "general",
                "description": "General knowledge, technical help, creative tasks, problem solving",
                "examples": [
                    "Explain machine learning concepts",
                    "Help me write a professional email",
                    "What's the weather like?",
                    "Write a Python function to sort a list"
                ]
            }
        ]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
