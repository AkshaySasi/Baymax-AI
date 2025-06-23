# Baymax.AI - Your Personal Mindcare Companion

## Overview
Baymax.AI is an innovative conversational AI designed to be a compassionate, emotionally intelligent companion. Built using the Gemini 2.5 Flash model from Google, this project aims to provide users with a supportive mindcare experience, adapting and learning from each conversation to offer personalized comfort. Unlike traditional AI chatbots that rely on next-word prediction, Baymax.AI simulates independent thinking, fostering a human-like connection while maintaining strict ethical standards.

This project was inspired by the need for AI agents that prioritize emotional intelligence and companionship, addressing the limitations highlighted in recent research like "The Illusion of Thinking," which critiques the predictive nature of most AI models.

## Features
- **Emotionally Intelligent Responses**: Detects and responds to user emotions with empathy and care.
- **Self-Learning Framework**: Evolves with each conversation to enhance comfort strategies.
- **Human-Like Dialogue**: Starts with a warm introduction, checks on your day, food, and water, and maintains natural conversation flow.
- **Ethical Design**: Respects privacy with local data storage and offers opt-out options for learning data.
- **Personalized Support**: Adapts to individual user preferences over time.

## Tools and Technologies
- **Python 3.12**: Core language for development and execution.
- **Google GenerativeAI**: Powers the Gemini 2.5 Flash model for conversational capabilities.
- **Transformers**: Utilizes BERT-based `EmotionDetector` for emotion recognition on CUDA-enabled GPUs.
- **uv**: Virtual environment and package management.
- **FastAPI**: Backend server framework with CORS support.
- **NumPy**: Supports numerical operations for learning adjustments.
- **Git**: Version control for project management.

## Installation and Setup

### Prerequisites
- **Operating System**: Windows 10/11 (tested on your setup at `C:\Akshay\baymax.ai\backend`).
- **Python 3.12**: Ensure Python is installed (download from [python.org](https://www.python.org/downloads/)).
- **Git**: Install from [git-scm.com](https://git-scm.com/downloads) for version control.
- **GPU (Optional)**: NVIDIA GeForce GTX 1070 Ti or similar with CUDA support for faster emotion detection.
- **Internet Connection**: Required for Gemini API access.

### Steps to Implement

1. **Clone the Repository**
   - Open a terminal (Command Prompt) and run:
     ```cmd
     git clone https://github.com/yourusername/baymax.ai.git
     cd baymax.ai\backend

2. **Set Up Virtual Environment**
Create and activate a virtual environment using uv:

uv venv
venv\Scripts\activate

3. **Install Dependencies**
Install required packages from requirements.txt:

uv pip install -r requirements.txt

Ensure requirements.txt contains:

google-generativeai==0.7.2
torch==2.3.0+cu121
numpy==1.26.4
appdirs==1.4.4
transformers==4.44.2
fastapi==0.111.0
uvicorn==0.30.1
python-dotenv==1.0.1

4. **Configure API Key**
Obtain a Gemini API key from Google AI Studio.
Create a .env file in the backend directory:

GEMINI_API_KEY=your_api_key_here

Alternatively, set it in the terminal:

set GEMINI_API_KEY=your_api_key_here

5. **Run the Backend**
Start the FastAPI server:

uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

6. **Set Up Frontend**
Navigate to the frontend directory:

cd ..\frontend
npm install
npm run dev

**Access the app at http://localhost:5173.**

**Usage**
Interact with Baymax.AI as you would a friend. It will introduce itself on first contact and check in on your well-being.
Respond naturally; Baymax will adapt and learn from your inputs over time.
Manage memory via the UI (book icon) or disable it in settings if desired.

**Contributing**
Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Open a pull request.


**Acknowledgments**
Inspired by the "The Illusion of Thinking" paper for moving beyond predictive AI.
Built with support from xAI’s Grok 3 and Google’s Gemini API.
Thanks to the open-source community for tools like FastAPI and Transformers.

**Happy Coding**
