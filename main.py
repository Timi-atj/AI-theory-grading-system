from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os
import re
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
print("Loaded API key:", api_key)

app = FastAPI()

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files like CSS/JS from /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html at root URL
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


# Input model for grading
class EvaluationInput(BaseModel):
    question: str
    real_answer: str
    student_answer: str
    keywords: Optional[List[str]] = []


# Function to send prompt to OpenRouter and parse structured result
async def grade_with_mistral(question: str, model_answer: str, student_answer: str, keywords: List[str]):
    keyword_str = ', '.join(keywords) if keywords else 'None'
                                                           
    prompt = f"""
You are an intelligent grading assistant. Grade the student's answer based on its accuracy, completeness,structure and smartness and relevance compared to the correct answer.

Follow these rules:
- Understand different ways the student may express the correct idea, even if the words used by student is not exactly the same.
- Be fair, very strict, and constructive.
- Highlight what was done well and what could be improved.
- Provide a score out of 10, a letter grade, and a brief, helpful feedback (1–2 sentences).
- If the student answer is completely incorrect or off-topic, clearly state so.

Output in this format:

Score: X/10  
Grade: (Letter Grade)  
Feedback: (Brief explanation of the grade)

Now evaluate this:

Question: {question}
Model Answer: {model_answer}
Student Answer: {student_answer}
Keywords: {keyword_str}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "TheoryMarkerAI",
        "Content-Type": "application/json"
    }

    body = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]

        # Extract score, grade, feedback using regex
        score_match = re.search(r"Score:\s*(\d+)/10", reply)
        grade_match = re.search(r"Grade:\s*([A-F][+-]?)", reply)
        feedback_match = re.search(r"Feedback:\s*(.*)", reply)

        return {
            "score": int(score_match.group(1)) if score_match else None,
            "grade": grade_match.group(1) if grade_match else None,
            "feedback": feedback_match.group(1).strip() if feedback_match else "No feedback found.",
            "raw_response": reply  # optional for debugging
        }


# POST route for grading
@app.post("/evaluate")
async def evaluate_answer(input: EvaluationInput):
    try:
        result = await grade_with_mistral(
            input.question,
            input.real_answer,
            input.student_answer,
            input.keywords
        )
        return {
            "score": result["score"],
            "grade": result["grade"],
            "feedback": result["feedback"],
            "raw_response": result["raw_response"]
        }
    except Exception as e:
        return {
            "error": "Grading failed. Please check your API key or internet connection.",
            "details": str(e)
        }
