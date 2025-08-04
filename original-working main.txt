from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import httpx
import json
import os
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


# Function to send prompt to OpenRouter
async def grade_with_mistral(question: str, model_answer: str, student_answer: str, keywords: List[str]):
    keyword_str = ', '.join(keywords) if keywords else 'None'

    prompt = f"""
You are an exam grader. Grade the student’s answer based on:

1. The question asked
2. The model answer (as a reference)
3. The presence of these required keywords: {keyword_str}

Your grading rules:
- The student MUST mention all important keywords or concepts listed.
- Do NOT rely solely on similarity to the model answer.
- Penalize any missing or incomplete keyword/concept clearly.
- You must extract which keywords were found vs missing.

Return:
- score (0 to 10, deduct for missing keywords)
- grade (A, B, C, D, F)
- feedback (brief, max 2 lines)
- found_keywords (list)
- missing_keywords (list)

Respond ONLY in this JSON format:
{{
  "score": 6,
  "grade": "C",
  "feedback": "The student missed important terms like 'life'.",
  "found_keywords": ["study"],
  "missing_keywords": ["life"]
}}

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
        return json.loads(reply)


# POST route for evaluation
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
            "found_keywords": result.get("found_keywords", []),
            "missing_keywords": result.get("missing_keywords", [])
        }
    except Exception as e:
        return {
            "error": "Grading failed. Please check your API key or internet connection.",
            "details": str(e)
        }
