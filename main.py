# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
import os
import re
from dotenv import load_dotenv

# --- Config & App -----------------------------------------------------------
# Why: keep runs reproducible and mildly flexible without multi-pass.
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_TOP_P: float = 0.9
DEFAULT_MAX_TOKENS: int = 256
DEFAULT_SEED: int = 42  # May be ignored by some models; harmless if unsupported.
MODEL_NAME: str = "mistralai/mistral-7b-instruct"
OPENROUTER_URL: str = "https://openrouter.ai/api/v1/chat/completions"

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY", "")

# Avoid printing full secrets in logs
if api_key:
    masked = api_key[:4] + "..." + api_key[-4:]
    print(f"Loaded OpenRouter API key: {masked}")
else:
    print("WARNING: OPENROUTER_API_KEY not set.")

app = FastAPI()

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional static hosting
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    index_path = os.path.join("static", "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "FastAPI server is running."}


# --- Models -----------------------------------------------------------------
class EvaluationInput(BaseModel):
    question: str
    real_answer: str
    student_answer: str
    keywords: List[str] = Field(default_factory=list)


# --- Helpers ----------------------------------------------------------------
SCORE_RE = re.compile(r"Score\s*:\s*(?P<score>\d+(?:\.\d+)?)\s*/\s*10", re.IGNORECASE)
GRADE_RE = re.compile(r"Grade\s*:\s*(?P<grade>[A-F][+-]?)", re.IGNORECASE)
FEEDBACK_RE = re.compile(r"Feedback\s*:\s*(?P<feedback>.+)", re.IGNORECASE | re.DOTALL)


def clamp_score(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, value))


def infer_grade(score10: float) -> str:
    """Why: ensure we still return a grade if the model omits it."""
    s = clamp_score(score10)
    if s >= 9.0:
        return "A"
    if s >= 8.0:
        return "B"
    if s >= 7.0:
        return "C"
    if s >= 6.0:
        return "D"
    if s >= 5.0:
        return "E"
    return "F"


def build_messages(question: str, model_answer: str, student_answer: str, keywords: List[str]) -> List[Dict[str, str]]:
    keyword_str = ", ".join(keywords) if keywords else "None"

    system_prompt = (
        "You are a professional, fair WAEC examiner. Grade consistently based on accuracy, "
        "completeness, clarity, structure, technical vocabulary, and depth of reasoning, comparing "
        "the student to the model answer.\n\n"
        "Rules:\n"
        "- No credit for vague/overly general responses.\n"
        "- Assess only content; ignore tone/effort.\n"
        "- Accept alternative phrasing if fully correct.\n"
        "- Penalize missing key terms, incorrect reasoning, or incomplete thoughts.\n"
        "- Reward clear, well-structured, technically accurate responses.\n"
        "- If meaning matches the model answer, allow 10/10 even if phrased differently.\n\n"
        "Output format (STRICT, nothing else):\n"
        "Score: X/10\n"
        "Grade: <A-F>\n"
        "Feedback: <1-2 sentence academic feedback>\n"
    )

    user_prompt = (
        f"Question: {question}\n"
        f"Model Answer: {model_answer}\n"
        f"Student Answer: {student_answer}\n"
        f"Keywords: {keyword_str}\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def call_openrouter(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",  # Why: identify your app per OpenRouter guidelines.
        "X-Title": "TheoryMarkerAI",
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "seed": DEFAULT_SEED,  # May be ignored by some routes; improves repeatability if supported.
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10)) as client:
        resp = await client.post(OPENROUTER_URL, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()


def parse_model_reply(reply_text: str) -> Dict[str, Any]:
    score_match = SCORE_RE.search(reply_text)
    grade_match = GRADE_RE.search(reply_text)
    feedback_match = FEEDBACK_RE.search(reply_text)

    score_val: Optional[float] = None
    if score_match:
        try:
            score_val = float(score_match.group("score"))
        except ValueError:
            score_val = None

    if score_val is None:
        # Fallback: try to find any number 0-10 as last resort
        nums = re.findall(r"\b(10|[0-9](?:\.[0-9])?)\b", reply_text)
        if nums:
            try:
                score_val = float(nums[0])
            except ValueError:
                score_val = None

    if score_val is None:
        score_val = 0.0

    score_val = clamp_score(score_val)
    # Round half up for stability on boundaries
    score_int = int(round(score_val))

    grade_str = grade_match.group("grade") if grade_match else infer_grade(score_val)

    feedback_str = (
        feedback_match.group("feedback").strip() if feedback_match else "No feedback found."
    )

    # Keep single-line feedback short
    feedback_str = feedback_str.splitlines()[0].strip()

    return {
        "score": score_int,
        "grade": grade_str,
        "feedback": feedback_str,
        "raw_response": reply_text,
    }


# --- Core grading -----------------------------------------------------------
async def grade_with_mistral(question: str, model_answer: str, student_answer: str, keywords: List[str]) -> Dict[str, Any]:
    messages = build_messages(question, model_answer, student_answer, keywords)
    data = await call_openrouter(messages)

    # Defensive parsing of OpenRouter shape
    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response shape: {exc}; payload keys: {list(data.keys())}")

    return parse_model_reply(reply)


# --- API Routes -------------------------------------------------------------
@app.post("/evaluate")
async def evaluate_answer(input: EvaluationInput):
    if not api_key:
        return {
            "error": "Server misconfiguration: OPENROUTER_API_KEY is missing.",
        }

    try:
        result = await grade_with_mistral(
            input.question,
            input.real_answer,
            input.student_answer,
            input.keywords,
        )
        return result
    except httpx.HTTPStatusError as e:
        return {
            "error": "OpenRouter returned an error.",
            "status_code": e.response.status_code if e.response else None,
            "details": str(e),
        }
    except httpx.HTTPError as e:
        return {
            "error": "Network error while contacting OpenRouter.",
            "details": str(e),
        }
    except Exception as e:
        return {
            "error": "Grading failed due to an unexpected error.",
            "details": str(e),
        }
