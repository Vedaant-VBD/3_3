import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# ==================================================
# FastAPI App
# ==================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# Models
# ==================================================

class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ==================================================
# Tool Function
# ==================================================

def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


# ==================================================
# AI PIPE ERROR ANALYSIS
# ==================================================

def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:

    client = OpenAI(
        api_key=os.environ.get("AIPIPE_API_KEY"),
        base_url=os.environ.get("AIPIPE_BASE_URL")
    )

    prompt = f"""
Analyze this Python code and its traceback.
Return ONLY JSON in this format:

{{ "error_lines": [line_numbers] }}

CODE:
{code}

TRACEBACK:
{traceback_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or whatever AI Pipe maps to
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    result = ErrorAnalysis.model_validate_json(
        response.choices[0].message.content
    )

    return result.error_lines


# ==================================================
# Endpoint
# ==================================================

@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    if execution["success"]:
        return CodeResponse(
            error=[],
            result=execution["output"]
        )

    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return CodeResponse(
        error=error_lines,
        result=execution["output"]
    )
