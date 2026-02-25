from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
from io import StringIO
import traceback
import os
import re

from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str

class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        _globals = {}
        exec(code, _globals)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = sys.stdout.getvalue()
        output += traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


def fallback_error_analyzer(traceback_str: str) -> List[int]:
    matches = re.findall(r'File ".*?", line (\d+)', traceback_str)
    if matches:
        return [int(matches[-1])]
    return []


def analyze_error_with_ai(code: str, traceback_str: str) -> List[int]:
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not gemini_key:
        return fallback_error_analyzer(traceback_str)

    try:
        client = genai.Client(api_key=gemini_key)

        prompt = f"""
Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{traceback_str}
"""

        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "error_lines": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.INTEGER)
                        )
                    },
                    required=["error_lines"]
                )
            )
        )

        result = ErrorAnalysis.model_validate_json(response.text)
        return result.error_lines

    except Exception:
        return fallback_error_analyzer(traceback_str)


@app.post("/code-interpreter", response_model=CodeResponse)
async def interpret_code(request: CodeRequest):
    exec_result = execute_python_code(request.code)

    if exec_result["success"]:
        return CodeResponse(error=[], result=exec_result["output"])

    error_lines = analyze_error_with_ai(request.code, exec_result["output"])
    return CodeResponse(error=error_lines, result=exec_result["output"])
