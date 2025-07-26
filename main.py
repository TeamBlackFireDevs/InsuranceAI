from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
from openai import OpenAI
import os
import json

# Initialize FastAPI app
app = FastAPI()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Request model
class QueryRequest(BaseModel):
    query: str

# Core logic to query OpenAI LLM
def query_llm(user_text: str) -> Dict[str, Any]:
    prompt = (
        "You are an insurance claim processing assistant. "
        "Given the following customer query, extract all relevant details (age, condition, policy duration, location, etc), "
        "retrieve matching rules from this Bajaj Allianz health insurance policy (waiting periods, exclusions, etc). "
        "Then output a JSON with: decision (approved/rejected/undetermined), amount (if applicable), and a justification "
        "(mapping each decision to the clause/rule). Use this format:\n\n"
        "{\n"
        '  "decision": "APPROVED",\n'
        '  "amount": "Up to Sum Insured as per Policy Schedule",\n'
        '  "justification": [\n'
        '    {\n'
        '      "criteria": "—",\n'
        '      "status": "PASSED/FAILED",\n'
        '      "explanation": "—",\n'
        '      "clause_reference": "—"\n'
        '    }\n'
        "  ]\n"
        "}\n\n"
        "Customer Query:\n"
        f"{user_text}\n"
        "Be concise and specific. If information is missing, set 'decision' to 'UNDETERMINED' and explain what is needed."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful insurance claim processor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=450,
        )

        content = response.choices[0].message.content
        start = content.find("{")
        end = content.rfind("}") + 1
        result_json = json.loads(content[start:end])
        return result_json

    except Exception as e:
        return {
            "decision": "UNDETERMINED",
            "justification": f"Could not parse LLM response: {str(e)}"
        }

# API route
@app.post("/api/v1/insurance-claim")
async def process_claim(req: QueryRequest):
    result = query_llm(req.query)
    return result
