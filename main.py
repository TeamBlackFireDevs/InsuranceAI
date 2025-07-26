from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, Dict

import os
import openai

app = FastAPI()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

class QueryRequest(BaseModel):
    query: str

def query_llm(user_text: str) -> Dict[str, Any]:
    # This function sends the query to OpenAI GPT-3.5/4 for processing
    # In production, you should design prompts to guide the LLM; here, we illustrate a basic call.
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

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if your OpenAI account has access
        messages=[
            {"role": "system", "content": "You are a helpful insurance claim processor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=450,
        n=1,
        stop=None,
    )
    # Extract the JSON from the LLM's response
    try:
        import json
        # Find and load the first JSON block in the response
        content = response['choices'][0]['message']['content']
        start = content.find("{")
        end = content.rfind("}") + 1
        result_json = json.loads(content[start:end])
        return result_json
    except Exception as e:
        return {
            "decision": "UNDETERMINED",
            "justification": f"Could not parse LLM response: {str(e)}"
        }

@app.post("/api/v1/insurance-claim")
async def process_claim(req: QueryRequest):
    result = query_llm(req.query)
    return result
