from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

class QueryRequest(BaseModel):
    query: str

@app.post("/api/v1/insurance-claim")
async def process_claim(req: QueryRequest):
    prompt = (
        "You are an insurance claim processing assistant. "
        "Given the customer query, provide a decision (approved/rejected/undetermined), "
        "an amount (if applicable), and a justification. Output in JSON.\n\n"
        f"Customer Query: {req.query}"
    )

    payload = {
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct:novita",
        "messages": [
            {"role": "system", "content": "You are a helpful insurance claim processor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    try:
        message = response.json()["choices"][0]["message"]["content"]
        return {"result": message}
    except Exception as e:
        return {
            "error": "Failed to parse response",
            "details": str(e),
            "raw": response.text
        }
