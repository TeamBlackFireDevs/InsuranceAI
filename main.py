from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in Vercel env vars
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

class QueryRequest(BaseModel):
    query: str

@app.post("/api/v1/insurance-claim")
async def process_claim(req: QueryRequest):
    prompt = (
        "You are an insurance claim processing assistant. "
        "Given the customer query, provide a decision (approved/rejected/undetermined) "
        "with explanation in JSON format.\n\n"
        f"Customer Query: {req.query}\n\n"
    )

    payload = {
        "inputs": prompt,
        "parameters": {"temperature": 0.3, "max_new_tokens": 300}
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    try:
        output_text = response.json()[0]["generated_text"]
        return {"result": output_text}
    except Exception as e:
        return {
            "error": "Failed to parse response",
            "details": str(e),
            "raw": response.text
        }
