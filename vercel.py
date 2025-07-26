from fastapi import FastAPI
from main import app  # your FastAPI app
from mangum import Mangum

# Wrap FastAPI app with AWS Lambda-compatible adapter
handler = Mangum(app)
