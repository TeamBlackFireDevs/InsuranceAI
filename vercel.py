from fastapi import FastAPI
from main import app
from asgiref.compatibility import guarantee_single_callable

# Convert FastAPI ASGI app to something Vercel's Python runtime understands
vercel_app = guarantee_single_callable(app)
