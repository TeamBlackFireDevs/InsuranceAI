from main import app
from asgiref.compatibility import guarantee_single_callable

vercel_app = guarantee_single_callable(app)
