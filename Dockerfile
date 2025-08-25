# ----------------------
# Stage 1: Build React frontend
# ----------------------
FROM node:18 AS frontend-builder

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install

COPY frontend/ .
RUN npm run build

# ----------------------
# Stage 2: Python backend + final image
# ----------------------
FROM python:3.10

WORKDIR /code

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy backend code
COPY . .

# copy built frontend into Flask static folder
COPY --from=frontend-builder /frontend/build /code/frontend/build

EXPOSE 7860

CMD ["python", "app.py"]
