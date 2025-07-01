FROM python:3.10-slim
WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the application code
COPY . .

# expose the port the app runs on
CMD ["uvicorn", "mentor_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
