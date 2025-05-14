
# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file to install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# In Dockerfile, after RUN pip install ...
RUN pip install ptvsd

# Copy the application code and data
COPY app/ ./app/
COPY data/ ./data/
COPY .env ./

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["python", "-m", "ptvsd", "--host", "0.0.0.0", "--port", "5678", "--wait", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]