# Use an official Python runtime as a base image.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgomp1

ADD . .
# Set environment variables to prevent Python from buffering stdout/stderr.
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container.
WORKDIR /

# Copy dependency lists first for caching benefits.
COPY requirements.txt .

# Install dependencies.
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
# If your primary entry point is the Flask application (e.g., flask_app.py), 
# make sure to update the command below accordingly.
CMD ["python", "flask_app_UI.py"]
