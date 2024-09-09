# Base image with Python and PyTorch
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download English and German models for spaCy
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download de_core_news_sm

# # Copy the rest of the application code into the container
COPY . .

# # Command to run your application
ENTRYPOINT ["python", "main.py"]
