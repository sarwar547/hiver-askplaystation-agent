FROM python:3.11-slim

WORKDIR /app

# Copy requirements if they exist
COPY requirements.txt* ./

# Install dependencies
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy application files
COPY . .

# Default command
CMD ["python", "-m", "main"]
