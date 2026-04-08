# Start from a Python 3.10 base image
FROM python:3.10-slim

# Install system dependencies (like Poppler) as ROOT
# This is the new/added section
RUN apt-get update && apt-get install -y poppler-utils --no-install-recommends && \
    # Clean up the apt cache to keep the final image small
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Copy the requirements file and install dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of your application code
COPY --chown=appuser:appuser . .

# Expose the port (Hugging Face Spaces default to 7860)
EXPOSE 7860

ENV PATH="/home/appuser/.local/bin:$PATH"

# Command to run your FastAPI app with Gunicorn
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "300", "--bind", "0.0.0.0:7860", "app:app"]