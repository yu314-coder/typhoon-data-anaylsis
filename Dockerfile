FROM python:3.10-slim

WORKDIR /code

# Install system dependencies required for cartopy and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libgeos-c1v5 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create data directory and set permissions
RUN mkdir -p /code/data && \
    chmod 777 /code/data

# Create matplotlib config directory and set permissions
RUN mkdir -p /tmp/matplotlib && \
    chmod 777 /tmp/matplotlib

# Set matplotlib config directory
ENV MPLCONFIGDIR=/tmp/matplotlib

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV DATA_PATH=/code/data
ENV DASH_DEBUG_MODE=false
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 7860

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /code

# Switch to non-root user
USER appuser

# Start the application
CMD ["python", "app.py"]