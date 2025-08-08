# Use Python 3.11 as the base image
FROM python:3.11.5

# Install system dependencies required by Playwright
RUN apt-get update && apt-get install -y \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libx11-xcb1 \
    libxcursor1 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Install Playwright and Firefox browser
RUN pip install playwright && playwright install firefox

# Copy the rest of your application code
COPY . .

# Set environment variables
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.playwright-browsers
ENV PORT=10000

# Expose the port Render will use
EXPOSE $PORT

# Run the FastAPI application with Uvicorn using shell form
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 60 --workers 1"]