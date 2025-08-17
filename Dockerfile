FROM python:3.11.5

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

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
    libxrender1 \
    libxtst6 \
    libdbus-1-3 \
    libfontconfig1 \
    libxshmfence1 \
    libgl1 \
    libegl1 \
    libopengl0 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install playwright && playwright install --with-deps chromium

COPY . .
# COPY .env .env  

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--timeout-keep-alive", "60", "--workers", "1"]