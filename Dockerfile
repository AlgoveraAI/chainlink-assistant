FROM python:3.10-slim

# Install essential libraries for Selenium, Chrome, the GPG key, and libxcb
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    gnupg \
    libxcb1 \
    libx11-xcb1 \
    libxrandr2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libsoundio2 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y xvfb

# macos specific
RUN apt-get install gcc python3-dev -y

ENV DISPLAY=:99

# Create the directory for Chrome
RUN mkdir -p /opt/google/chrome

# Download, unzip, and move Chrome binary to the previously created directory
RUN wget "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/116.0.5845.96/linux64/chrome-linux64.zip" \
    && unzip chrome-linux64.zip -d /opt/google/chrome \
    && rm chrome-linux64.zip \
    && ln -s /opt/google/chrome/chrome /usr/bin/google-chrome

# Download, unzip, and move ChromeDriver binary to /usr/local/bin/
RUN wget "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/116.0.5845.96/linux64/chromedriver-linux64.zip" \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm -r chromedriver-linux64 \
    && rm chromedriver-linux64.zip


WORKDIR /chainlink-assistant

COPY ./templates /chainlink-assistant/templates
COPY ./chat /chainlink-assistant/chat
COPY ./ingest /chainlink-assistant/ingest
COPY ./search /chainlink-assistant/search
COPY ./*.py /chainlink-assistant/
COPY ./requirements.txt /chainlink-assistant/requirements.txt
COPY ./.env /chainlink-assistant/
COPY ./data /chainlink-assistant/data

RUN pip install -r /chainlink-assistant/requirements.txt 
#--no-cache-dir 

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]