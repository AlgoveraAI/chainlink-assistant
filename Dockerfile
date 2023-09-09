# Use the official Python 3.10 image as a base image
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y xvfb

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

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Create a directory for our application and set it as working directory
WORKDIR /app

# Copy over the rest of our application
COPY . /app/

# Run Jupyter Lab by default when the container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
