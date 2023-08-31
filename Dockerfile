FROM selenium/standalone-chrome:latest

# Install Python3 and pip3 debians
RUN sudo apt-get update && sudo apt-get install -y python3 python3-pip

# Set display port and dbus for Chrome to run headlessly
ENV DISPLAY=:99

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy project
COPY . .

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run jupyter lab when the container launches
CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
