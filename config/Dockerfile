
FROM mtgupf/essentia:latest

# Install pip and additional Python packages
RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install librosa soundfile numpy scipy matplotlib

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Expose port for web interface
EXPOSE 8000

# Run the application
CMD ["python3", "main.py"]
