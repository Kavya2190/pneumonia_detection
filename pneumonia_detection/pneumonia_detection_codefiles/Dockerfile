# Use a lightweight stable Python base image optimized for general use
FROM python:3.10-slim

# Establish the working directory inside the container runtime environment
WORKDIR /app

# Copy only the requirements configuration file to leverage build caching
COPY requirements.txt .

# Upgrade pip and install the defined Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the remaining application source code and resources into the workspace
COPY . .

# Expose the default networking port used by Streamlit applications
EXPOSE 8501

# Define the runtime execution command with options optimized for cloud deployments
CMD ["streamlit", "run", "app_standalone.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
