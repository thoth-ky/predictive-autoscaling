FROM python:3.11-slim

WORKDIR /app

# Install requirements
RUN pip install requests

# Copy the load generator script
COPY load_generator.py .

# Run baseline load by default
CMD ["python", "load_generator.py", "baseline", "--duration", "86400"]