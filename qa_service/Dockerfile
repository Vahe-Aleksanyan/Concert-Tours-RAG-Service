FROM python:3.11-slim
WORKDIR /code

# 1. install deps
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt
ENV PATH="/opt/venv/bin:$PATH"

# 2. copy service code
COPY qa_service /code/qa_service

# 3. start server
CMD ["uvicorn", "qa_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]