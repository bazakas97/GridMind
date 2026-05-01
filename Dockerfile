FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    TZ=Europe/Athens

WORKDIR /app

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt /app/requirements-app.txt
RUN pip install --upgrade pip && pip install -r /app/requirements-app.txt

COPY app/ /app/app/
COPY src/ /app/src/
COPY outputs/metrics/ /app/outputs/metrics/

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
