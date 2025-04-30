FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m streamlit_user

RUN mkdir -p /app/temp && \
    chown streamlit_user:streamlit_user /app/temp

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8501 8000

ENV PYTHONPATH=/app
ENV TEMP_DIR=/app/temp

USER streamlit_user
CMD ["streamlit", "run", "app/streamlit_app.py"]
