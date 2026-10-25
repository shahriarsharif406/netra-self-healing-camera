# Multi-arch (amd64 and arm64/Graviton). opencv-python-headless has aarch64 wheels.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/tmp STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt pandas
COPY netra ./netra
COPY app.py .
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"
USER 1000
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
