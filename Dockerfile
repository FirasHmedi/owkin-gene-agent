FROM python:3.12-slim

# Links the published image to this repository on GitHub.
LABEL org.opencontainers.image.source=https://github.com/FirasHmedi/owkin-gene-agent

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gene_agent/ gene_agent/
COPY data/ data/
COPY streamlit_app.py cli.py ./

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
