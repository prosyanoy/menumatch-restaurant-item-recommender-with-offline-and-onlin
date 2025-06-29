FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml ./
COPY menumatch ./menumatch
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu .
COPY dags ./dags
EXPOSE 8000
CMD ["uvicorn", "menumatch.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
