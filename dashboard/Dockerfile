FROM python:3.12-slim
WORKDIR /srv
RUN pip install --no-cache-dir streamlit==1.41.1 pandas==2.2.3 "psycopg[binary]==3.2.3" sqlalchemy==2.0.36
COPY dashboard/app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
