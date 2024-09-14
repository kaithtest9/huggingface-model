FROM python:3.11-slim-bookworm

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN python load_model.py

CMD ["python", "app.py"]