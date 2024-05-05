FROM python:3.12

COPY main.py /app/main.py
COPY model.py /app/model.py
COPY optimizer.py /app/optimizer.py
COPY flights_prophet-0.1.0.pkl /app/flights_prophet-0.1.0.pkl
COPY sales_catboost-0.1.0.pkl /app/sales_catboost-0.1.0.pkl
COPY requirements.txt /app/requirements.txt

WORKDIR /app

EXPOSE 8080

RUN pip install -r /app/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]