FROM python:3.12

WORKDIR /transformation-services

COPY requirements.txt requirements.txt
RUN pip install --upgrade -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]