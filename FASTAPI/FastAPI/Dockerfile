
FROM tiangolo/uvicorn-gunicorn:python3.9
RUN mkdir /fastapi

COPY requirements.txt /fastapi

WORKDIR /fastapi

RUN pip install -r requirements.txt

COPY . /fastapi

EXPOSE 8000

CMD ["uvicorn", "mercari_2:api", "--host", "0.0.0.0", "--port", "8000"]