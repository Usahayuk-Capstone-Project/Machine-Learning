FROM python:3.8

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./Code_for_ML.ipynb /code/Code_for_ML.ipynb

COPY ./main.ipynb /code/main.ipynb

COPY ./data_usaha.csv /code/data_usaha.csv

COPY ./ccusahayuk-firebase-adminsdk-5cq5j-77c8c8f8a8.json /code/ccusahayuk-firebase-adminsdk-5cq5j-77c8c8f8a8.json

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
