FROM python:3.8

WORKDIR /usr/src/app

RUN pip3 install --no-cache-dir python-dotenv pymysql tensorflow-recommenders --upgrade tensorflow-datasets  numpy pandas flask

COPY ./Docker/wsgi.ini ./
COPY . .

CMD ["uwsgi", "--ini", "wsgi.ini"]
