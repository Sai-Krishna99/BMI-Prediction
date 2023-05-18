FROM python:3.10.9

# Install dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir tensorflow==2.10.0
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
