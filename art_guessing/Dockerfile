FROM python:3.10.6-buster
WORKDIR /prod

# Copy requirements
COPY requirements.txt requirements.txt

# Copy files and folders
COPY backend backend
COPY setup.py setup.py

# install all
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv

# Copy the model
COPY models models

# Copy params
COPY params.py params.py

CMD uvicorn backend.api.fast_api:app --host 0.0.0.0 --port $PORT
