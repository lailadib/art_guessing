FROM python:3.10.6-buster
WORKDIR /prod

# Copy requirements
COPY requirements.txt requirements.txt

# Copy dog_prediction logic
COPY backend backend
COPY setup.py setup.py

# install all
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv

# Copy the model to container
COPY models models

# Copy params to container
COPY params.py params.py

CMD uvicorn backend.api.fast_api:app --host 0.0.0.0 --port $PORT
