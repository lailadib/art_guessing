############### PACKAGE ACTIONS ######################
reinstall_package:
	@pip install -y art-guessing || :
	@pip install -e .

install_requirements:
	pip install -r requirements.txt

streamlit:
	streamlit run api/frontend/app.py

run_api:
	uvicorn backend.api.fast_api:app --host 0.0.0.0 --port $PORT

docker_build:
	docker build -t ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod .

docker_push:
	docker push ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod

docker_run:
	docker run -e PORT=8000 -p 8000:8000 --env-file .env ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod

docker_interactive:
	docker run -it --env-file .env ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod /bin/bash

docker_deploy:
	gcloud run deploy --image ${GCR_REGION}/${GCP_PROJECT}/${GCR_IMAGE}:prod --memory ${GCR_MEMORY} --region ${GCP_REGION} --env-vars-file .env.yaml
