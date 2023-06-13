
install_requirements:
	pip install -r requirements.txt

streamlit:
	streamlit run api/frontend/app.py

run_api:
	uvicorn api.fast_api:app --host 0.0.0.0 --port $PORT
