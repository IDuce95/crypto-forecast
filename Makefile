clear_processed_data_dir:
	-rm ./data/processed/*.csv

run_docker_compose:
	- docker compose -f app/compose.yaml up --build

stop_docker_compose:
	- docker compose -f app/compose.yaml down

all_images = $(shell docker images -aq)
remove_all_images:
	- docker rmi -f ${all_images}

api:
	- uvicorn app.backend.main:app --port 5000 --reload

streamlit:
	- streamlit run app/frontend/main.py

clear_logs:

clear_models:

docs_server: