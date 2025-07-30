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

api_v2:
	- uvicorn app.backend.main_v2:app --port 5000 --reload

streamlit:
	- streamlit run app/frontend/main.py

test_api:
	- python test_api_v2.py

clear_logs:
	- rm -f logs/*.log app/logs/*.log

clear_models:
	- rm -f models/*.pkl

docs_server:
	- mkdocs serve