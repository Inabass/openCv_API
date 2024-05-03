up:
	docker-compose up -d

down:
	docker-compose down

exec: 
	docker-compose exec python /bin/bash

api:
	docker-compose exec python uvicorn api.main:app --reload --host 0.0.0.0 --port 9004

gaf:
	docker-compose exec python python3 GAF.py ../input/train/abeA.jpeg

gfd:
	docker-compose exec python python3 GFD.py ../output/crop/face001.jpg 

fr:
	docker-compose exec python python3 FR.py ../input/test/abeB.jpeg