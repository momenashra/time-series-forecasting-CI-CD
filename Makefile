install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black *.py

lint:
	pylint --disable=R,C,W0621,E0102,E0611,E0401 *.py || true  # FIX: Replace spaces with TAB

deploy:
	echo "deployment begun!"

test:
	CUDA_VISIBLE_DEVICES=-1 coverage run -m pytest -v test_file.py && coverage report

all: install lint test format deploy
