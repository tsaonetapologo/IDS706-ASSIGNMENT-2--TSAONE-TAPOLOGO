install:
	python -m pip install -r requirements.txt

format:
	python -m black ecommerce_analysis.py test_ecommerce_analysis.py

lint:
	python -m flake8 ecommerce_analysis.py test_ecommerce_analysis.py

test:
	python -m pytest -v
