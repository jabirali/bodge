fmt:
	black --line-length 99 .

test:
	clear
	python -m pytest tests
