VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

test_all: test

$(VENV)/bin/activate: requirements.txt
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

test: $(VENV)/bin/activate
	$(PYTHON) -m pytest libs

.PHONY: test test_all

