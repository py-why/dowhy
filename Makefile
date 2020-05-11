init:
	pip install -r requirements.txt  
test:
	py.test tests
check: 
	python setup.py check
sdist:
	python setup.py sdist
jupyter:
	jupyter nbconvert --to html do_why_simple_notebook.ipynb
.PHONY: init test check sdist
