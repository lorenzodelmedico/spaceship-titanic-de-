
########################################################################################################################
# Project setup
########################################################################################################################

init_env : init_virtualenv load_direnv install precommit_install
	@echo "✅ Environment initialized and ready to use 🔥"

init_virtualenv :
	@echo "Initializing environment ..."
	@if pyenv virtualenvs | grep -q 'titanic-env'; then \
		echo "Virtualenv 'titanic-env' already exists"; \
	else \
		echo "Virtualenv 'titanic-env' does not exist"; \
		echo "Creating virtualenv 'titanic-env' ..."; \
		pyenv virtualenv 3.10.12 titanic-env; \
	fi
	@pyenv local titanic-env
	@echo "✅ Virtualenv 'titanic-env' activated"

load_direnv:
	@echo "Loading direnv ..."
	@direnv allow
	@echo "✅ Direnv loaded"

precommit_install:
	@echo "Installing pre-commit hooks ..."
	@pre-commit install
	@echo "✅ Pre-commit hooks installed"

install :
	@echo "Installing dependencies ..."
	@pip install --upgrade -q pip
	@pip install -q -r requirements.txt
	@echo "✅ Dependencies installed"
	@echo "Installing local package titanic ..."
	@tree src
	@pip install -q -e .


########################################################################################################################
# Training the model
########################################################################################################################

#.PHONY: train
#train:
#	@echo "Training the model ..."
#	python -m titanic.main
#	@echo "✅ Model trained"