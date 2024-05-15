CONFIG:='config/config.yml'
CHECKPOINT:='epoch_epoch=04-val_total_loss=0.1080.ckpt'
MODEL_NAME:='tune_model.pth'

.PHONY: install
install:
	git clone https://github.com/avidale/encodechka
	PYTHONPATH=. python encodechka/setup.py install
	rm -rf encodechka

.PHONY: train
train:
	PYTHONPATH=. python src/train.py $(CONFIG)

.PHONY: infer
infer:
	PYTHONPATH=. python src/evalution.py $(CONFIG) $(MODEL_NAME)

.PHONY: save
save:
	PYTHONPATH=. python src/save.py $(CONFIG) $(CHECKPOINT) $(MODEL_NAME)

.PHONY: lint
lint:
	flake8 src/*.py

