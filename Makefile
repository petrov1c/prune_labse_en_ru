.PHONY: train
train:
	PYTHONPATH=. python src/train.py config/config.yml

.PHONY: infer
infer:
	PYTHONPATH=. python src/evalution.py config/config.yml

.PHONY: save
save:
	PYTHONPATH=. python src/save.py experiments/exp1_efficientnet_b3/epoch_epoch=09-val_f1=0.643.ckpt

.PHONY: lint
lint:
	flake8 src/*.py

