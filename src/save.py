import os
import logging
import argparse
import torch

from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.lightning_module import PruneModule


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    parser.add_argument('ckpt_path', type=str, help='checkpoint')
    parser.add_argument('model_name', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()

    config = Config.from_yaml(args.config_file)
    model = PruneModule(config)

    ckpt_path = os.path.join(EXPERIMENTS_PATH, config.experiment_name, args.ckpt_path)
    torch.cuda.empty_cache()
    checkpoint = torch.load(str(ckpt_path))
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    student_model = model.student_model
    if student_model is None:
        logging.info('Чекпойнт не содержит запруненную модель')
    else:
        student_model.zero_grad()
        student_model.to('cpu')
        student_model.eval()
        torch.save(student_model, 'models/{0}'.format(args.model_name))
