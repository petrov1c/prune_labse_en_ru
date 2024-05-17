import argparse
import logging
import torch
from transformers import AutoTokenizer
from importlib import reload

from encodechka_eval import tasks
from encodechka_eval.bert_embedders import embed_bert_both

from src.config import Config


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    parser.add_argument('model_name', type=str)
    return parser.parse_args()


def evaluate(model: torch.nn.Module, tokenizer):
    reload(tasks)

    sent_tasks = [TaskClass() for TaskClass in tasks.SENTENCE_TASKS]
    speed_task_cpu = tasks.SpeedTask()
    speed_task_gpu = tasks.SpeedTask()

    model_name = model.name_or_path
    model.eval()
    model.cpu()
    speed_task_cpu.eval(lambda x: embed_bert_both(x, model, tokenizer), model_name)

    model.cuda()
    speed_task_gpu.eval(lambda x: embed_bert_both(x, model, tokenizer), model_name)
    for task in sent_tasks:
        task.eval(lambda x: embed_bert_both(x, model, tokenizer), model_name)

    scores = {type(t).__name__: t.score_cache[model_name] for t in sent_tasks}
    scores['mean_s'] = sum([scores[key] for key in scores])/len(scores)
    scores['cpu_speed'] = speed_task_cpu.score_cache[model_name]
    scores['gpu_speed'] = speed_task_gpu.score_cache[model_name]

    return scores


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml(args.config_file)

    tokenizer_for_model = AutoTokenizer.from_pretrained(config.model.name)
    test_model = torch.load('models/{0}'.format(args.model_name))

    result = evaluate(test_model, tokenizer_for_model)
    logging.info(str(result))
