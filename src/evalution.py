import logging
import torch
from transformers import AutoTokenizer

from src.train import arg_parse
from src.config import Config

from encodechka_eval import tasks
from encodechka_eval.bert_embedders import embed_bert_both

from importlib import reload
reload(tasks)


def evaluate(model: torch.nn.Module, tokenizer):
    sent_tasks = [TaskClass() for TaskClass in tasks.SENTENCE_TASKS]
    speed_task_cpu = tasks.SpeedTask()
    speed_task_gpu = tasks.SpeedTask()

    model_name = model.name_or_path
    model.cuda()
    for task in sent_tasks:
        task.eval(lambda x: embed_bert_both(x, model, tokenizer), model_name)

    speed_task_gpu.eval(lambda x: embed_bert_both(x, model, tokenizer), model_name)

    model.cpu()
    speed_task_cpu.eval(lambda x: embed_bert_both(x, model, tokenizer), model_name)

    scores = {type(t).__name__: t.score_cache[model_name] for t in sent_tasks}
    scores['cpu_speed'] = speed_task_cpu.score_cache[model_name]
    scores['gpu_speed'] = speed_task_gpu.score_cache[model_name]

    return scores


if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=logging.INFO)

    config = Config.from_yaml(args.config_file)

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    model = torch.load('models/prune_model.pth').cuda()

    scores = evaluate(model, tokenizer)
    print(scores)
