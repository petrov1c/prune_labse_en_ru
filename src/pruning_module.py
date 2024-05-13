import logging
import torch_pruning as tp
import torch
from transformers.models.bert.modeling_bert import BertSelfAttention

from src.datamodule import SentenceDM
from src.lightning_module import PruneModule

from copy import deepcopy
from tqdm import tqdm


def prune_model(lightning_module: PruneModule, datamodule: SentenceDM):
    device = 'cuda' if lightning_module.config.accelerator == 'gpu' else 'cpu'

    if lightning_module.config.pruning.save_model:
        lightning_module.teacher_model.to('cpu')
        torch.save(lightning_module.teacher_model, 'models/origin_model.pth')

    model = deepcopy(lightning_module.teacher_model)
    model.to(device)

    input_example = torch.randint(
        high=model.embeddings.word_embeddings.num_embeddings,
        size=(datamodule.cfg.batch_size, lightning_module.max_length),
        device=device,
    )

    ops, params = tp.utils.count_ops_and_params(model, input_example)
    logging.info(f'Model complexity: {ops / 1e6} MMAC, {params / 1e6} M params')

    ignored_layers = []
    for name, module in model.named_modules():
        if name in ('pooler', 'embeddings'):
            ignored_layers.append(module)

    taylor_criteria = tp.importance.GroupTaylorImportance()
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=input_example,
        importance=taylor_criteria,
        pruning_ratio=lightning_module.config.pruning.pruning_ratio,
        global_pruning=lightning_module.config.pruning.global_pruning,
        iterative_steps=lightning_module.config.pruning.iterative_steps,
        ignored_layers=ignored_layers,
    )

    datamodule.setup('fit')
    train_dataloader = datamodule.train_dataloader()

    mse_loss = torch.nn.MSELoss()
    model.zero_grad()
    model.train()

    for idx, (ru, en) in enumerate(tqdm(train_dataloader)):
        inp_ru = lightning_module.tokenize(ru, device)
        inp_en = lightning_module.tokenize(en, device)

        ru_out = model(**inp_ru)
        ru_out = torch.nn.functional.normalize(ru_out.pooler_output)

        en_out = model(**inp_en)
        en_out = torch.nn.functional.normalize(en_out.pooler_output)

        loss = mse_loss(ru_out, en_out)
        loss.backward()

    for step in range(lightning_module.config.pruning.iterative_steps):
        for i, group in enumerate(pruner.step(interactive=True)):
            group.prune()

    for module in model.modules():
        if isinstance(module, BertSelfAttention):
            module.attention_head_size = module.attention_head_size // 4
            module.all_head_size = module.all_head_size // 4

    lightning_module.student_model = model

    ops, params = tp.utils.count_ops_and_params(model, input_example)
    logging.info(f'Model complexity (After taylor pruning): {ops / 1e6} MMAC, {params / 1e6} M params')

    if lightning_module.config.pruning.save_model:
        model.to('cpu')
        torch.save(model, 'models/prune_model.pth')
