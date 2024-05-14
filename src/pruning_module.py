import logging
import torch_pruning as tp
import torch
from transformers.models.bert.modeling_bert import BertSelfAttention

from src.datamodule import SentenceDM
from src.lightning_module import PruneModule, chain

from copy import deepcopy
from tqdm import tqdm


def prune_model(lightning_module: PruneModule, datamodule: SentenceDM):
    device = 'cuda' if lightning_module.config.accelerator == 'gpu' else 'cpu'

    model = deepcopy(lightning_module.teacher_model)
    model.to(device)

    datamodule.setup('fit')
    train_dataloader = datamodule.train_dataloader()

    batch = next(iter(train_dataloader))
    input_example = lightning_module.tokenize([text for text in chain(*batch)], device)

    ops, params = tp.utils.count_ops_and_params(model, input_example)
    logging.info(f'Model complexity: {ops / 1e6} MMAC, {params / 1e6} M params')

    ignored_layers = []
    for name, module in model.named_modules():
        if name in ('pooler', 'embeddings'):
            ignored_layers.append(module)

    num_heads = {}
    for m in model.modules():
        if isinstance(m, BertSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads

    taylor_criteria = tp.importance.GroupTaylorImportance()
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=input_example,
        importance=taylor_criteria,
        pruning_ratio=lightning_module.config.pruning.pruning_ratio,
        global_pruning=lightning_module.config.pruning.global_pruning,
        iterative_steps=lightning_module.config.pruning.iterative_steps,
        num_heads=num_heads,
        prune_head_dims=True,
        prune_num_heads=True,
        head_pruning_ratio=lightning_module.config.pruning.pruning_ratio,
        ignored_layers=ignored_layers,
    )

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
            module.num_attention_heads = pruner.num_heads[module.query]
            module.attention_head_size = module.query.out_features // module.num_attention_heads
            module.all_head_size = module.query.out_features

    lightning_module.student_model = model

    ops, params = tp.utils.count_ops_and_params(model, input_example)
    logging.info(f'Model complexity (After taylor pruning): {ops / 1e6} MMAC, {params / 1e6} M params')

    if lightning_module.config.pruning.save_model:
        model.zero_grad()
        model.to('cpu')
        torch.save(model, 'models/prune_model.pth')
