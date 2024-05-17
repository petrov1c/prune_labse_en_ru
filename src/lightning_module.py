import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoTokenizer, AutoModel
from itertools import chain
from typing import Optional
from copy import deepcopy

from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.utils import load_object
from src.evalution import evaluate

mse_loss = nn.MSELoss()
kl_loss = nn.KLDivLoss()


class PruneModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        self.teacher_model = AutoModel.from_pretrained(self.config.model.name)
        self.student_model: Optional[nn.Module] = None
        self.max_length = self.teacher_model.embeddings.position_embeddings.num_embeddings

        self._losses = get_losses(self.config.losses)
        metrics = get_metrics()
        self._valid_metrics = metrics.clone(prefix='val_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.save_hyperparameters()

    def tokenize(self, input_text, device):
        return {
            k: v.to(device) for k, v in
            self.tokenizer(input_text, return_tensors='pt', padding=True, max_length=self.max_length,
                           truncation=True).items()
        }

    def configure_optimizers(self):
        optimizer = load_object(self.config.optimizer)(
            self.student_model.parameters(),
            **self.config.optimizer_kwargs,
        )
        scheduler = load_object(self.config.scheduler)(optimizer, **self.config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        self.teacher_model.eval()
        self.student_model.train()

        with torch.no_grad():
            # Возможно эта строчка лишняя
            inputs = self.tokenize([text for text in chain(*batch)], self.device)
            teacher_output = self.teacher_model(**inputs, output_attentions=True, output_hidden_states=True)

        inputs = self.tokenize([text for text in chain(*batch)], self.device)
        student_output = self.student_model(**inputs, output_attentions=True, output_hidden_states=True)

        loss = self._calculate_loss(student_output, teacher_output, 'train_', len(batch[0]))
        return loss

    def validation_step(self, batch, batch_idx):
        input_text = [text for text in chain(*batch)]
        inputs = self.tokenize(input_text, self.device)

        teacher_output = self.teacher_model(**inputs, output_attentions=True, output_hidden_states=True)
        teacher_out = torch.nn.functional.normalize(teacher_output.pooler_output)

        student_output = self.student_model(**inputs, output_attentions=True, output_hidden_states=True)
        student_out = torch.nn.functional.normalize(student_output.pooler_output)

        self._calculate_loss(student_output, teacher_output, 'val_', len(batch[0]))
        self._valid_metrics(student_out, teacher_out)

    def test_step(self, batch, batch_idx):
        input_text = [text for text in chain(*batch)]
        inputs = self.tokenize(input_text, self.device)

        teacher_output = self.teacher_model(**inputs, output_attentions=True, output_hidden_states=True)
        teacher_output = torch.nn.functional.normalize(teacher_output.pooler_output)

        student_output = self.student_model(**inputs, output_attentions=True, output_hidden_states=True)
        student_output = torch.nn.functional.normalize(student_output.pooler_output)

        self._test_metrics(student_output, teacher_output)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        metrics = self._valid_metrics.compute()
        if not self.trainer.sanity_checking:
            self._benchmark_metrics_compute(metrics, 'val_')

    def on_test_epoch_end(self) -> None:
        metrics = self._test_metrics.compute()
        self._benchmark_metrics_compute(metrics, 'test_')

        # Сохраню модель костыль
        model = deepcopy(self.student_model)
        model.zero_grad()
        model.to('cpu')
        model.eval()
        torch.save(model, 'models/tune_model.pth')

    def _benchmark_metrics_compute(self, metrics, prefix: str = ''):
        scores = evaluate(self.student_model, self.tokenizer)
        metrics[prefix + 'mean_s'] = torch.tensor(scores['mean_s'].astype('float32'), device=self.device)
        self.log_dict(metrics, on_epoch=True)

    def _calculate_loss(
        self,
        student_outputs,
        teacher_outputs,
        prefix: str,
        batch_size: int,
    ) -> torch.Tensor:

        loss_weight = 0.5
        last_layer_loss_weight = 0.5
        intermediate_attn_layers_weights = (0, 0, 0, 1)
        student_teacher_attention_mapping = {0: 1, 1: 3, 2: 5, 3: 7}
        intermediate_feat_layers_weights = (0, 0, 0, 1)

        ru_out = torch.nn.functional.normalize(student_outputs.pooler_output[:batch_size])
        en_out = torch.nn.functional.normalize(student_outputs.pooler_output[batch_size:])

        loss = mse_loss(ru_out, en_out)
        self.log(f'{prefix}mse_loss', loss.item(), batch_size=batch_size)

        last_layer_loss = calc_last_layer_loss(
            student_outputs.last_hidden_state,
            teacher_outputs.last_hidden_state,
        )
        self.log(f'{prefix}last_layer_loss', last_layer_loss.item(), batch_size=batch_size)

        student_attentions, teacher_attentions = student_outputs.attentions, teacher_outputs.attentions
        student_hidden_states, teacher_hidden_states = student_outputs.hidden_states, teacher_outputs.hidden_states

        intermediate_layer_att_loss = calc_intermediate_layers_attn_loss(
            student_attentions,
            teacher_attentions,
            intermediate_attn_layers_weights,
            student_teacher_attention_mapping,
        )

        intermediate_layer_feat_loss = calc_intermediate_layers_feat_loss(
            student_hidden_states,
            teacher_hidden_states,
            intermediate_feat_layers_weights,
        )

        total_loss = loss * loss_weight + last_layer_loss * last_layer_loss_weight
        if intermediate_layer_att_loss is not None:
            total_loss += intermediate_layer_att_loss

        if intermediate_layer_feat_loss is not None:
            total_loss += intermediate_layer_feat_loss

#                log_student_sm = F.log_softmax(student_outputs, dim=-1)
#                teacher_sm = F.softmax(teacher_outputs, dim=-1)
#                loss = cur_loss.loss(input=log_student_sm, target=teacher_sm)

        self.log(f'{prefix}total_loss', total_loss.item(), batch_size=batch_size)
        return total_loss

    def _calculate_loss1(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        prefix: str,
        batch_size: int
    ) -> torch.Tensor:
        total_loss = 0
        for cur_loss in self._losses:
            # Костыль (временный)
            if cur_loss.name == 'kl':
                log_student_sm = F.log_softmax(student_outputs, dim=-1)
                teacher_sm = F.softmax(teacher_outputs, dim=-1)
                loss = cur_loss.loss(input=log_student_sm, target=teacher_sm)
            else:
                loss = cur_loss.loss(student_outputs, teacher_outputs)
            total_loss += cur_loss.weight * loss
            self.log(f'{prefix}{cur_loss.name}_loss', loss.item(), batch_size=batch_size)
        self.log(f'{prefix}total_loss', total_loss.item(), batch_size=batch_size)
        return total_loss

    # Костыль (временный)
    #def calc_last_layer_loss(student_logits, teacher_logits, temperature, weight):
    #    inputs = F.log_softmax(student_logits/temperature, dim=-1)
    #    target = F.softmax(teacher_logits/temperature, dim=-1)
    #    loss = nn.KLDivLoss(input=inputs, target=target) * temperature ** 2

    #    return loss

    # the whole batch, all tokens after the [cls], the whole dimension
    # kd_loss = torch.nn.KLDivLoss(reduction='batchmean')(
    #    F.log_softmax(outputs.prediction_logits[:, 1:, :] / temperature, dim=1),
    #    F.softmax(big_out.prediction_logits[:, 1:, vocab_mapping] / temperature, dim=1)
    #) / outputs.prediction_logits.shape[-1]


def calc_last_layer_loss(student_logits, teacher_logits):
    return mse_loss(student_logits, teacher_logits)

def calc_intermediate_layers_attn_loss(student_logits, teacher_logits, weights, student_teacher_attention_mapping):
    return None

def calc_intermediate_layers_feat_loss(student_feat, teacher_feat, weights):
    return None