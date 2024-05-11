import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoTokenizer, AutoModel
from itertools import chain
from typing import Optional

from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.utils import load_object


mse_loss = nn.MSELoss()
kl_loss = nn.KLDivLoss()


class PruneModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.teacher_model = AutoModel.from_pretrained(self.config.model_name)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

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
        """
        Считаем лосс.
        """
        self.teacher_model.eval()
        self.student_model.train()

        inputs = self.tokenize([text for text in chain(*batch)], self.device)
        student_output = self.student_model(**inputs, output_attentions=True, output_hidden_states=True)

        ru_out = torch.nn.functional.normalize(student_output.pooler_output[:len(batch[0])])
        en_out = torch.nn.functional.normalize(student_output.pooler_output[len(batch[0]):])

        loss = mse_loss(ru_out, en_out)

        with torch.no_grad():
            teacher_output = self.teacher_model(**inputs, output_attentions=True, output_hidden_states=True)

#        last_layer_loss = calc_last_layer_loss(
#            student_logits,
#            teacher_output.logits,
#            train_params.temperature,
#            train_params.last_layer_loss_weight,
#        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Считаем лосс и метрики.
        """
        input_text = [text for text in chain(*batch)]
        inputs = self.tokenize(input_text, self.device)

        teacher_out = self.teacher_model(**inputs)
        teacher_out = torch.nn.functional.normalize(teacher_out.pooler_output)

        student_out = self.student_model(**inputs)
        student_out = torch.nn.functional.normalize(student_out.pooler_output)

        self._calculate_loss(student_out, teacher_out, 'val_', len(batch[0]))
        self._valid_metrics(student_out, teacher_out)

    def test_step(self, batch, batch_idx):
        """
        Считаем метрики.
        """
        images, gt_labels = batch
        pr_logits = self(images)
        pr_labels = torch.sigmoid(pr_logits)
        self._test_metrics(pr_labels, gt_labels)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _calculate_loss(
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
    def calc_last_layer_loss(student_logits, teacher_logits, temperature, weight):
        inputs = F.log_softmax(student_logits/temperature, dim=-1)
        target = F.softmax(teacher_logits/temperature, dim=-1)
        loss = nn.KLDivLoss(input=inputs, target=target) * temperature ** 2

        return loss
