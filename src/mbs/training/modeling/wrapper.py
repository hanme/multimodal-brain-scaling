import time
import gc

import torch
import torch.nn as nn

import lightning as l
import torchmetrics
from torchmetrics import MetricCollection

from mbs.training.modeling.loss_funcs import create_loss_func
from mbs.training.modeling.optim import create_optimizer
from mbs.training.modeling.lr_schedulers import create_scheduler

import scipy


class PearsonRPerTarget(torchmetrics.Metric):
    """
    Per-target Pearson correlation across the batch dimension.

    Matches:
      scipy.stats.pearsonr(y_true, y_pred, axis=0)[0]

    Args:
      num_targets: D (size of last dim after flattening to [B, D])
      reduction: "none" -> [D], "mean" -> scalar, "sum" -> scalar
    """

    full_state_update: bool = False

    def __init__(
        self,
        num_targets: int,
        reduction: str = "none",
        eps: float = 1e-16,
        dist_sync_on_step: bool = False,
        dtype=torch.float64,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if num_targets <= 0:
            raise ValueError(f"num_targets must be > 0, got {num_targets}")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be one of ['none','mean','sum'], got {reduction}")

        self.num_targets = int(num_targets)
        self.reduction = reduction
        self.eps = float(eps)
        self.dtype = dtype

        self.add_state("n", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

        z = torch.zeros(self.num_targets, dtype=dtype)
        self.add_state("sum_x", default=z.clone(), dist_reduce_fx="sum")
        self.add_state("sum_y", default=z.clone(), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=z.clone(), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=z.clone(), dist_reduce_fx="sum")
        self.add_state("sum_xy", default=z.clone(), dist_reduce_fx="sum")

    @staticmethod
    def _to_2d(x: torch.Tensor) -> torch.Tensor:
        # Accept [B], [B, D], or [B, ..., D] -> flatten to [B, D]
        if x.ndim == 1:
            return x.unsqueeze(1)
        if x.ndim == 2:
            return x
        return x.reshape(x.shape[0], -1)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = self._to_2d(preds).to(dtype=self.dtype)     # <-- force dtype
        target = self._to_2d(target).to(dtype=self.dtype)   # <-- force dtype

        if preds.shape != target.shape:
            raise ValueError(f"preds and target must have same shape, got {preds.shape} vs {target.shape}")

        if preds.shape[1] != self.num_targets:
            raise ValueError(f"Expected num_targets={self.num_targets}, got {preds.shape[1]}")

        preds = preds.to(dtype=self.dtype) if not torch.is_floating_point(preds) else preds
        target = target.to(dtype=self.dtype) if not torch.is_floating_point(target) else target

        b = preds.shape[0]
        self.n += b
        self.sum_x += preds.sum(dim=0)
        self.sum_y += target.sum(dim=0)
        self.sum_x2 += (preds * preds).sum(dim=0)
        self.sum_y2 += (target * target).sum(dim=0)
        self.sum_xy += (preds * target).sum(dim=0)
        
    @torch.no_grad()
    def reset(self) -> None:
        super().reset()
        self.n.zero_()
        self.sum_x.zero_()
        self.sum_y.zero_()
        self.sum_x2.zero_()
        self.sum_y2.zero_()
        self.sum_xy.zero_()

    def compute(self) -> torch.Tensor:
        # SciPy pearsonr needs >=2 samples; for safety:
        if self.n.item() < 2:
            out = torch.full((self.num_targets,), float("nan"), device=self.sum_x.device, dtype=self.sum_x.dtype)
            return out if self.reduction == "none" else torch.tensor(float("nan"), device=self.sum_x.device)

        n = self.n.to(dtype=self.sum_x.dtype)

        ex = self.sum_x / n
        ey = self.sum_y / n
        ex2 = self.sum_x2 / n
        ey2 = self.sum_y2 / n
        exy = self.sum_xy / n

        cov = exy - ex * ey
        var_x = ex2 - ex * ex
        var_y = ey2 - ey * ey

        denom = torch.sqrt(torch.clamp(var_x, min=0.0)) * torch.sqrt(torch.clamp(var_y, min=0.0))
        r = cov / (denom + self.eps)

        # Constant series -> NaN (closer to SciPy behavior)
        const_mask = denom <= self.eps
        if const_mask.any():
            r = r.clone()
            r[const_mask] = float("nan")
        
        
        # print("sum_x", self.sum_x)
        # print("sum_y", self.sum_y)
        # print("sum_x2", self.sum_x2)
        # print("sum_y2", self.sum_y2)
        # print("sum_xy", self.sum_xy)
        # print("cov", cov)
        # print("var_x", var_x)
        # print("var_y", var_y)
        # print("denom", denom)
        # print("PearsonRPerTarget compute:", r)

        if self.reduction == "none":
            return r
        if self.reduction == "sum":
            return torch.nansum(r)
        return torch.nanmean(r)


class LightningWrapper(l.LightningModule):
    def __init__(
        self, 
        model:nn.Module, 
        **kwargs
        ):
        
        super().__init__()
        self.model = model
        self.model_identifier = kwargs.get('model_identifier', kwargs.get('run_name'))
        # self.target_brain_regions = kwargs.get('target_brain_regions', [])
        self.target_brain_regions = kwargs.get('rois', [])
        print("Target brain regions:", self.target_brain_regions)
        # self.target_brain_regions = [] if self.target_brain_regions in ["", "None", None] else self.target_brain_regions.split(",")
        self.subjects = kwargs['subjects']
        if kwargs.get('decoder_type') == 'concat':
            self.target_brain_regions = ["all_concat"]

        self._configure_loss_weights(**kwargs)
        self._configure_loss_functions(**kwargs)
        self._configure_metrics(**kwargs)
        
        self.save_hyperparameters(kwargs)
        
    def _configure_loss_functions(self, **kwargs):
        loss_func_image = kwargs.pop('loss_func_image', 'cross_entropy')
        # loss_func_neural_behavior = kwargs.pop('loss_func_neural_behavior', 'cross_entropy')
        loss_func_neural_response = kwargs.pop('loss_func_neural_response', {f"subj_{subj}-roi_{region}":'mse' for subj in self.subjects for region in self.target_brain_regions})
        print("Loss functions for neural response:", loss_func_neural_response)
        
        self.loss_functions = {
            "image": create_loss_func(loss_func=loss_func_image, **kwargs),
            # "neural_behavior": create_loss_func(loss_func=loss_func_neural_behavior, **kwargs),
            "neural_response": {
                region:create_loss_func(loss_func=loss_fn, **kwargs) 
                for region, loss_fn in loss_func_neural_response.items()
                }
        }
        
        self.loss_function_names = {
            "image": f"Image_{self.loss_functions['image'].__class__.__name__}",
            # "neural_behavior": f"NeuralBehavior_{self.loss_functions['neural_behavior'].__class__.__name__}",
            "neural_response": {
                region: f"NeuralResponse_{region}_{loss_fn.__class__.__name__}"
                for region, loss_fn in self.loss_functions['neural_response'].items()
            }
        }
        
    def _configure_loss_weights(self, **kwargs):
        self.loss_weights = {
            "image": kwargs.get('lambda_image', 1.0),
            # "neural_behavior": kwargs.get('lambda_neural_behavior', 1.0),
            "neural_response": kwargs.get('lambda_neural_response', 1.0),
        }
        
    def _configure_metrics(self, **kwargs):
        
        neural_metrics_train = {
            f"NeuralPearsonR_{key}": PearsonRPerTarget(num_targets=kwargs['decoder_output_dims'][key] ,reduction='mean')
            for key in self.loss_functions['neural_response'].keys()
        }
        self.train_metrics = MetricCollection({
                "Image_MulticlassAccuracy": torchmetrics.Accuracy(
                    num_classes=kwargs['num_classes'], task='multiclass', average="micro"
                ),
                # "NeuralStimuli_MulticlassAccuracy": torchmetrics.Accuracy(
                #     num_classes=kwargs['num_classes_neural'], task='multiclass', average="micro"
                # ),
                **neural_metrics_train
            },
            prefix="train/"
        )
        
        
        neural_metrics_val = {
            f"NeuralPearsonR_{key}": PearsonRPerTarget(num_targets=kwargs['decoder_output_dims'][key] ,reduction='mean')
            for key in self.loss_functions['neural_response'].keys()
        }
        self.val_metrics = MetricCollection({
                "Image_MulticlassAccuracy": torchmetrics.Accuracy(
                    num_classes=kwargs['num_classes'], task='multiclass', average="micro"
                ),
                # "NeuralStimuli_MulticlassAccuracy": torchmetrics.Accuracy(
                #     num_classes=kwargs['num_classes_neural'], task='multiclass', average="micro"
                # ),
                **neural_metrics_val
             },
            prefix="eval/"
        )
        
    
    def configure_optimizers(self):
        
        optimizer = create_optimizer(self.model, **self.hparams)
        scheduler = create_scheduler(optimizer, **self.hparams)
        lr_scheduler_interval = self.hparams.get('lr_scheduler_interval', 'step')
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": lr_scheduler_interval,
                "frequency": 1,
            }
        }
        
    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx=0, dataloader_idx=0):
        # outputs = self.model(batch['image'], mode='image_only')
        outputs = self.model(batch, mode='separate')
        
        # 
        image_target = batch['image']['target']
        # neural_target = batch['neural']['neural_behavior']
        neural_response = batch['neural']['neural_response']
        
        # Compute the loss
        image_loss = self.loss_functions['image'](outputs['output'], image_target)
        # neural_behavior_loss = self.loss_functions['neural_behavior'](outputs['neural_behavior'], neural_target)
        neural_response_loss = {}
        for subj in self.subjects:
            for region in self.target_brain_regions:
                key = f"subj_{subj}-roi_{region}"
                # print(self.loss_functions['neural_response'].keys(), outputs.keys(), neural_response.keys())
                neural_response_loss[key] = self.loss_functions['neural_response'][key](outputs[f"pearsonr_{key}"], neural_response[subj][region])

        # print(self.loss_weights)
        
        loss = self.loss_weights['image'] * image_loss \
                + self.loss_weights['neural_response'] * sum(neural_response_loss.values())
                # + self.loss_weights['neural_behavior'] * neural_behavior_loss \
                
        # loss = self.loss_weights['image'] * image_loss
                
        # Add l2 regularization for decoders
        # wd_decoder = self.hparams.get('wd_decoder', 0.0)
        # for name, decoder in self.model.decoders.named_children():
        #     for param_name, param in decoder.named_parameters():
        #         if 'weight' in param_name:  # Only apply L2 regularization to weights
        #             loss += wd_decoder * torch.sum(param ** 2) / 2  # L2 regularization term
        
        # Log the metrics
        logs = {
            f"train/{self.loss_function_names['neural_response'][region]}": loss_val
            for region, loss_val in neural_response_loss.items()
        }
        logs.update({
            f"train/{self.loss_function_names['image']}": image_loss,
            # f"train/{self.loss_function_names['neural_behavior']}": neural_behavior_loss,
            "train/TotalLoss": loss
        })
        self.log_dict(logs, on_step=True, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
        self.train_metrics["Image_MulticlassAccuracy"].update(outputs['output'], image_target)
        # self.train_metrics["NeuralStimuli_MulticlassAccuracy"].update(outputs['neural_behavior'], neural_target)
        # self.train_metrics["NeuralResponse_PearsonCorrCoef"].update(outputs['IT'], neural_response)
        
        for subj in self.subjects:
            for region in self.target_brain_regions:
                key = f"subj_{subj}-roi_{region}"
                metric_name = f"NeuralPearsonR_{key}"
                self.train_metrics[metric_name].update(outputs[f"pearsonr_{key}"], neural_response[subj][region])
                # self.log(f"train/{metric_name}", self.train_metrics[metric_name], on_step=False, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, add_dataloader_idx=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        
        
        if set(batch.keys()) == set(['input', 'target']):
            outputs = self.model(batch, mode='image_only')
            image_target = batch['target']
            
            # Compute the loss
            image_loss = self.loss_functions['image'](outputs['output'], image_target)
                        
            # Log the metrics
            self.log_dict({
                f"eval/{self.loss_function_names['image']}": image_loss,
                }, on_step=False, on_epoch=True, add_dataloader_idx=False, sync_dist=True
            )
            metric_name = "Image_MulticlassAccuracy"
            self.val_metrics[metric_name].update(outputs['output'], image_target)
            self.log(f"eval/{metric_name}", self.val_metrics[metric_name], on_step=False, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
            
        elif set(batch.keys()) == set(['neural_stimulus', 'neural_response']):
            outputs = self.model(batch, mode='neural_stimulus_only')
            # neural_target = batch['neural_behavior']
            neural_response = batch['neural_response']
            
            # neural_behavior_loss = self.loss_functions['neural_behavior'](outputs['neural_behavior'], neural_target)
            neural_response_loss = {}
            for subj in self.subjects:
                for region in self.target_brain_regions:
                    key = f"subj_{subj}-roi_{region}"
                    # print(self.loss_functions['neural_response'].keys(), outputs.keys(), neural_response.keys())
                    # print(outputs[f"pearsonr_{key}"].shape,  neural_response[subj][region].shape)
                    neural_response_loss[key] = self.loss_functions['neural_response'][key](outputs[f"pearsonr_{key}"], neural_response[subj][region])
                        
            # Log the metrics
            logs = {
                f"eval/{self.loss_function_names['neural_response'][region]}": loss_val
                for region, loss_val in neural_response_loss.items()
            }
            logs.update({
                # f"eval/{self.loss_function_names['neural_behavior']}": neural_behavior_loss,
            })
            self.log_dict(logs, on_step=False, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
            # metric_name = "NeuralStimuli_MulticlassAccuracy"
            # self.val_metrics[metric_name].update(outputs['neural_behavior'], neural_target)
            # self.log(f"eval/{metric_name}", self.val_metrics[metric_name], on_step=False, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
            
            for subj in self.subjects:
                for region in self.target_brain_regions:
                    key = f"subj_{subj}-roi_{region}"
                    metric_name = f"NeuralPearsonR_{key}"
                    self.val_metrics[metric_name].update(outputs[f"pearsonr_{key}"], neural_response[subj][region])
                    self.log(f"eval/{metric_name}", self.val_metrics[metric_name], on_step=False, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
                    
                    # scipy_pearsonr = scipy.stats.pearsonr(
                    #     outputs[key].detach().cpu().float().numpy(),
                    #     neural_response[subj][region].detach().cpu().float().numpy(),
                    #     axis=0
                    # )[0].mean()
                    # print(f"Scipy PearsonR for {key}: {scipy_pearsonr}")
                    
                    # m = self.val_metrics[metric_name]
                    # print("n before:", int(m.n))
                    # m.update(outputs[key], neural_response[subj][region])
                    # print("n after :", int(m.n))

                    # torch_val = m.compute().item()
                    # scipy_val = scipy.stats.pearsonr(
                    #     outputs[key].detach().cpu().float().numpy(),
                    #     neural_response[subj][region].detach().cpu().float().numpy(),
                    #     axis=0
                    # )[0].mean()

                    # print("torchmetric:", torch_val)
                    # print("scipy     :", scipy_val)
            
            # metric_name = "NeuralResponse_PearsonCorrCoef"
            # self.val_metrics[metric_name].update(outputs['IT'], neural_response)
            # self.log(f"eval/{metric_name}", self.val_metrics[metric_name], on_step=False, on_epoch=True, add_dataloader_idx=False)
        else:
            raise ValueError("Invalid batch keys", list(batch.keys()))
        
    # def on_train_start(self):
    #     self.run_benchmarks()
    
    def on_save_checkpoint(self, checkpoint):
        # Save only the encoder backbone weights to reduce checkpoint size
        # Decoders take a lot of space in the checkpoint
        checkpoint['state_dict'] = self.model.encoder.backbone.state_dict()
        del checkpoint['optimizer_states']
        del checkpoint['lr_schedulers']
        
    def on_validation_epoch_end(self):
        gc.collect()
        torch.cuda.empty_cache()
        if self.trainer.is_global_zero:
            pass
            # self.run_benchmarks()
            
    def on_validation_epoch_start(self):
        self.val_metrics.reset()