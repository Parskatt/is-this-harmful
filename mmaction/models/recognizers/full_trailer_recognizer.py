import torch
from ..registry import RECOGNIZERS
from .base import BaseRecognizer

@RECOGNIZERS.register_module()
class FullTrailerModel(BaseRecognizer):
    """Audio recognizer model framework."""
    def __init__(self, *args, log_space_input=True, **kwargs):
        super().__init__(*args,**kwargs)
        self.log_space_input=log_space_input

    def forward(self, preds, label=None, return_loss=True):
        """Define the computation performed at every call."""
        if self.log_space_input:
            x = preds.log().sum(dim=-2).permute(0,2,1) # To fit with pytorch standard ordering
            x = (x-x.mean())#/x.std()
        else:
            x = preds.prod(dim=-2)
            x /= x.sum(dim=-1,keepdim=True)

        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(x, label)

        return self.forward_test(x)

    def forward_train(self, x, labels):
        """Defines the computation performed at every call when training."""

        cls_score = self.cls_head(x)
        loss = self.cls_head.loss(cls_score, labels)
        return loss

    def forward_test(self, x):
        """Defines the computation performed at every call when evaluation and
        testing."""
        cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score)
        return cls_score.cpu().numpy()

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        preds = data_batch['preds']

        label = data_batch['label']

        losses = self(preds, label)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        preds = data_batch['preds']

        label = data_batch['label']

        losses = self(preds, label)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def forward_gradcam(self, imgs):
        raise NotImplementedError