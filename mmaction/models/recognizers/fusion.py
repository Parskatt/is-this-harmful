import torch
from ..registry import RECOGNIZERS
from .base import BaseRecognizer

@RECOGNIZERS.register_module()
class FusionModel(BaseRecognizer):
    """Audio recognizer model framework."""

    def forward(self, video_pred, audio_pred, label=None, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(video_pred,audio_pred, label)

        return self.forward_test(video_pred, audio_pred)

    def forward_train(self, video_pred, audio_pred, labels):
        """Defines the computation performed at every call when training."""
        x = (torch.cat((video_pred,audio_pred),dim=1).log()+2)/2
        cls_score = self.cls_head(x)
        loss = self.cls_head.loss(cls_score, labels)

        return loss

    def forward_test(self, video_pred, audio_pred):
        """Defines the computation performed at every call when evaluation and
        testing."""
        #print(video_pred,audio_pred)
        x = (torch.cat((video_pred,audio_pred),dim=1).log()+2)/2
        cls_score = self.cls_head(x)
        cls_score = self.average_clip(cls_score)
        #print(cls_score)
        #exit()
        #cls_score = video_pred*audio_pred
        #cls_score /= cls_score.sum(dim=1,keepdim=True)
        return cls_score.cpu().numpy()

    def forward_gradcam(self, audios):
        raise NotImplementedError

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
        video_pred = data_batch['video_pred']
        audio_pred = data_batch['audio_pred']

        label = data_batch['label']

        losses = self(video_pred, audio_pred, label)

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
        video_pred = data_batch['video_pred']
        audio_pred = data_batch['audio_pred']

        label = data_batch['label']

        losses = self(video_pred, audio_pred, label)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
