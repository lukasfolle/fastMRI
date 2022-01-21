from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from torch.nn.parallel.distributed import DistributedDataParallel


class DDPwGradientCheckpointing(DDPPlugin):
    def _setup_model(self, model):
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        ddp = DistributedDataParallel(module=model, device_ids=self.determine_ddp_device_ids(), **self._ddp_kwargs)
        ddp._set_static_graph()
        return ddp
