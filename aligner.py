import pytorch_lightning as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from model import AlignerModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path='', config_name='aligner')
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get('exp_manager', None))
    model = AlignerModel(cfg=cfg.model, trainer=trainer)
    trainer.callbacks.extend([pl.callbacks.LearningRateMonitor(), LogEpochTimeCallback()])  # noqa
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
