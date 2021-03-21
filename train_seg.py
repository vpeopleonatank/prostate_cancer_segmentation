import os
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.technical_utils import load_obj, flatten_omegaconf, convert_to_jit
from src.utils.utils import set_seed, save_useful_info

warnings.filterwarnings('ignore')


def run(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model

    Args:
        new_dir:
        cfg: hydra config

    """
    set_seed(cfg.training.seed)
    run_name = os.path.basename(os.getcwd())
    hparams = flatten_omegaconf(cfg)

    cfg.callbacks.model_checkpoint.params.dirpath = os.getcwd() + cfg.callbacks.model_checkpoint.params.dirpath
    callbacks = []
    for callback in cfg.callbacks.other_callbacks:
        if callback.params:
            callback_instance = load_obj(callback.class_name)(**callback.params)
        else:
            callback_instance = load_obj(callback.class_name)()
        callbacks.append(callback_instance)

    loggers = []
    if cfg.logging.log:
        for logger in cfg.logging.loggers:
            if 'experiment_name' in logger.params.keys():
                logger.params['experiment_name'] = run_name
            loggers.append(load_obj(logger.class_name)(**logger.params))

    callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping.params))

    trainer = pl.Trainer(
        logger=loggers,
        # early_stop_callback=EarlyStopping(**cfg.callbacks.early_stopping.params),
        checkpoint_callback=ModelCheckpoint(**cfg.callbacks.model_checkpoint.params),
        callbacks=callbacks,
        **cfg.trainer,
    )
    if cfg.general.resume_from_path is False:
        model = load_obj(cfg.training.lightning_module_name)(hparams=hparams, cfg=cfg)
    else:
        model = load_obj(cfg.training.lightning_module_name).load_from_checkpoint(
            cfg.general.resume_from_path, hparams=hparams, cfg=cfg
        )

    dm = load_obj(cfg.datamodule.data_module_name)(hparams=hparams, cfg=cfg)
    if cfg.trainer.auto_lr_find:
        trainer.tune(model, datamodule=dm)
    trainer.fit(model, dm)

    if cfg.general.save_pytorch_model and cfg.general.save_best:
        if os.path.exists(trainer.checkpoint_callback.best_model_path):  # type: ignore
            best_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            # extract file name without folder
            save_name = os.path.basename(os.path.normpath(best_path))
            model = model.load_from_checkpoint(best_path, hparams=hparams, cfg=cfg, strict=False)
            if not os.path.exists('saved_models'):  # type: ignore
                os.makedirs('saved_models')
            model_name = f'saved_models/best_{save_name}'.replace('.ckpt', '.pth')
            torch.save(model.model.state_dict(), model_name)
        else:
            os.makedirs('saved_models', exist_ok=True)
            model_name = 'saved_models/last.pth'
            torch.save(model.model.state_dict(), model_name)

    if cfg.general.convert_to_jit and os.path.exists(trainer.checkpoint_callback.best_model_path):  # type: ignore
        convert_to_jit(model, save_name, cfg)


@hydra.main(config_path='conf', config_name='seg_config')
def run_model(cfg: DictConfig) -> None:
    os.makedirs('logs', exist_ok=True)
    print(cfg.pretty())
    if cfg.general.log_code:
        save_useful_info()
    run(cfg)


if __name__ == '__main__':
    run_model()
