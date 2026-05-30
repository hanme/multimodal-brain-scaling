import argparse
from pathlib import Path

import numpy as np

import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
import  lightning as l

import h5py

from mbs.training.modeling import create_lightning_model
from mbs.training.utils import load_yaml, create_argparser
from mbs.training.data import create_datamodule

# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = True



def main(args: argparse.Namespace):
    config = vars(args)
    
    l.seed_everything(args.seed)
    
    # https://github.com/Lightning-AI/pytorch-lightning/issues/790#issuecomment-582696775
    if args.cutmix != 0.0 or args.mixup != 0.0:
        raise NotImplementedError("Cutmix and Mixup not supported yet")
    
    # https://lightning.ai/forums/t/adopting-exponential-moving-average-ema-for-pl-pipeline/488/4
    if args.model_ema:
        raise NotImplementedError("Model Exponential Moving Average not supported yet")
    
    if args.target_batch_size is not None or args.accumulation_steps > 1:
        raise NotImplementedError("Gradient accumulation and target batch size not supported yet")
    
    with h5py.File(Path(args.data_path_neural) / args.data_neural_filename, 'r') as f:
        subjects = f.attrs['subjects']
        
    if 'randsubj' in args.data_neural_subjects:
        print("Using random subject selection for neural data")
        num_randsubj = int(args.data_neural_subjects.replace('randsubj_', ''))
        assert num_randsubj <= len(subjects), "Number of random subjects exceeds available subjects"
        rng = np.random.default_rng(seed=args.seed)
        subjects = list(rng.choice(subjects, size=num_randsubj, replace=False))
        print("Selected subjects:", subjects)
    elif args.data_neural_subjects != 'all':
        selected_subjects = args.data_neural_subjects.split(',')
        for subj in selected_subjects:
            assert subj in subjects, f"Subject {subj} not found in neural data file"
        subjects = selected_subjects
    else:
        subjects = list(subjects)
    config['data_neural_subjects'] = subjects
    config['subjects'] = subjects
    
    rois = args.data_neural_regions.split(',')
    config['rois'] = rois
    
    if args.things_image_db_path in ['None', 'none', 'null', '']:
        config['things_image_db_path'] = None
    if args.nsd_image_h5_path in ['None', 'none', 'null', '']:
        config['nsd_image_h5_path'] = None
    
    data_module = create_datamodule(**config)
    data_module.setup()
    dl_train = data_module.train_dataloader()
    iterations_per_epoch = len(iter(dl_train)) // args.ngpus
    if iterations_per_epoch <= 10:
        args.drop_last_train = False
        config['drop_last_train'] = False
        data_module = create_datamodule(**config)
        data_module.setup()
        dl_train = data_module.train_dataloader()
        iterations_per_epoch = len(iter(dl_train)) // args.ngpus
        print("Setting `drop_last_train` to False since iterations_per_epoch is less than 10")
    del dl_train
    

    
    output_dims = {}
    with h5py.File(Path(args.data_path_neural) / args.data_neural_filename, 'r') as f:
        nc_max = f.attrs.get('max_nc', 100.0)
        for subj in subjects:
            for roi in rois:
                dims = torch.tensor(f['train']['neural_data'][subj][roi].shape[1:])
                if len(dims) == 2:
                    noise_ceiling = f['noise_ceilings'][subj][roi][()]
                    noise_ceiling = noise_ceiling.flatten() / nc_max + 1e-6
                    noise_ceiling= np.sqrt(noise_ceiling)
                    
                    noise_ceiling_mask = noise_ceiling > 0.1
                    dims = int(noise_ceiling_mask.sum().item())
                else:
                    dims = int(dims.prod().item())
                output_dims[f"subj_{subj}-roi_{roi}"] = dims
    print("Decoder output dims:", output_dims)
    config['decoder_output_dims'] = output_dims

    # Create the model
    model = create_lightning_model(iterations_per_epoch=iterations_per_epoch, **config)
    if args.checkpoint:
        model.load_from_checkpoint(args.checkpoint)

    last_checkpoint = None
    if args.continue_from_last:
        checkpoints_dir = Path(args.save_dir)
        last_checkpoint = checkpoints_dir / 'last.ckpt'
        if last_checkpoint.exists():
            print("Continuing from last checkpoint")
        else:
            last_checkpoint = None

    if args.compile_mode and args.compile_mode != 'None':
        print("Compiling model with mode:", args.compile_mode)
        model = torch.compile(model, mode=args.compile_mode)

    
    callbacks = []
    
    # Learning rate monitor
    if args.lr_monitor and not args.disable_wandb:
        lr_monitor = LearningRateMonitor(logging_interval='step', log_weight_decay=True)
        callbacks.append(lr_monitor)
    
    # Model checkpoint
    if Path(args.save_dir).exists() and any(Path(args.save_dir).glob("*.ckpt")) and not args.save_overwrite:
        raise ValueError(f"Checkpoint files already exist in {args.save_dir}. Use --save_overwrite to overwrite.")
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        # filename='best_model',
        save_top_k=-1,
        every_n_epochs=args.save_interval,
        save_last=True,
        enable_version_counter=False,
    )
    callbacks.append(checkpoint_callback)
    
    loggers = []
    if not args.disable_wandb:
        wandb_id = args.wandb_id if args.wandb_id is not None else None
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            log_model=False,
            name=args.run_name,
            id=wandb_id,
            config=config,
            resume='allow'
        )
        loggers.append(wandb_logger)
    
    enable_progress_bar = not args.disable_progress_bar
    trainer = Trainer(
        logger=loggers,
        precision=args.precision,
        devices=args.ngpus,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        num_sanity_val_steps=args.num_sanity_val_steps,
        log_every_n_steps=args.log_interval,
        check_val_every_n_epoch=args.eval_interval,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=args.clip_grad,
        strategy='ddp_find_unused_parameters_true',
        # reload_dataloaders_every_n_epochs=1,
        # gradient_clip_algorithm='norm'
    )
    

    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=last_checkpoint
    )


def cli():
    config_parser = argparse.ArgumentParser(description='Experiment Configuration', add_help=False)
    config_parser.add_argument('-ce', '--config-encoder', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments for the encoder')
    config_parser.add_argument('-cd', '--config-decoder', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments for the decoder')
    config_parser.add_argument('-cl', '--config-lora', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments for the LoRA')

    # Inherit config flags so they appear in the main parser's --help output.
    parser = create_argparser(parents=[config_parser])
    args_config, remaining = config_parser.parse_known_args()

    # Load config file
    if args_config.config_encoder:
        cfg = load_yaml(args_config.config_encoder)
        parser.set_defaults(**cfg)

    if args_config.config_decoder:
        cfg = load_yaml(args_config.config_decoder)
        parser.set_defaults(**cfg)

    if args_config.config_lora:
        cfg = load_yaml(args_config.config_lora)
        parser.set_defaults(config_lora=cfg)

    args = parser.parse_args(remaining)

    print(args)

    main(args)


if __name__ == '__main__':
    cli()