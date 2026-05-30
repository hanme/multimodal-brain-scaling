import argparse

def str2bool(v):
    """
    Converts a string representation of a boolean value to its corresponding boolean value.
    
    Args:
        v (str): The string representation of the boolean value.
        
    Returns:
        bool: The corresponding boolean value.
    """
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_argparser(parents=None):
    """Get parser for the training script."""
    parser = argparse.ArgumentParser(description="Training CV models", parents=list(parents or []))
    ################## Model ##################
    parser.add_argument(
        "--backbone-source",
        dest="backbone_source",
        choices=["timm", "spvvs", "hf"],
        default='spvvs',
        help="Model library for loading the backbone",
    )
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default=None,
        help="Model architecture",
    )
    parser.add_argument(
        "--num-classes",
        default=1000,
        type=int,
        metavar="N",
        help="number of classes in image dataset (default: 1000)",
    )
    parser.add_argument(
        "--num-classes-neural",
        default=8,
        type=int,
        metavar="N",
        help="number of classes in neural dataset (default: 8)",
    )
    parser.add_argument(
        "--pretrained-model-id",
        dest="pretrained_model_id",
        default=None,
        type=str,
        help="Pretrained model id",
    )
    parser.add_argument(
        "--training-mode",
        dest="training_mode",
        choices=["finetune", "scratch"],
        default="finetune",
        type=str,
        help="Training mode (default: finetune)",
    )
    parser.add_argument(
        "--feat-layers-label",
        dest="feat_layers_label",
        default="spvvs",
        type=str,
        help="The name of the list of feature layers to use",
    )
    
    
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--continue-from-last",
        dest="continue_from_last",
        action="store_true",
        help="continue training from last checkpoint",
    )
    parser.add_argument(
        "--use-timm",
        dest="use_timm",
        action="store_true",
        help="use a timm model",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "--linear-probes-dir",
        dest="linear_probes_dir",
        default=None,
        type=str,
        help="Directory to load linear probes from (default: None)",
    )
    parser.add_argument(
        "--compile-mode",
        dest="compile_mode",
        default="None",
        type=str,  
        choices=["None", "default", "" "max-autotune", "reduce-overhead"],
        help="compile mode (default: None)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--device", 
        default="gpu", 
        type=str, 
        help="device to use for training / testing"
    )
    parser.add_argument(
        "--ngpus", 
        default=1, 
        type=int, 
        help="number of gpus to use for training / testing. Used for total batch calculation"
    )
    parser.add_argument(
        "--layer-config",
        dest="layer_config",
        type=str,
        default=None,
        help="Layer configuration for ResNetFlex model (default: None)",
    )
    parser.add_argument(
        '--model-ema', 
        dest='model_ema',
        type=str2bool, 
        default=False,
        help='enable model exponential moving average (default: False)'
        )
    parser.add_argument(
        '--model-ema-decay', 
        dest='model_ema_decay',
        type=float, 
        default=0.9999, 
        help='decay factor for model_ema (default: 0.9999)'
        )
    parser.add_argument(
        '--model-ema-eval',
        dest='model_ema_eval',
        type=str2bool,
        default=False,
        help='use model_ema for evaluation (default: False)'
    )
    parser.add_argument(
        '--load-model-ema',
        dest='load_model_ema',
        type=str2bool,
        default=False,
        help='Load model ema weight to model.ema (default: False)'
    )
    parser.add_argument(
        '--precision',
        dest='precision',
        type=str,
        default='bf16-mixed',
        choices=['16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true'],
        help='Mixed precision training (default: bf16-mixed)'
    )
    parser.add_argument(
        '--decoder-type',
        dest='decoder_type',
        type=str,
        default='region_specific',
        choices=['whole_brain', 'region_specific'],
        help='Decoder type (default: whole_brain)'
    )
    parser.add_argument(
        '--layer-commitments',
        dest='layer_commitments',
        type=str,
        default=None,
        help='Path to layer commitments json file'
    )
    parser.add_argument(
        '--layer-commitment-dataset',
        dest='layer_commitment_dataset',
        type=str,
        default=None,
        help='Dataset name for layer commitments'
    )
    parser.add_argument(
        '--overwrite-proj-dim',
        dest='overwrite_proj_dim',
        type=int,
        default=None,
        help='Provide an integer value for the projection dimension (default: None)'
    )
    ################## Dataset ##################
    parser.add_argument(
        "--data-path-image", 
        dest="data_path_image",
        metavar="DIR", 
        help="Path to image data directory"
    )
    parser.add_argument(
        "--dataset",
        default="imagenet",
        type=str,
        choices=["imagenet", "ecoset", "webvision", "imagenet21k", "imagenet21kP", "webvisionP", "places365", "inaturalist", "infimnist"],
        metavar="DATASET",
        help="dataset to use (default: imagenet)",
    )
    parser.add_argument(
        "--data-path-neural", 
        dest="data_path_neural",
        metavar="DIR", 
        help="Path to neural data directory"
    )
    parser.add_argument(
        "--data-neural-filename", 
        dest="data_neural_filename",
        metavar="FILE",
        help="Filename for neural data (default: None)",
    )
    parser.add_argument(
        "--things-image-db-path", 
        dest="things_image_db_path",
        default=None,
        metavar="DIR",
        help="Directory for THINGS image database (default: None)",
    )
    parser.add_argument(
        "--nsd-image-h5-path", 
        dest="nsd_image_h5_path",
        default=None,
        metavar="PATH",
        help="Path for NSD image h5 file (default: None)",
    )
    parser.add_argument(
        "--data-neural-regions", 
        dest="data_neural_regions",
        default="V1,V2,V4,IT",
        help="Comma separated list of regions to use for neural data",
    )
    
    parser.add_argument(
        "--data-neural-subjects", 
        dest="data_neural_subjects",
        default="all",
        help="Comma separated list of subjects to use for neural data. If 'all', use all subjects available in the data file (default: all)",
    )
    
    parser.add_argument(
        "--batch-size-image",
        dest="batch_size_image",
        default=32,
        type=int,
        metavar="N",
        help="mini-batch size for image data (default: 32)",
    )
    parser.add_argument(
        "--batch-size-neural",
        dest="batch_size_neural",
        default=32,
        type=int,
        metavar="N",
        help="mini-batch size for neural data (default: 32)",
    )
    parser.add_argument(
        "--target-batch-size",
        dest="target_batch_size",
        default=None,
        type=int,
        metavar="N",
        help="Total batch size for training (default: 64). Based on ngpus, this will be used to calculate" \
                + " the accumulation steps and batch_size will be set as per gpu mini batch size",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--pin-memory", 
        dest="pin_memory",
        default=True, 
        type=bool, 
        help="pin_memory for dataloader"
    )
    parser.add_argument(
        "--repeated-aug",
        dest="repeated_aug",
        type=str2bool,
        default=False,
        help="repeated augmentation sampler (default: False)"
    )
    parser.add_argument(
        "--drop-last-train",
        dest="drop_last_train",
        type=str2bool,
        default=True,
        help="Drop last batch in training dataloader. Useful if training dataset is subsampled (default: None)"
    )
    parser.add_argument(
        "--combined-loader-mode-train",
        dest="combined_loader_mode_train",
        type=str,
        default='min_size',
        choices=['min_size', 'max_size_cycle', 'max_size', 'sequential'],
        help='Combined loader mode for training (default: min_size)'
    )
    parser.add_argument(
        "--combined-loader-mode-val",
        dest="combined_loader_mode_val",
        type=str,
        default='sequential',
        choices=['min_size', 'max_size_cycle', 'max_size', 'sequential'],
        help='Combined loader mode for validation (default: sequential)'
    )
    parser.add_argument(
        "--neural-data-pct",
        dest="neural_data_pct",
        default=1.0,
        type=float,
        metavar="N",
        help="percentage of neural data to use for training (default: 1.0)",
    )   
    parser.add_argument(
        "--neural-data-random-shuffle",
        dest="neural_data_random_shuffle",
        default=False,
        type=str2bool,
        metavar="N",
        help="Whether to randomly shuffle neural data to break image-neural correspondence (default: False)",
    )   
    
    ################## Training / Optimizer ##################
    parser.add_argument(
        "--loss-fn",
        dest="loss_fn",
        default="cross_entropy",
        type=str,
        choices=["cross_entropy", "soft_target_cross_entropy", "label_smoothing_cross_entropy", "multiclass_bce", "ntxent_loss", "dino_loss"],
        help="Loss function",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="sgd",
        help="optimizer to use (default: sgd)",
        # choices=["sgd", "adam", "adamw"]
    )
    parser.add_argument(
        "--lr-encoder",
        dest="lr_encoder",
        default=1e-4,
        type=float,
        help="initial learning rate for encoder (default: 1e-4)",
    )
    parser.add_argument(
        "--lr-decoder",
        dest="lr_decoder",
        default=1e-4,
        type=float,
        help="initial learning rate for decoders (default: 1e-4)",
    )
    parser.add_argument(
        "--frozen-decoders",
        dest="frozen_decoders",
        default=True,
        type=str2bool,
        help="whether to freeze decoders during training (default: True)",
    )
    parser.add_argument(
        "--lr-scheduler",
        dest="lr_scheduler",
        default="steplr",
        help="learning rate scheduler (default: steplr)",
        choices=["steplr", "cosineannealinglrwarmup", "steplr", "steplrwarmup"],
    )
    parser.add_argument(
        "--lr-step-size",
        dest="lr_step_size",
        default=30,
        type=int,
        metavar="N",
        help="number of epochs to decay learning rate by lr_gamma; ignored if lr_scheduler is cosineannealinglr",
    )
    parser.add_argument(
        "--lr-gamma",
        dest="lr_gamma",
        default=0.1,
        type=float,
        metavar="LR",
        help="multiplicative factor of learning rate decay; ignored if lr_scheduler is cosineannealinglr",
    )
    parser.add_argument(
        "--lr-warmup-duration",
        dest="lr_warmup_duration",
        default=5,
        type=int,
        metavar="N",
        help="Duration of learning rate warmup (default: 5ep)",
    )
    parser.add_argument(
        "--lr-scheduler-interval",
        dest="lr_scheduler_interval",
        default="step",
        type=str,
        choices=["step", "epoch"],
        help="Interval for updating the learning rate (default: step)",
    )
    
    parser.add_argument(
        "--min-lr",
        dest="min_lr",
        default=1e-5,
        type=float,
        metavar="LR",
        help="minimum learning rate",
    )
    parser.add_argument(
        "--plateaulr-patience",
        dest="plateaulr_patience",
        default=3,
        type=int,
        metavar="N",
        help="patience for plateau lr scheduler (default: 3)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum",
    )
    parser.add_argument(
        "--wd-encoder",
        dest="wd_encoder",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay for encoder (default: 1e-4)",
    )
    parser.add_argument(
        "--wd-decoder",
        dest="wd_decoder",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay for decoders (default: 1e-4)",
    )

    parser.add_argument(
        '--opt-eps',
        dest='opt_eps',
        default=None, 
        type=float, 
        metavar='EPSILON',
        help='Optimizer Epsilon (default: None, use opt default)'
    )
    parser.add_argument(
        '--opt-betas',
        dest='opt_betas',
        default=None, 
        type=float, 
        nargs='+', 
        metavar='BETA',
        help='Optimizer Betas (default: None, use opt default)'
    )
    parser.add_argument(
        '--clip-grad', 
        dest='clip_grad',
        type=float, 
        default=None, 
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)'
    )
    parser.add_argument(
        '--label-smoothing',
        dest='label_smoothing',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Label smoothing (default: 0.0)'
    )

    
    parser.add_argument(
        "--eval-interval",
        dest="eval_interval",
        default=1,
        type=int,
        metavar="N",
        help="evaluate every N epochs (default: 1ep)",
    )
    parser.add_argument(
        "--num-sanity-val-steps",
        dest="num_sanity_val_steps",
        default=2,
        type=int,
        metavar="N",
        help="number of sanity check validation steps to run before training (default: 2)",
    )
    
    parser.add_argument(
        "--max-epochs",
        dest="max_epochs",
        default=100,
        type=int,
        metavar="N",
        help="maximum duration for training (default: 100 epochs)",
    )
    parser.add_argument(
        '--accumulation-steps',
        dest='accumulation_steps', 
        default=1,
        type=int,
        help='gradient accumulation steps'
        )

    
    ################## Data Augmentation ##################
    parser.add_argument(
        '--transform-lib',
        dest='transform_lib',
        type=str,
        default='albumentations',
        choices=['albumentations', 'pytorch', 'timm', 'lightly'],
        help='Library to use for data augmentation (default: albumentations)',
    )
    parser.add_argument(
        "--train-crop-size",
        dest="train_crop_size",
        default=224,
        type=int,
        metavar="N",
        help="crop size for training (default: 224)",
    )
    parser.add_argument(
        "--val-resize-size",
        dest="val_resize_size",
        default=256,
        type=int,
        metavar="N",
        help="resize size for validation (default: 256)",
    )
    parser.add_argument(
        "--val-crop-size",
        dest="val_crop_size",
        default=224,
        type=int,
        metavar="N",
        help="crop size for validation (default: 224)",
    )
    parser.add_argument(
        '--interpolation',
        default='bilinear',
        type=str,
        metavar="N",
        help="interpolation mode for resizing (default: bilinear)",
    )
    parser.add_argument(
        '--albumentations-aug-set',
        dest='albumentations_aug_set',
        type=str,
        default="default",
        choices=['default', 'heavy', 'ThreeAugment'],
        help='Albumentations augmentation set (default: default)',
    )
    parser.add_argument(
        '--lightly-aug-set',
        dest='lightly_aug_set',
        type=str,
        default=None,
        choices=['dino', 'simclr'],
        help='Lightly augmentation set for SSL training (default: None)',
    )
    parser.add_argument(
        '--pytorch-aug-set',
        dest='pytorch_aug_set',
        type=str,
        default=None,
        choices=['default', 'ThreeAugment', 'kNN'],
        help='Pytorch augmentation set(default: None)',
    )
    parser.add_argument(
        '--color-jitter',
        dest='color_jitter', 
        type=float, 
        default=0.0, 
        metavar='PCT',
        help='Color jitter factor (default: 0.0)'
    )
    parser.add_argument(
        '--aa', 
        type=str, 
        # default='rand-m9-mstd0.5-inc1',
        default=None,
        metavar='NAME',
        help='Use AutoAugment policy. "rand-m9-mstd0.5-inc1". " + "(default: None)'
    )
    parser.add_argument(
        '--re-prob', 
        dest='re_prob',
        type=float, 
        default=0.0, 
        metavar='PCT',
        help='Random erase probability (default: 0.0)'
    )
    parser.add_argument(
        '--re-mode',
        dest='re_mode',
        type=str, 
        default='pixel',
        help='Random erase mode (default: "pixel")'
    )
    parser.add_argument(
        '--re-count',
        dest='re_count',
        type=int, 
        default=1,
        help='Random erase count (default: 1)'
    )
    
    ################## Mixup/Cutmix ##################
    parser.add_argument(
        '--mixup', 
        type=float, 
        default=0.0,
        help='MixUp probability (alpha), enabled if > 0.'
    )
    parser.add_argument(
        '--cutmix', 
        type=float, 
        default=0.0,
        help='CutMix probability (alpha), enabled if > 0.'
    )
    parser.add_argument(
        '--cutmix-minmax', 
        dest='cutmix_minmax',
        type=float, 
        nargs='+', 
        default=None,
        help='CutMix min/max ratio, overrides alpha and enables cutmix if set (default: None)'
    )
    parser.add_argument(
        '--mixup-prob',
        dest='mixup_prob',
        type=float, 
        default=0.0,
        help='Probability of performing mixup or cutmix when either/both is enabled'
    )
    parser.add_argument(
        '--mixup-switch-prob', 
        dest='mixup_switch_prob',
        type=float, 
        default=0.0,
        help='Probability of switching to CutMix when both MixUp and CutMix enabled'
    )
    parser.add_argument(
        '--mixup-mode',
        dest='mixup_mode',
        type=str, 
        default='batch',
        help='How to apply MixUp/CutMix params. Per "batch", "pair", or "elem"'
    )
    
    ################## Logging ##################
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        default="./outputs",
        type=str,
        metavar="PATH",
        help="path to save output (default: ./outputs/)",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        default=None,
        type=str,
        help="Run name",
    )
    parser.add_argument(
        "--disable-wandb",
        dest="disable_wandb",
        action="store_true",
        help="Disable wandb for logging",
    )
    parser.add_argument(
        "--wandb-project",
        dest="wandb_project",
        default=None,
        type=str,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        dest="wandb_entity",
        default=None,
        type=str,
        help="Weights & Biases entity name",
    )
    parser.add_argument(
        "--wandb-id",
        dest="wandb_id",
        default=None,
        type=str,
        help="Weights & Biases run id",
    )
    parser.add_argument(
        "--log-dir",
        dest="log_dir",
        default=None,
        type=str,
        help="Log directory for txt files",
    )
    parser.add_argument(
        "--log-interval",
        dest="log_interval",
        default=1,
        type=int,
        help="Log interval for console",
    )
    parser.add_argument(
        "--lr-monitor",
        dest="lr_monitor",
        action="store_true",
        help="Monitor LR",
    )
    parser.add_argument(
        "--disable-progress-bar",
        dest="disable_progress_bar",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--save-overwrite",
        dest="save_overwrite",
        action="store_true",
        help="Overwrite save directory",
    )
    parser.add_argument(
        "--save-interval",
        dest="save_interval",
        type=int,
        default=1,
        help="Save interval (default: 1)",
    )
    parser.add_argument(
        "--brainscore-benchmarks-interval",
        dest="brainscore_benchmarks_interval",
        type=int,
        default=0,
        help="Interval for running brainscore benchmarks (default: 0)",
    )
    parser.add_argument(
        "--benchmarks",
        dest="benchmarks",
        type=str,
        default="V1_FZ,V2_FZ,V4_MH,IT_MH,Behavior_RJLHM",
        help="BrainScore benchmarks to run (default: V1_FZ,V2_FZ,V4_MH,IT_MH,Behavior_RJLHM)",
    )
    parser.add_argument(
        "--benchmark-metric",
        dest="benchmark_metric",
        type=str,
        default="ridge_cv",
        help="Metric for ridge regression. Either ridge_cv or ridge-`<alpha>`",
    )
    parser.add_argument(
        "--run-benchmarks-initial-epochs",
        dest="run_benchmarks_initial_epochs",
        action="store_true",
        help="Run benchmarks for the first 10 epochs",
    )
    
    
    return parser
