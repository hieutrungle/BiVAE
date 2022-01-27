import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import configargparse
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from utils import utils
from train import train
from eval import evaluate
from generate import generate
from reconstruct import reconstruct_img
# from hard_code_model import VariationalAutoencoder
from model import VariationalAutoencoder
import data_io

import numpy as np
import tensorflow as tf

def main(args):

    strategy = tf.distribute.MirroredStrategy()
    batch_size = args.batch_size
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    print(f"global batch size: {global_batch_size}")
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    with strategy.scope():

        dataio = data_io.Data(args.data_path, global_batch_size, args.tile_size)

        iterator = iter(strategy.experimental_distribute_dataset(
            dataio.load_data(args.dataset))
            )

        # sys.exit()

        # Model Initialization
        in_shape = list(dataio.data_dim[1:])
        model_arch = utils.get_model_arch(args.model_arch)
        print(f"model_arch: {args.model_arch}")

        vae = VariationalAutoencoder(args, model_arch, global_batch_size, in_shape)
        print(f"is using se: {vae.use_se}\n")
        # vae.build(input_shape=([None]+ in_shape))
        vae.model().summary()

        # Set up for training, evaluation, or generation
        model_path = args.model_path
        print(f"\nlogging information to: {model_path}\n")
        
        resume_checkpoint={}
        if args.resume or args.generate:
            weight_path = model_path + '/checkpoints/' + f'model_{args.iter:06d}'
            vae.load_weights(weight_path)
            if args.resume:
                print(f"Resume trainig...")
                resume_checkpoint['resume_epoch'] = args.iter
                print(resume_checkpoint)
            print(f"Model weights successfully loaded.")
            
        # sys.exit()

        # Training, Generating, or Evaluating the model
        if args.generate:
            print(f"Generating images...")
            if (args.dataset == "mnist"):
                generate(vae, iterator, args.path_img_output)
            else:
                reconstruct_img(vae, iterator, normalizer=dataio.normalizer, padder=dataio.padder, 
                                img_folder=args.path_img_output, img_name='gen_image.png')
        else:
            if args.eval:
                print("Evaluation...")
                evaluate(vae, iterator, model_path=model_path, 
                        save_encoding=args.save_encoding, padding=None)
            else:
                # Training parameters
                epochs = args.epochs
                lr = args.learning_rate
                lr_min = args.learning_rate_min
                train_portion = args.train_portion
                steps_per_execution = dataio.data_dim[0] // global_batch_size

                # optimizer
                # decay_steps = int(dataio.data_dim[0] * train_portion) * (epochs/4)
                decay_steps = int(steps_per_execution*0.9)
                lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                    lr, first_decay_steps=decay_steps,
                    t_mul=2, m_mul=0.75, alpha=lr_min)
                optimizer = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)

                train(vae, iterator, epochs=epochs, optimizer=optimizer, train_portion=train_portion,
                    model_dir=model_path, batch_size=global_batch_size,
                    steps_per_execution=steps_per_execution,
                    kl_anneal_portion=args.kl_anneal_portion,
                    epochs_til_ckpt=args.epochs_til_ckpt,
                    steps_til_summary=args.steps_til_summary,
                    resume_checkpoint=resume_checkpoint, strategy=strategy)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cesm', 'isabel'],
                        help='which dataset to use, default="mnist')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='location of the data corpus')
    parser.add_argument('--tile_size', type=int, default=64,
                        help="tile size after partitioning scientific dataset")
    # Genral training options
    parser.add_argument('--eval', action='store_true', default=False,
                        help="run evaluation on testing dataset")
    parser.add_argument('--save_encoding', action='store_true', default=False,
                        help="save encoding vectors during eval")
    parser.add_argument('--generate', action='store_true', default=False,
                        help="run generation")
    parser.add_argument('--model_path', default="./model_output/distributed_training",
                        help="Path to model folder")
    parser.add_argument('--path_img_output', default=None,
                        help="Path to image output folder when generating new images")
    parser.add_argument('--train_portion', type=float, default=0.95,
                        help="train portion after spliting the original dataset")
    # logging options
    # parser.add_argument('--experiment_name', type=str, required=True,
    #                help='path to directory where checkpoints & tensorboard events will be saved.')
    parser.add_argument('--epochs_til_ckpt', type=int, default=5,
                        help="Epochs until checkpoint is saved")
    parser.add_argument('--steps_til_summary', type=int, default=50,
                        help="Number of iterations until tensorboard summary is saved")
    parser.add_argument('--logging_root', type=str, default='./logs',
                        help="root for logging")
    # optimization
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="batch size. default=32")
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=5e-5,
                        help='min learning rate')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--model_arch', type=str, default='res_wnelu',
                        help='which model architecture to use')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.4,
                        help='The portions epochs that KL is annealed')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=1,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    # latent variables
    parser.add_argument('--num_channels_of_latent', type=int, default=1,
                        help='number of channels of latent variables')
    # Initial channel
    parser.add_argument('--num_initial_channel', type=int, default=16,
                        help='number of channels in pre-enc and post-dec')
    # Share parameter of preprocess and post-process blocks
    parser.add_argument('--num_process_blocks', type=int, default=1,
                        help='number of preprocessing and post-processing blocks')
    # Preprocess cell
    parser.add_argument('--num_preprocess_cells', type=int, default=2,
                        help='number of cells per proprocess block')
    # Encoder and Decoder Tower
    parser.add_argument('--num_scales', type=int, default=2,
                        help='the number of scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=1,
                        help='number of groups per scale')
    parser.add_argument('--is_adaptive', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_cell_per_group_enc', type=int, default=1,
                        help='number of cells per group in encoder')
    # decoder parameters
    parser.add_argument('--num_cell_per_group_dec', type=int, default=1,
                        help='number of cell per group in decoder')
    # Post-process cell
    parser.add_argument('--num_postprocess_cells', type=int, default=2,
                        help='number of cells per post-process block')
    # Squeeze-and-Excitation
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    # Resume
    parser.add_argument('--resume', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    parser.add_argument('--iter', type=int, default=0,
                        help='resume iteration')
    args = parser.parse_args()

    if (args.generate and (args.model_path is None or args.path_img_output is None or args.iter is None)):
        parser.error('The --generate argument requires the --model_path and --path_img_output')

    if (args.resume and args.iter is None):
        parser.error('The --resume argument requires the --iter')

    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print()
    
    devices = tf.config.list_physical_devices()
    print(devices)
    print(f"Tennsorflow version: {tf.__version__}\n")

    main(args=args)