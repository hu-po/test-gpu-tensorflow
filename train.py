""" Mock training script for testing GPU support."""

import argparse
import os
import pprint
import time

import platform
import wandb

parser = argparse.ArgumentParser(description='Mock training script for testing GPU support.')
parser.add_argument("--project", type=str, default="launch-examples", help="W&B project")
parser.add_argument('--gpu', type=int, default=-1, help='Specify which GPU to use (-1 for all gpus)')

# How long should this fake training script run for?
parser.add_argument("--train_time", type=int, default=5)

if __name__ == "__main__":
    args = parser.parse_args()
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    run = wandb.init(project=args.project)

    # Print out wandb information
    print('\n\n---\tWandB Information\t---\n')
    print(f'\tentity: {run._entity or os.environ.get("WANDB_ENTITY")}')
    print(f'\tproject: {run._project or os.environ.get("WANDB_PROJECT")}')
    print(f'\tconfig: {pprint.pformat(wandb.config)}\n')

    # Print out some system information
    print('\n---\tSystem Information\t---\n')
    print(f'\tsystem: {platform.system()}')
    print(f'\tarchitecture: {platform.architecture()}')
    print(f'\tprocessor: {platform.processor()}')
    print(f'\tmachine: {platform.machine()}')
    print(f'\tpython_version: {platform.python_version()}')

    try:
        import tensorflow as tf
    except Exception as e:
        raise ImportError(f'Error importing Tensorflow: {e}')

    # Tell framework to use specific GPUs
    print('\n---\tGPU Information\t---\n')

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'\tTensorflow was able to find CUDA')
        print(f'\tTensorflow was able to find {len(gpus)} GPUs')
    else:
        raise AssertionError('Tensorflow was unable to find CUDA')

    _devices = []
    if args.gpu >= 0:
        assert args.gpu < len(gpus), f'Invalid GPU: {args.gpu} specified, but only found {len(gpus)}'
        print(f'\tUsing GPU {args.gpu}')
        _devices += [f'/device:GPU:{args.gpu}']
    else:
        print(f'\tUsing all {len(gpus)} available GPUs')
        _devices += [f'/device:GPU:{i}' for i in range(len(gpus))]

    # Do some fake taining
    assert args.train_time > 0, "Fake training must last longer than 0 seconds."
    print('\n---\tFake Training\t---\n')
    start_time = time.time()
    while time.time() - start_time < args.train_time:
        time_remaining = int(args.train_time - (time.time() - start_time))
        print(f'\tTraining, {time_remaining} seconds remaining.')
        wandb.log({"alive_time": time_remaining})
        for _device in _devices:
            with tf.device(_device):
                a = tf.random.normal([64, 64], 0, 1, tf.float32)
                b = tf.random.normal([64, 64], 0, 1, tf.float32)
                c = tf.matmul(a, b)


    wandb.finish()
