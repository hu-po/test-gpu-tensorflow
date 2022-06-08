""" Mock training script for testing GPU support."""

import argparse
import os
import pprint
import time

import platform
import wandb
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, NVMLError

parser = argparse.ArgumentParser(description='Mock training script for testing GPU support.')
parser.add_argument("--project", type=str, default="summerhack", help="W&B project")
parser.add_argument('--gpu', type=str, default="0", help='Comma seperated list of GPU ids to use')

# How long should this fake training script run for?
parser.add_argument("--train_time", type=int, default=30)

# TODO: args?
TARGET_GPU_UTILIZATION = 0.9
MIN_LOG_INTERVAL = 1
GPU_MEM_GROWTH_RATE = 1.07
INITIAL_TENSOR_SIZE = 8

if __name__ == "__main__":
    args = parser.parse_args()
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    run = wandb.init(project=args.project)
    run.log_code(".")

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

    # Use NVIDIA Python bindings to get GPU information
    print('\n---\tGPU Information\t---\n')
    nvmlInit()
    nvidia_devices = {}
    for id in args.gpu.split(','):
        try:
            _id = int(id)
            _nvidia_device = nvmlDeviceGetHandleByIndex(_id)
        except (NVMLError, TypeError) as e:
            raise AssertionError(f"Error getting device handle to GPU {_id}: {e}")
        print(f'\tGPU {id} has {nvmlDeviceGetMemoryInfo(_nvidia_device).total} total memory')
        print(f'\tGPU {id} has {nvmlDeviceGetMemoryInfo(_nvidia_device).free} free memory')
        print(f'\tGPU {id} has {nvmlDeviceGetMemoryInfo(_nvidia_device).used} used memory')
        nvidia_devices[_id] = _nvidia_device
    assert len(nvidia_devices) > 0, 'No NVIDIA GPUs found'

    # Import framework and check for GPU support
    try:
        import tensorflow as tf
    except Exception as e:
        raise ImportError(f'Error importing Tensorflow: {e}')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'\tTensorflow was able to find CUDA')
        print(f'\tTensorflow was able to find {len(gpus)} GPUs')
    else:
        raise AssertionError('Tensorflow was unable to find CUDA')
    # Prevent TF from allocating entire GPU, slowly grow over time
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Register devices with Tensorflow
    devices = {}
    for id in nvidia_devices.keys():
        devices[id] = f'/device:GPU:{args.gpu}'
        print(f'\tTensorflow: created device using GPU {id}')

    # Increase size of tensor over time until max GPU memory utilization
    tensor_size = {id: INITIAL_TENSOR_SIZE for id in devices.keys()}

    # Do some fake taining
    assert args.train_time > 0, "Fake training must last longer than 0 seconds."
    print('\n---\tFake Training\t---\n')
    start_time = time.time()
    old_time_remaining = args.train_time
    while time.time() - start_time < args.train_time:
        time_remaining = int(args.train_time - (time.time() - start_time))
        # Don't spam too much
        if old_time_remaining - time_remaining < MIN_LOG_INTERVAL:
            continue
        old_time_remaining = time_remaining
        print(f'\tTraining, {time_remaining} seconds remaining.')
        wandb.log({"time.remaining": time_remaining})
        wandb.log({"time.now": time.time()})
        for id, device in devices.items():
            _tensor_size = int(tensor_size[id])
            with tf.device(device):
                a = tf.random.normal([_tensor_size, _tensor_size], 0, 1, tf.float32)
                b = tf.random.normal([_tensor_size, _tensor_size], 0, 1, tf.float32)
                c = tf.matmul(a, b)
            _used = nvmlDeviceGetMemoryInfo(_nvidia_device).used
            _total = nvmlDeviceGetMemoryInfo(_nvidia_device).total
            utilization = _used / _total
            wandb.log({f"gpu.mem.utilization.{id}": utilization})
            print(f'\t\tGPU {id} Memory Utilization at {utilization}')
            if utilization < TARGET_GPU_UTILIZATION:
                tensor_size[id] *= GPU_MEM_GROWTH_RATE
            print(f'\t\t A ({_tensor_size},{_tensor_size}) x B ({_tensor_size},{_tensor_size})')

    wandb.finish()
