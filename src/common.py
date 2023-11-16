from argparse import Namespace
import stat
import tensorflow as tf
from typing import Callable, List, Tuple, Union,NamedTuple
from tensorboard.plugins.hparams import api as hp

class ReplayHistoryType(NamedTuple):
    states: tf.Tensor
    actions: tf.Tensor
    rewards: tf.Tensor
    next_states: tf.Tensor
    dones: tf.Tensor

class PPOReplayHistoryType(NamedTuple):
    states: tf.Tensor
    actions: tf.Tensor
    advantages: tf.Tensor
    returns: tf.Tensor
    logprobability: tf.Tensor
    dones: tf.Tensor

class HistorySampleType(NamedTuple):
    states: tf.Tensor
    actions: tf.Tensor
    logprobability: tf.Tensor
    advantages: tf.Tensor

class HistorySampleCriticType(NamedTuple):
    states: tf.Tensor
    returns: tf.Tensor

class EnvStepReturnType(NamedTuple):
    state: tf.Tensor
    reward: tf.Tensor
    done: tf.Tensor

EnvStepType = Callable[[tf.Tensor], EnvStepReturnType]

import datetime
import config

def splash_screen(params: Namespace):
    print(r"""
    $$\                 $$\  $$\                                              $$\       
    $$ |                $$ | $  |                                             $$ |      
    $$ |      $$$$$$\ $$$$$$\\_/ $$$$$$$\        $$$$$$$\  $$$$$$\   $$$$$$\  $$ |  $$\ 
    $$ |     $$  __$$\\_$$  _|  $$  _____|      $$  _____|$$  __$$\ $$  __$$\ $$ | $$  |
    $$ |     $$$$$$$$ | $$ |    \$$$$$$\        $$ /      $$ /  $$ |$$ /  $$ |$$$$$$  / 
    $$ |     $$   ____| $$ |$$\  \____$$\       $$ |      $$ |  $$ |$$ |  $$ |$$  _$$<  
    $$$$$$$$\\$$$$$$$\  \$$$$  |$$$$$$$  |      \$$$$$$$\ \$$$$$$  |\$$$$$$  |$$ | \$$\ 
    \________|\_______|  \____/ \_______/        \_______| \______/  \______/ \__|  \__|                                                                       
    """)
    run_name = config.RUN_NAME
    print(f" Version: {params.version} ".center(88, "="))
    print(f" Run Name: {run_name} ".center(88, "="))

    # if DRY_RUN = False, then add dry on the beginning of the run name
    if params.DRY_RUN:
        run_name = "dry_" + run_name
    train_summary_writer = tf.summary.create_file_writer(config.LOG_DIR_ROOT + run_name + params.version) # type: ignore
    train_summary_writer.set_as_default()

    nl = "\n"
    tf.summary.text("run_name", f"""
    ### {run_name}
    #### {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    ---
    ## params
    {nl.join([f"- {k}: {v}" for k, v in params.__dict__.items()])}
    """, step=0)
    
    return params