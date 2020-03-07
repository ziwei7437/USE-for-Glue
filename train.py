import os
import sys
import time
import argparse
import json
import pandas as pd
import numpy as np

import logging


import torch
from torch import optim
import torch.nn as nn

from tasks import get_task, MnliMismatchedProcessor
from models import InferSent, SimpleClassifier

import initialization
from runner import RunnerParameters, GlueTaskClassifierRunner

import tensorflow as tf
import tensorflow_hub as hub


def get_args(*in_args):
    parser = argparse.ArgumentParser(description='USE-for-Glue')

    # === Required Parameters ===
    parser.add_argument("--data_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="training dataset directory")
    parser.add_argument("--task_name",
                        type=str,
                        default=None,
                        required=True,
                        help='the name of the task to train.')
    parser.add_argument("--output_dir", 
                        type=str, 
                        default=None,
                        required=True,
                        help="Output directory")

    # === Optional Parameters ===
    # tasks
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_val", action="store_true")

    # training args for classifier
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    # model
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="classifier hidden dropout probability")
    parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
    parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")

    # gpu
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=-1, help="seed")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    # others
    parser.add_argument("--verbose", action="store_true", help='showing information.')
    
    args = parser.parse_args(*in_args)
    return args


def print_args(args):
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = get_args()
    print_args(args)
    device, n_gpu = initialization.init_cuda_from_args(args, logger=logger)
    initialization.init_seed(args, n_gpu=n_gpu, logger=logger)
    initialization.init_train_batch_size(args)
    initialization.init_output_dir(args)
    initialization.save_args(args)
    task = get_task(args.task_name, args.data_dir)
    use_cuda = False if args.no_cuda else True    parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")

    config = {
        'dropout_prob'  :   args.dropout_prob,
        'n_classes'     :   args.n_classes,
        'fc_dim'        :   args.fc_dim,
        'enc_dim'       :   512,
    }

    # load model
    print("loading Universal Sentence Encoder......")
    USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    classifier = SimpleClassifier(config)
    classifier = classifier.cuda() if not args.no_cuda else classifier

    # get train examples
    train_examples = task.get_train_examples()
    # calculate t_total
    t_total = initialization.get_opt_train_steps(len(train_examples), args)


    # build optimizer.
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    # create running parameters
    r_params = RunnerParameters(
        local_rank=args.local_rank,
        n_gpu=n_gpu,
        learning_rate=5e-5,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        t_total=t_total,
        warmup_proportion=args.warmup_proportion,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        verbose=verbose
    )
    
    # create runner class for training and evaluation tasks.
    runner = GlueTaskClassifierRunner(
        encoder_model = USE,
        classifier_model = classifier,
        optimizer = optimizer,
        label_list = task.get_labels(),
        device = device,
        rparams = r_params
    )

    if args.do_train:
        runner.run_train_classifier(train_examples)

    if args.do_val:
        val_examples = task.get_dev_examples()
        results = runner.run_val(val_examples, task_name=task.name, verbose=verbose)

        df = pd.DataFrame(results["logits"])
        df.to_csv(os.path.join(args.output_dir, "val_preds.csv"), header=False, index=False)
        metrics_str = json.dumps({"loss": results["loss"], "metrics": results["metrics"]}, indent=2)
        print(metrics_str)
        with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
            f.write(metrics_str)

        # HACK for MNLI-mismatched
        if task.name == "mnli":
            mm_val_example = MnliMismatchedProcessor().get_dev_examples(task.data_dir)
            mm_results = runner.run_val(mm_val_example, task_name=task.name, verbose=verbose)

            df = pd.DataFrame(results["logits"])
            df.to_csv(os.path.join(args.output_dir, "mm_val_preds.csv"), header=False, index=False)
            combined_metrics = {}
            for k, v in results["metrics"].items():
                combined_metrics[k] = v
            for k, v in mm_results["metrics"].items():
                combined_metrics["mm-"+k] = v
            combined_metrics_str = json.dumps({
                "loss": results["loss"],
                "metrics": combined_metrics,
            }, indent=2)
            print(combined_metrics_str)
            with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
                f.write(combined_metrics_str)

if __name__ == '__main__':
    main()