import argparse
import gc
import logging
import os

import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

import initialization
from runner import convert_examples_to_features, get_full_batch, get_label_mode
from train import print_args

import tensorflow_hub as hub

from tqdm.auto import tqdm, trange
from tasks import get_task, MnliMismatchedProcessor


def get_dataloader(input_examples, label_map, batch_size):
    train_features = convert_examples_to_features(
        input_examples, label_map, verbose=True,
    )
    full_batch = get_full_batch(
        train_features, label_mode=get_label_mode(label_map),
    )
    sampler = SequentialSampler(full_batch.pairs)
    dataloader = DataLoader(
        full_batch.pairs, sampler=sampler, batch_size=batch_size,
    )
    return dataloader


def get_args(*in_args):
    parser = argparse.ArgumentParser(description='USE-for-Glue-Embedding')
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
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    args = parser.parse_args(*in_args)
    return args


def run_encoding(loader, model, args, mode='train'):
    # run embedding for train set
    tensor_list_a, tensor_list_b, labels_tensor_list = [], [], []
    for step, batch in enumerate(tqdm(loader, desc="Encode {} set".format(mode))):
        s1 = list(batch[0])
        s2 = list(batch[1])

        s1_emb = model(s1).numpy()
        s2_emb = model(s2).numpy()

        s1_emb = torch.tensor(s1_emb)
        s2_emb = torch.tensor(s2_emb)
        labels = batch[-1] if (mode != 'test' or mode != 'mm_test') else None

        tensor_list_a.append(s1_emb)
        tensor_list_b.append(s2_emb)
        if mode != 'test' or mode != 'mm_test':
            labels_tensor_list.append(labels)

    # save the rest part
    train_emb_a = torch.cat(tensor_list_a).cpu()
    train_emb_b = torch.cat(tensor_list_b).cpu()

    print("shape of {} set sentence a: {}".format(mode, train_emb_a.shape))
    print("shape of {} set sentence b: {}".format(mode, train_emb_b.shape))

    if mode != 'test' or 'mm_test':
        train_labels = torch.cat(labels_tensor_list).cpu()
        print("shape of {} set labels: {}".format(mode, train_labels.shape))
        dataset_embeddings = TensorDataset(train_emb_a, train_emb_b, train_labels)
    else:
        dataset_embeddings = TensorDataset(train_emb_a, train_emb_b)

    # save to output dir
    torch.save(dataset_embeddings,
               os.path.join(args.output_dir, "{}.dataset".format(mode)))
    print("embeddings saved at: {}".format(os.path.join(args.output_dir, "{}.dataset".format(mode))))
    del dataset_embeddings
    gc.collect()


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = get_args()
    print_args(args)
    initialization.init_output_dir(args)
    initialization.save_args(args)
    task = get_task(args.task_name, args.data_dir)

    # load model
    print("loading Universal Sentence Encoder......")
    USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    # prepare dataset
    label_list = task.get_labels()
    label_map = {v: i for i, v in enumerate(label_list)}

    print("loading raw data ... ")
    train_examples = task.get_train_examples()
    val_examples = task.get_dev_examples()
    test_examples = task.get_test_examples()

    print("converting to data loader ... ")
    train_loader = get_dataloader(train_examples, label_map, args.train_batch_size)
    val_loader = get_dataloader(val_examples, label_map, args.eval_batch_size)
    test_loader = get_dataloader(test_examples, label_map, args.test_batch_size)

    # run embedding for train set
    print("Run embedding for train set")
    for _ in trange(1, desc="Epoch"):
        run_encoding(loader=train_loader,
                     model=USE,
                     args=args,
                     mode='train')

    print("Run embedding for dev set")
    for _ in trange(1, desc="Epoch"):
        run_encoding(loader=val_loader,
                     model=USE,
                     args=args,
                     mode='dev')

    print("Run embedding for test set")
    for _ in trange(1, desc="Epoch"):
        run_encoding(loader=test_loader,
                     model=USE,
                     args=args,
                     mode='test')

    # HACK FOR MNLI mis-matched
    if args.task_name == 'mnli':
        print("Run Embedding for MNLI Mis-Matched Datasets")
        print("loading raw data ... ")
        mm_val_example = MnliMismatchedProcessor().get_dev_examples(args.data_dir)
        mm_test_examples = MnliMismatchedProcessor().get_test_examples(args.data_dir)
        print("converting to data loader ... ")
        mm_val_loader = get_dataloader(mm_val_example, label_map, args.eval_batch_size)
        mm_test_loader = get_dataloader(mm_test_examples, label_map, args.test_batch_size)

        print("Run embedding for mm_dev set")
        for _ in trange(1, desc="Epoch"):
            run_encoding(loader=mm_val_loader,
                         model=USE,
                         args=args,
                         mode='mm_dev')

        print("Run embedding for test set")
        for _ in trange(1, desc="Epoch"):
            run_encoding(loader=mm_test_loader,
                         model=USE,
                         args=args,
                         mode='mm_test')


if __name__ == '__main__':
    main()
