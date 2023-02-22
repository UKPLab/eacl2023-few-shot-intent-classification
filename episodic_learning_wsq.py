import collections
import itertools
import json
import logging
import os
import pdb
import random
import time

import numpy as np
import torch
import torch.nn as nn

from data_loader import EpisodeDataset
from model import BEBC, CEBC, MBEMBC
from utils import (init_logger, load_model, save_model, set_seeds, print_running_device, get_optimizer)
from utils import evaluate

logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_binary_target_ids(y_query, y_support):
    binary_target_ids = []
    for q in y_query:
        # loop over query
        for s in y_support:
            if s == q: bin_label = 1.0
            else: bin_label = 0.0
            binary_target_ids.append(bin_label)
    binary_target_ids = torch.FloatTensor(binary_target_ids).to(device)
    return binary_target_ids


def create_batch_within_episode(episode):
    batch_size = max(1, 128 // len(episode['xs']))
    for bidx in range(0, len(episode['xq']), batch_size):
        yield {
                'xs': episode["xs"],
                'xq': episode["xq"][bidx:bidx+batch_size],
                'ys_label': episode["ys_label"],
                'yq_label': episode["yq_label"][bidx:bidx + batch_size],
                'labels': episode["labels"],
                'ys': episode["ys"],
                'yq': episode["yq"][bidx:bidx + batch_size]
        }

def get_loss(model, episode):
    loss = 0.0
    for batch_id, batch_episode in enumerate(create_batch_within_episode(episode)):
        # Get binary target ids
        binary_target_ids = get_binary_target_ids(
            y_query=batch_episode["yq"], y_support=batch_episode["ys"])
        # [n, m, d]
        sim_scores = model(x_support=batch_episode["xs"], x_query=batch_episode["xq"])
        batch_loss = nn.functional.binary_cross_entropy(
            sim_scores.squeeze(1), 
            binary_target_ids
        )
        batch_loss.backward()
        loss += batch_loss.item()
    loss /= (batch_id + 1)
    return loss


def train(args, model, train_dataset, dev_dataset):
    logger.info("Creating optimizer")
    optimizer = get_optimizer(args, model)
    
    train_results = {
        "step": -1,
        "train_runtime": 0,
        "train_loss": np.inf
    }
    logger.info("-------------- Intial dev performance --------------")
    acc_dev = evaluate(model, dev_dataset)
    best_dev = {"step": -1, "acc": acc_dev}
    logger.info(f"acc_dev: {acc_dev * 100:.2f}")
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    save_model(model, best_model_path)
    loss = nn.BCELoss()
    n_eval_since_last_best = 0
    n_train_episodes = len(train_dataset)
    episode_indices = list(range(n_train_episodes))
    # random.shuffle(episode_indices)
    st_time = time.time()

    logger.info("-------------- Training --------------")
    for step in range(args.max_iter):
        model.train()
        episode = train_dataset[episode_indices[step % n_train_episodes]]
        optimizer.zero_grad()
        # releases all unoccupied cached memory
        torch.cuda.empty_cache()
        loss = get_loss(model, episode)
        optimizer.step()

        if (step + 1) % args.log_every == 0:
            logger.info(f"train | step:{step+1} | loss:{loss:.4f}")
            train_results["step"] = step+1
            train_results["train_loss"] = loss
            train_results["train_runtime"] = time.time() - st_time
        
        if args.dev_path and (step + 1) % args.evaluate_every == 0:
            acc_dev = evaluate(model, dev_dataset)
            logger.info(f"dev | step:{step+1} | acc:{acc_dev:.4f}")
            if acc_dev > best_dev["acc"]:
                best_dev["acc"] = acc_dev
                best_dev["step"] = step+1
                n_eval_since_last_best = 0
                logger.info("---> Best dev results: {}".format(best_dev["acc"]))
                save_model(model, best_model_path)
                logger.info(f"Saving model checkpoint to {best_model_path}")
            else:
                n_eval_since_last_best += 1
                logger.info(f"Worse dev results ({n_eval_since_last_best} / {args.early_stop})")

            if args.early_stop and n_eval_since_last_best >= args.early_stop:
                logger.warning(f"Early stopping.")
                break

    # Save train results
    with open(os.path.join(args.output_dir, "train_results.json"), "w") as file:
        json.dump(train_results, file, ensure_ascii=False)
    logger.info('-------------- End training.')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser("Run episodic learning with support query set")
    parser.add_argument("--do_train", action="store_true", help="Training mode")
    parser.add_argument("--do_test", action="store_true", help="Testing mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--no_cuda", action="store_true", help="Disable cuda")

    # Task related config
    parser.add_argument("--train_path", default=None, type=str, help="The train data dir")
    parser.add_argument("--dev_path", default=None, type=str, help="The development data dir")
    parser.add_argument("--test_path", default=None, type=str, help="The test data dir")
    
    # Model config
    parser.add_argument("--model_name_or_path", type=str, required=True, 
                        help="Transformer model to use")
    parser.add_argument("--output_dir", default=None, required=True, type=str,
                        help="Path to save, load model")
    parser.add_argument("--encoder_type", default="ce", type=str, help="Encoder type be (bi-ecoder) and ce (cross-encoder): {be | ce}")
    parser.add_argument("--pooling", type=str, default="cls", 
                        help="sentence embedding pooling", choices=("cls", "avg"))

    # Training setting
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, 
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_iter", type=int, default=10000, 
                        help="Max number of training episodes")
    parser.add_argument("--evaluate_every", type=int, default=100, 
                        help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--log_every", type=int, default=10, 
                        help="Number of training episodes between each logging")
    parser.add_argument("--early_stop", type=int, default=0, 
                        help="Number of worse evaluation steps before stopping. 0=disabled")

    # Specific params
    parser.add_argument("--batch_size", type=int, default=32, help="Random seed to set")

    args = parser.parse_args()

    if args.do_train:
        assert args.train_path, f"{args.train_path} is required for training"

    if args.do_test:
        assert args.test_path, f"{args.test_path} is required for testing"

    args.logfile = args.output_dir
    if not os.path.exists(args.output_dir):
        # This will ignore if the output_dir exists --> overwrite previous logs
        os.makedirs(args.output_dir)

    init_logger(args)
    # Set random seed
    set_seeds(args)
    logger.info(str(args))

    return args


def main():
    args = parse_args()
    print_running_device()
    
    # Check if data directory(s) exist
    for arg in [args.train_path, args.dev_path, args.test_path]:
        if arg and not os.path.exists(arg):
            raise FileNotFoundError(f"Data @ {arg} not found.")

    # Load data
    logger.info("Loading data")
    if args.do_train:
        train_dataset = EpisodeDataset(args.train_path)
    if args.dev_path:
        dev_dataset = EpisodeDataset(args.dev_path)
    else: dev_dataset = None
    if args.test_path:
        test_dataset = EpisodeDataset(args.test_path)
    else: test_dataset = None

    logger.info("Building model")
    if args.encoder_type == "be":
        model = BEBC(args)
    elif args.encoder_type == "ce":
        model = CEBC(args)
    elif args.encoder_type == "bemb":
        model = MBEMBC(args)
    model.to(device)
    print_running_device()
    if args.do_train:
        train(args, model, train_dataset, dev_dataset)
    else:
        logger.info(f'No training')

    if not args.do_test:
        logger.info(f'No final evaluation?')
    else:
        logger.info('------------- Final Evaluation --------------')
        
        best_model_path = os.path.join(args.output_dir, "best_model.pt")
        
        if not os.path.exists(best_model_path):
            
            logger.info(f'Faild to load the model from: {best_model_path}')
            
            return
        
        logger.info(f'Loading the best model from {best_model_path}')
        
        model = load_model(model, best_model_path)
        test_results = {
            "valid_runtime": 0,
            "valid_accuracy": 0,
            "test_runtime": 0,
            "test_accuracy": 0,
        }
        if dev_dataset:
            st_time = time.time()
            acc_dev = evaluate(model, dev_dataset)
            test_results["valid_runtime"] = time.time() - st_time
            test_results["valid_accuracy"] = acc_dev
            logger.info(f"Best dev acc: {acc_dev * 100:.2f}")
        if test_dataset:
            st_time = time.time()
            acc_test = evaluate(model, test_dataset)
            test_results["test_runtime"] = time.time() - st_time
            test_results["test_accuracy"] = acc_test
            logger.info(f"Test acc: {acc_test * 100:.2f} ")
        
        # Save test results
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as file:
            json.dump(test_results, file, ensure_ascii=False)
        logger.info(f'End of evaluation.')

    # Save config
    with open(os.path.join(args.output_dir, "args.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info("Took %5.2f seconds" % (end_time - start_time))
