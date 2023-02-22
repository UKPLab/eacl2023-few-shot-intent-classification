import json
import copy
import logging
import os
import pdb
import time
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import torch
import torch.nn as nn
from allennlp.modules.elmo import batch_to_ids
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset
from tqdm import tqdm, trange
from data_loader import EpisodeDataset, NonEpisodeDataset
from model import (BEBC, CEBC, MBEMBC,)
from utils import (set_seeds, init_logger, load_model, save_model, print_running_device, eval_one_episode, evaluate, get_optimizer)
import argparse

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

     # path    
    parser.add_argument("--train_path", default='dataset_episodes/atis/train/train.csv', type=str, help="The train data")
    parser.add_argument("--dev_path", default='dataset_episodes/atis/seed-0/dev', type=str, help="The dir of development episodes")
    parser.add_argument("--test_path", default='dataset_episodes/atis/seed-0/test', type=str, help="The dir of testing episodes")
    parser.add_argument("--output_dir", default="tmp", type=str,  help="Path to save, load model and log files")    
    
    # model
    parser.add_argument("--model_name_or_path", type=str, required=True,  help="Transformer model to use")
    parser.add_argument("--pooling", type=str, default="cls", help="sentence embedding pooling", choices=("cls", "avg"))
    parser.add_argument("--encoder_type", default="be", type=str, help="Encoder type be (bi-ecoder) and ce (cross-encoder), mbe (metric-based be): {be | ce | mbe}")
    parser.add_argument("--max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization.")


    # training 
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_iter", type=int, default=1,  help="Max number of training episodes")
    parser.add_argument("--evaluate_every", type=int, default=100, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--log_every", type=int, default=10,  help="Number of training episodes between each logging")
    parser.add_argument("--early_stop", type=int, default=5e-5,  help="Number of worse evaluation steps before stopping. 0=disabled")    

    # evaluation
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    
    # Misc.
    parser.add_argument("--batch_size", type=int, default=64, help="batch size (should be greater than 1)")
    parser.add_argument("--seed", default = 1, type= int,  help="sedd value")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")



    args = parser.parse_args()
    
    if args.do_train:
        assert args.train_path, f"{args.train_path} is required for training"

    if args.do_eval:
        assert args.test_path, f"{args.test_path} is required for testing"

    args.logfile = args.output_dir
    if not os.path.exists(args.output_dir):
        # This will ignore if the output_dir exists --> overwrite previous logs
        os.makedirs(args.output_dir)


    init_logger(args)

    set_seeds(args)
    
    logger.info(args)
    
    return args


##
#
def train(args, model, train_data, dev_data):
    
    optimizer = get_optimizer(args, model)
    
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
    if args.no_cuda:
        device = "cpu"
        

    model.to(device)
    
    best_model = None

    loss = nn.BCELoss()
    
    logger.info('-------------- Training --------------')

    acc_score_dev = evaluate(model, dev_data)

    best_dev = {"step": -1, "acc": acc_score_dev}

    best_model_path = os.path.join(args.output_dir, "best_model.pt")

    n_eval_since_last_best = 0

    logger.info("Initial epoch")

    logger.info(f'acc_dev: {acc_score_dev  * 100:.2f}')

    save_model(model, best_model_path)

    tr_loss = 0
    
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle = False)    

    n_updates = 0
    
    while True:

        model.train()
        
        for batch_id, batch in enumerate(train_data_loader): 
            
            optimizer.zero_grad()
            
            if n_updates == args.max_iter:
                
                return
        
            x_batch = batch['x']

            y_batch = batch['y'].tolist()
            
            batch_size = len(x_batch)

            batch_loss = 0.0

            x_q = x_batch

            y_q = y_batch

            x_s = x_batch

            y_s = y_batch
            
            try: 
                scores = model(x_support = x_s, x_query = x_q)

                #scores = scores.squeeze(1)
                scores = torch.flatten(scores)

                target_scores = []

                for label_s in y_s:

                    for label_q in y_q:

                        if label_s == label_q:

                            target_scores.append(1)

                        else:

                            target_scores.append(0)


                target_scores = torch.Tensor(target_scores).to(device)


                torch.cuda.empty_cache()

                batch_loss = loss(scores, target_scores)

                batch_loss.backward()
               
                
            except:
                # if the size of the batch makes OOM in CUDA (happens in xenc)
                
                ###
                # examined values
                # mini_batch_size = 5 #(default)
                # mini_bath_size = 2 # CE LIU and HWU datasets (CV=1 and CV=2) because of CUDA OOM
                ###
                mini_batch_size = 5 

                torch.cuda.empty_cache()


                num_mini_batches = batch_size // mini_batch_size

                if batch_size % mini_batch_size != 0:

                    num_mini_batches +=  1


                t = 0

                loss_mini_batch = 0

                while t < num_mini_batches:

                    start_index = t*mini_batch_size
                    end_index = (t+1)*mini_batch_size

                    x_s_mini = x_s[start_index:end_index]
                    x_q_mini = x_q[start_index:end_index]

                    y_s_mini = x_s[start_index:end_index]
                    y_q_mini = x_q[start_index:end_index]

                    t += 1

                    scores = model(x_support = x_s_mini, x_query = x_q_mini)

                    #scores = scores.squeeze(1)
                    scores = torch.flatten(scores)

                    target_scores = []

                    for label_s in y_s_mini:

                        for label_q in y_q_mini:

                            if label_s == label_q:

                                target_scores.append(1)

                            else:

                                target_scores.append(0)


                    target_scores = torch.Tensor(target_scores).to(device)

                    loss_mini_batch = loss(scores, target_scores)

                    loss_mini_batch = loss_mini_batch / num_mini_batches
                    
                    loss_mini_batch.backward()            
                    
                    batch_loss += loss_mini_batch 


            optimizer.step()

            n_updates += 1

            if n_updates % args.log_every == 0:

                logger.info(f">>>>> n_updates {n_updates}: loss={batch_loss.item()}")

            tr_loss += batch_loss.item()

            if n_updates % args.evaluate_every == 0:

                acc_score_dev = evaluate(model, dev_data)

                if acc_score_dev >= best_dev["acc"]:

                    best_dev["acc"] = acc_score_dev

                    best_dev["step"] = batch_id

                    n_eval_since_last_best = 0

                    save_model(model, best_model_path)

                    logger.info(f'Best model saved at n_updates: {n_updates}, acc_score_dev:{acc_score_dev*100:.2f}')

                    logger.info(f"Saving model checkpoint to {best_model_path}")

                else:

                    n_eval_since_last_best += 1

                    logger.info(f"Worse acc_dev:{acc_score_dev*100:.2f}, Best acc: {best_dev['acc']*100:.2f}, number of worse scores on dev so far:{n_eval_since_last_best}")

                if args.early_stop > 0 and n_eval_since_last_best >= args.early_stop:

                    logger.warning(f"Early stopping.")

                    logger.info('------------End of training.')

                    return

    logger.info('------------End of training.')
    
##
#

# def train(args, model, train_data, dev_data):
    
#     optimizer = get_optimizer(args, model)
    
#     device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    
#     if args.no_cuda:
#         device = "cpu"
        

#     model.to(device)
    
#     best_model = None

#     loss = nn.BCELoss()
    

#     logger.info('-------------- Training --------------')

#     acc_score_dev = evaluate(model, dev_data)

#     best_dev = {"step": -1, "acc": acc_score_dev}

#     best_model_path = os.path.join(args.output_dir, "best_model.pt")

#     n_eval_since_last_best = 0

#     logger.info("Initial epoch")

#     logger.info(f'acc_dev: {acc_score_dev  * 100:.2f}')

#     save_model(model, best_model_path)

#     tr_loss = 0
    
#     train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle = False)    

#     n_updates = 0
    
#     while True:

#         model.train()
        
#         for batch_id, batch in enumerate(train_data_loader):
            
#             if n_updates == args.max_iter:
                
#                 return
        
#             x_batch = batch['x']

#             y_batch = batch['y'].tolist()
            
#             batch_size = len(x_batch)

#             batch_loss = 0.0

#             x_q = x_batch

#             y_q = y_batch

#             x_s = x_batch

#             y_s = y_batch
            
#             scores = model(x_support = x_s, x_query = x_q)

#             #scores = scores.squeeze(1)
#             scores = torch.flatten(scores)

#             target_scores = []

#             for label_s in y_s:
                
#                 for label_q in y_q:
                    
#                     if label_s == label_q:
                        
#                         target_scores.append(1)
                    
#                     else:
                        
#                         target_scores.append(0)

            
#             target_scores = torch.Tensor(target_scores).to(device)
            
#             batch_loss = loss(scores, target_scores)

#             batch_loss.backward()

#             optimizer.step()

#             optimizer.zero_grad()

#             n_updates += 1

#             if n_updates % args.log_every == 0:

#                 logger.info(f">>>>> n_updates {n_updates}: loss={batch_loss.item()}")

#             tr_loss += batch_loss.item()

#             if n_updates % args.evaluate_every == 0:

#                 acc_score_dev = evaluate(model, dev_data)

#                 if acc_score_dev >= best_dev["acc"]:

#                     best_dev["acc"] = acc_score_dev

#                     best_dev["step"] = batch_id

#                     n_eval_since_last_best = 0

#                     save_model(model, best_model_path)

#                     logger.info(f'Best model saved at n_updates: {n_updates}, acc_score_dev:{acc_score_dev*100:.2f}')

#                     logger.info(f"Saving model checkpoint to {best_model_path}")

#                 else:

#                     n_eval_since_last_best += 1

#                     logger.info(f"Worse acc_dev:{acc_score_dev*100:.2f}, Best acc: {best_dev['acc']*100:.2f}, number of worse scores on dev so far:{n_eval_since_last_best}")

#                 if args.early_stop > 0 and n_eval_since_last_best >= args.early_stop:

#                     logger.warning(f"Early stopping.")

#                     logger.info('------------End of training.')

#                     return

#     logger.info('------------End of training.')


##
#
def main():
    
    args = parse_args()
    
    print_running_device()
    
    logger.info(f'Loading data')
    
    meta_train = NonEpisodeDataset(args.train_path)
    
    meta_dev =  EpisodeDataset(args.dev_path)
    
    meta_test = EpisodeDataset(args.test_path)
    
    logger.info("Building model")

    if args.encoder_type == "be":
    
        model = BEBC(args)

    elif args.encoder_type == "ce":
    
        model = CEBC(args)
        
    else: 
        model = MBEMBC(args)
    
    if not args.do_train:
        
        logger.info(f'No training')

    else:
        
        train(args, model, meta_train, meta_dev)
        
    if args.do_eval == False:
        
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
        
        if meta_dev:
            st_time = time.time()
            acc_dev = evaluate(model, meta_dev)
            test_results["valid_runtime"] = time.time() - st_time
            test_results["valid_accuracy"] = acc_dev
            logger.info(f"Best dev acc: {acc_dev * 100:.2f}")
        
        if meta_test:
            st_time = time.time()
            acc_test = evaluate(model, meta_test)
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
    
    logger.info(f'config saved in {args.output_dir}/args.json.')
    logger.info(f'end')






##
#
# def main():
    
#     args = parse_args()
    
#     print_running_device()
    
#     logger.info(f'Loading data')
    
#     meta_train = NonEpisodeDataset(args.train_path)
    
#     meta_dev =  EpisodeDataset(args.dev_path)
    
#     meta_test = EpisodeDataset(args.test_path)
    
#     logger.info(f'Constructing model')
    
#     model = MBEMBC(args)
    
#     if not args.do_train:
        
#         logger.info(f'No training')

#     else:
        
#         train(args, model, meta_train, meta_dev)
        
#     if args.do_eval == False:
        
#         logger.info(f'No final evaluation?')
#     else:
        
#         logger.info('------------- Final Evaluation --------------')

#         best_model_path = os.path.join(args.output_dir, "best_model.pt")
        
#         if not os.path.exists(best_model_path):
            
#             logger.info(f'Faild to load the model from: {best_model_path}')
            
#             return
        
#         logger.info(f'Loading the best model from {best_model_path}')
        
#         model = load_model(model, best_model_path)
        
#         test_results = {
#             "valid_runtime": 0,
#             "valid_accuracy": 0,
#             "test_runtime": 0,
#             "test_accuracy": 0,
#         }
        
#         if meta_dev:
#             st_time = time.time()
#             acc_dev = evaluate(model, meta_dev)
#             test_results["valid_runtime"] = time.time() - st_time
#             test_results["valid_accuracy"] = acc_dev
#             logger.info(f"Best dev acc: {acc_dev * 100:.2f}")
        
#         if meta_test:
#             st_time = time.time()
#             acc_test = evaluate(model, meta_test)
#             test_results["test_runtime"] = time.time() - st_time
#             test_results["test_accuracy"] = acc_test
#             logger.info(f"Test acc: {acc_test * 100:.2f} ")
        
#         # Save test results
#         with open(os.path.join(args.output_dir, "test_results.json"), "w") as file:
#             json.dump(test_results, file, ensure_ascii=False)

#         logger.info(f'End of evaluation.')

#     # Save config
#     with open(os.path.join(args.output_dir, "args.json"), "w") as file:
#         json.dump(vars(args), file, ensure_ascii=False)
    
#     logger.info(f'condig saved in {args.output_dir}/args.json.')
#     logger.info(f'end')
    
if __name__ == '__main__':
    
    start_time = time.time()

    main()
    
    end_time = time.time()
    
    logger.info("Took %5.2f seconds" % (end_time - start_time))