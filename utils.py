import datetime
import logging
import random
import numpy as np
import torch
from transformers import BertTokenizer, DistilBertTokenizer
from torch.optim import Adam
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    

def load_model(model, model_path):
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model.to(device)


def init_logger(args):
    logging_level = logging.INFO
#     if args.debug:
#         logging_level = logging.DEBUG
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)
    if args.logfile is not None and args.do_train:
        log_filename = '{}/seed-{}.txt'.format(args.logfile, args.seed)
        logging.basicConfig(
            # filename=log_filename,
            format='[%(asctime)s,%(msecs)d] %(levelname)s: %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging_level,
            handlers=[
                stream_handler,
                logging.FileHandler(log_filename, mode='w')
            ])
    else:
        logging.basicConfig(
            # filename=log_filename,
            format='[%(asctime)s,%(msecs)d] %(levelname)s: %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging_level,
            handlers=[
                stream_handler
            ])

    
def print_running_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s'%device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        logger.info(torch.cuda.get_device_name(0))
        logger.info('Memory Usage:')
        logger.info('Allocated: %f GB'%round(torch.cuda.memory_allocated(0)/1024**3,1))
        logger.info('Cached:   %f GB'%round(torch.cuda.memory_reserved(0)/1024**3,1))
    else:
        logger.info('devide: CPU')

def get_optimizer(args, model):

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_parameters, lr=args.learning_rate)

    return optimizer

def load_tokenizer(args):
    if "bert" in args.model_name_or_path:
        return BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        NotImplementedError("select a pretrained model type (bert)")

from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score

class Dataset():
    def __init__(self, x, y):
        '''
        x: is a list of sentecnes
        y: is a list of labels given for sentences in x
        '''
        self.size = len(x)
        
        self.sents = x
        
        self.labels = y
        
        self.items = []
        
    def __len__(self):
        
        return self.size

    def __getitem__(self, idx):
        
        x = self.sents[idx].lower().strip()
        
        y = self.labels[idx]
        
        sample = {"x":x, "y":y}
        
        return sample
    
def eval_one_episode(model, x_sup, x_q, y_sup, y_q):
    '''
    model: is a pretrained model that should be evaluated on the given episode >
            we assume the model object has a function name predict which predicts the labels for a query examples according to support examples
    x_sup: is a list of sentences in the support set of the episode
    x_q: is a list of sentences to which we should assign a label (x_q >> x_s)
    y_sup: is a list of labels given for sentecnes in x_s
    y_q: is a list of reference labels given for sentences in x_q
    '''

    with torch.no_grad():
        
        query_data = Dataset(x_q, y_q)
        
        query_dataloader = DataLoader(query_data, batch_size=5, shuffle=False)
        
        predictions = []
        
        targets = []
        
        for batch_id, batch in enumerate(query_dataloader):
            
            x_q = batch['x']     
            
            y_q = batch['y']
            
            y_q_hat = model.predict(x_support=x_sup, x_query=x_q, y_support=y_sup)
            
            predictions.extend(y_q_hat)
            
            targets.extend(y_q)

        acc = accuracy_score(predictions, targets)
        
        return acc, predictions

def evaluate(model, meta_dataset):
    '''
    model: is a model that should be evaluated
    meta_dataset: it is a set of episodes designed for evaluting a model
    '''
    
    model.eval()

    acc_scores = []
    
    for episode_id, episode in enumerate(meta_dataset):
        ''' episode = {
                            'xs': [list of support samples],
                            'xq': [list of query samples],
                            'ys_label': [list of label name],
                            'yq_label': [list of label names],
                            'labels': [list of unique intent labels],
                            'ys': [list of label indices],
                            'yq': [list of label indices]
                            }
        '''
        x_sup = episode['xs']
        x_q = episode['xq']
        y_sup = episode['ys']
        y_q = episode['yq']

        acc_score_episode, predictions_episode = eval_one_episode(model, x_sup, x_q, y_sup, y_q)

        acc_scores.append(acc_score_episode)

    acc_score_mean = np.mean(acc_scores)

    return acc_score_mean