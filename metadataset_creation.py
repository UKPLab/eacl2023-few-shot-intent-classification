import time
import logging
import torch
from utils import  load_tokenizer, parse_args
from torch.utils.data import Dataset, TensorDataset
import numpy as np
import math
import os

logger = logging.getLogger(__name__)


def get_intent_labels(path):
    """Returns a list of all intent labels stored in the specified path """
    return [label.strip() for label in open(path, 'r', encoding='utf-8')]


def get_slot_labels(path):
    """Returns a list of all slot labels stored in the specified path """
    return [label.strip() for label in open(path, 'r', encoding='utf-8')]


def get_unique_slots(slot_list):
    """Returns a list of unique slots for a given list of slot_labels"""
    unique_slots=[]
    for slots in slot_list:
        if type(slots) == str:
            slots = slots.split()
        for slot in slots:
            if slot not in unique_slots:
                unique_slots.append(slot)
    return unique_slots


def process_unknown_slots(slot_list, all_unique_slots):
    """Returns a new list of slots where unknown slot labels not seen in all_unique_slots is mapped to O """
    result=[]
    for i in range(len(slot_list)):
        slot = ""
        for slot_val in slot_list[i].split():
            if slot_val not in all_unique_slots:
                slot += "O "
            else:
                slot += slot_val + " "
        # get rid of last whitespace
        result.append(slot[:len(slot)-1])
    return result


class MetaDataset(Dataset):
    """ Our Fewshot IC/SF dataset.
    returns a pytorch dataset of length max_episodes where each episode
    consists of a support and query set sampled according to the algorithm above"""

    def __init__(self, args, individual_dataset, split, tokenizer, k_max=None,  max_episodes=None):
        """
        Args:
           args (ArgumentParse): Arguments for specific run
           individual_train (string): Joint training when None else specific task name to sample from
           split (string): train, dev or test as strings
           tokenizer: the transformers tokenizer e.g BertTokenizer
        """
        self.args = args
        self.data_dir = args.data_dir
        self.split = split
        self.max_episodes = args.max_episodes

        if max_episodes is not None:
            self.max_episodes = max_episodes

        self.k_max = args.k_max

        if max_episodes is not None:
            self.k_max = k_max

        self.individual_dataset = individual_dataset

        self.tokenizer = tokenizer

        # the maximum sentence length for BERT
        self.max_seq_length = self.args.max_seq_length

        # the list of all intent labels
        self.intent_labels = get_intent_labels(os.path.join(args.data_dir, args.intent_label_file))

        # the list of all slot labels
        self.slot_labels = get_slot_labels(os.path.join(self.args.data_dir, args.slot_label_file))

        # seed value for deterministic episode creation: Will be incremented by one
        self.seed = args.seed

        # random number generator for seeded sampling
        self.RNG = np.random.RandomState(args.seed)

        # save the data in a dictionary. Keys are intent class ids (class_id:int) and content is a dataframe including all samples of that class
        self.data_dict = self.load_full_data()

        # compute the size of the dataset (total number instances in thee given split, i.e. train, dev, test)
        self.full_data_size = len(pd.concat(self.data_dict.values()))
        print("number of utterances in split (full split size):", self.full_data_size)

        # list of all ids for intent labels
        self.all_class_ids = list(self.data_dict.keys())

        # perform this function in init
        self.create_episodes()

    def load_full_data(self):

        if self.individual_dataset == None or (self.individual_dataset.startswith('snips') and self.split == 'dev'):
            if True:
                dataset = 'atis'
            else:
                dataset = sample_dataset_uniformly(self.RNG, self.split)
        else:
            dataset = self.individual_dataset


        data = {}

        # We define a new id  (new_id) for data points as "u_id" is not unique in the TOP dataset
        start_index = 0

        # where should we load the data from?
        split_path = os.path.join(self.data_dir, dataset, self.split)

        for curr_class_id in sorted(os.listdir(split_path)):
            df = pd.read_csv(os.path.join(split_path, curr_class_id), sep="\t")
            curr_class_id = int(curr_class_id)
            data[curr_class_id]= df[["utterance", "slot-labels", "intent", "u_id"]]
            end_index = len(data[curr_class_id])
            data[curr_class_id].insert(4, 'new_id', range(start_index, start_index + end_index))
            start_index += end_index
        return data

    def num_samples_per_class_id(self, class_ids):
        out = np.array([len(self.data_dict[cl]) for cl in class_ids])
        return out

    def compute_num_query(self, class_ids):
        """Computes the number of query points for an episode
          returns the number of query samples per class
          episodes are supposed to be balanced for fair comparisons
        """

        num_classes = self.num_samples_per_class_id(class_ids)

        out =  min(10, np.min(np.floor(0.5 * num_classes)))

        return  int(out)

    def compute_support_size(self, class_ids, num_query):
        beta = 0.3
        num_classes = self.num_samples_per_class_id(class_ids)
        support_size_contributions = np.minimum(20, (num_classes - num_query))
        out =  np.minimum(self.k_max, np.ceil(beta * support_size_contributions).sum())
        return int(out)

    def sample_num_support_per_class(self, class_ids, n_way, S, num_query):
        num_classes = self.num_samples_per_class_id(class_ids)
        L = n_way  # selected classes
        unnormalized_proportions = np.exp(self.RNG.uniform(math.log(0.5, 10), math.log(2, 10), n_way)) * num_classes
        R_l = (unnormalized_proportions / unnormalized_proportions.sum())
        num_desired_per_class = np.floor(R_l * (S - L)).astype('int32') + 1
        out = np.minimum(num_desired_per_class, (num_classes - num_query))
        return out

    def create_episodes(self):

        # we keep buffer as a list of new_ids to first process items that have not visited for query set yet
        buffer = {k:list(v['new_id']) for k,v in self.data_dict.items()}

        # keep tracking of N_way|selected_classes|num_query_shot_per_class|max_total_sup_size|class_label:num_of_selected_shots_of_class_for_sup
        summary = {}

        self.supports = []  # support set
        self.queries =  []   # query set

        # we save  new_id to data points that are used for query sets so far
        query_ids = set()

        episode_id = -1

        # generate episodes until we see any datapoints in a queryset
        while len(query_ids) < self.full_data_size:

            episode_id += 1
            summary[episode_id] = ""

            # Step 1: select n way randomly
            class_count = len(self.all_class_ids)


            n_way = self.RNG.randint(low=3, high=class_count + 1)

            summary[episode_id] += str(n_way)+"|"

            class_ids = [self.all_class_ids.pop(0) for i in range(n_way)]
            for item in class_ids:
                summary[episode_id] += str(item)+","
            summary[episode_id] += "|"
            self.RNG.shuffle(class_ids)
            self.all_class_ids.extend(class_ids)

            # Step 2: compute query size (number_of_shots) per class
            num_query = self.compute_num_query(class_ids)

            summary[episode_id]+= f"{num_query}|"

            # compute max total support size
            S = self.compute_support_size(class_ids, num_query)

            summary[episode_id]+= str(int(S)) +"|"

            # compute the support size for each class
            num_support_per_class = self.sample_num_support_per_class(class_ids, n_way, S, num_query)

            all_supports_classes = [] # all support points for a given episode
            all_query_classes = []     # all query points for a given episode


            for i in range(n_way): # for every selected intent class

                curr_class_id = class_ids[i]
                curr_num_of_support = int(num_support_per_class[i])

                summary[episode_id]+= (f"{curr_class_id}:{curr_num_of_support},")

                # shuffle buffer to avoid  any bias
                self.RNG.shuffle(buffer[curr_class_id])

                # select query samples that have not been seen yet
                selected_u_ids_for_query = []
                for i in range(num_query):
                    for u_id in buffer[curr_class_id]:
                        if u_id not in query_ids:
                            selected_u_ids_for_query.append(u_id)
                            query_ids.add(u_id)
                            buffer[curr_class_id].remove(u_id)
                            break

                # cover all samples that are needed for the query set
                selected_u_ids_for_query.extend([buffer[curr_class_id].pop(0) for i in range(num_query-len(selected_u_ids_for_query))])

                # select sup samples from the beginning of the buffer
                selected_u_id_for_sup = buffer[curr_class_id][:int(curr_num_of_support)]

                # extract examples for query set based on u_ids
                exam_query = self.data_dict[curr_class_id].loc[
                        self.data_dict[curr_class_id]['new_id'].isin(selected_u_ids_for_query), ["utterance", "slot-labels","intent", "new_id"]].values

                # extract examples for sup set based on u_ids
                exam_support = self.data_dict[curr_class_id].loc[self.data_dict[int(curr_class_id)]['new_id'].isin(selected_u_id_for_sup), ["utterance", "slot-labels","intent", "new_id"]].values

                # append selected_id for query to the buffer to be used for next episode
                buffer[curr_class_id].extend(selected_u_ids_for_query)

                # double check unique examples in query set
                for e in exam_query:
                    query_ids.add(e[-1])

                all_supports_classes.extend(exam_support)
                all_query_classes.extend(exam_query)

                #let's suffule examples in the sup and q to avoid putting samples with same intent next to each other
                self.RNG.shuffle(all_supports_classes)
                self.RNG.shuffle(all_query_classes)

            self.supports.append(all_supports_classes)
            self.queries.append(all_query_classes)

            # TODO: do we still need to increment seed?
            # Finally, after creating each episode, we increment the seed value and RNG to allow for variable sampling
            # Not doing so will result in the same dataset, n-way and shots in each episode
            self.seed = self.seed + 1
            self.RNG = np.random.RandomState(self.seed)

            print(f"episode_id:{episode_id}, count_visited_samples_in_query_so_far: {len(query_ids)}")

            # print(summary[episode_id])

        with open(f"./{self.args.task}_seed:{self.args.seed}_episodes.stat","a") as f:
            f.write("\n".join(summary.values()))
            f.write("\n====\n")

        # check #supports MUST equal to #queries
        assert len(self.supports) == len(self.queries)

    def create_feature_set(self, examples, example_index):
        """Formats a given input array to BERT's requirements and preprocesses the
           slot label as described above

           Args:
             examples: Either a single support set or query set
             example_index: List index of the support/queryset

          Returns:
             A TensorDataset where slot labels have been preprocessed and input has been formatted for BERT
        """
        assert type(example_index) == int, "example_index: {} {}".format(type(example_index), example_index)
        
        # First process slots:
        # Get unique slot values of support and query set
        # TODO this should be done outside so that 
        # no need to build common slots twice 
        # separately for supports and queries
        support_slots = np.array(self.supports[example_index])[:,1]
        query_slots   = np.array(self.queries[example_index])[:,1]
        unique_support_slots = get_unique_slots(support_slots)
        unique_query_slots= get_unique_slots(query_slots)
        # get only slot values contained in both support and query
        common_slots = np.intersect1d(unique_support_slots, unique_query_slots)
        examples = np.array(examples)
        # Map all excluded slots to 'O'
        examples[:,1] = process_unknown_slots(examples[:,1], common_slots)

        all_input_ids        = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_attention_mask   = torch.empty(len(examples), self.max_seq_length, dtype = torch.float)
        all_segment_ids      = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_slot_label_ids   = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_intent_label_ids = torch.empty(len(examples), dtype = torch.long)

        for index, (sent, slot, intent, uid) in enumerate(examples):
            input_ids = self.tokenizer.encode(sent) # includes [CLS] and [SEP] tokens
            attention_mask = [1] * len(input_ids)
            segment_ids    = [0] * len(input_ids)
            slot_label_ids = []
            slot_label_ids.append(self.slot_labels.index("PAD"))

            """ For slot filling we need add labels by tokenizing word by word taking 
                into account the subwords:
  
                E.g:  Sentence:   restaurant        nearby
                      Slots   :      O       BookRestaurant/location
  
                Tokenized sent:   rest  ##ura      ##nts            nearby
                      Slots   :    O    [ignore]  [ignore]    BookRestaurant/location
            """
            for word, slot_label in zip(sent.split(), slot.split()):
                word_tokens = self.tokenizer.tokenize(word)
                slot_index = self.slot_labels.index(slot_label) if slot_label in self.slot_labels else self.slot_labels.index("UNK")
                slot_label_ids.extend([slot_index] + [0] * (len(word_tokens) - 1)) # ignore value is 0 same as [PAD] value

            slot_label_ids.append(self.slot_labels.index("PAD"))
            while len(input_ids) < self.max_seq_length:
                # Add [PAD] label of 0 until constant max sentence length
                input_ids.append(0)
                attention_mask.append(0)
                segment_ids.append(0)
                slot_label_ids.append(0)

            intent_label_id = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            # TODO use list then convert to Tensor later...
            all_input_ids[index] = torch.Tensor(input_ids).to(torch.long)
            all_attention_mask[index] = torch.Tensor(attention_mask).to(torch.float)
            all_segment_ids[index] = torch.Tensor(segment_ids).to(torch.long)
            all_intent_label_ids[index] = torch.Tensor([intent_label_id]).to(torch.long)
            all_slot_label_ids[index] = torch.Tensor(slot_label_ids).to(torch.long)

        tensor_set = TensorDataset(
            all_input_ids, all_attention_mask, all_segment_ids, 
            all_intent_label_ids, all_slot_label_ids)
        return tensor_set

    def __getitem__(self, index):
        support_set = self.create_feature_set(self.supports[index], index)
        query_set   = self.create_feature_set(self.queries[index], index)
        return support_set, query_set

    def __len__(self):
        # TODO: now that we don't have max_episodes, what should we return as __len__
        return len(self.supports)


def process_set(supset):
    outputs = []

    for shot_id, shot in enumerate(supset):
        shot_text, token_tags, intent_label, u_id = shot
        outputs.append(f"{shot_id}|{shot_text}|{intent_label}|{token_tags}")

    return outputs

def save_episodes_to_files(meta_set, dir_path):

    for episode_id, support in enumerate(meta_set.supports):
        query = meta_set.queries[episode_id]
        sup_set = process_set(support)
        q_set = process_set(query)
        text = "\n".join(sup_set)
        text += "\n\n"
        text += "\n".join(q_set)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(os.path.join(dir_path, f"episode - {episode_id}"), "w") as f:
            f.write(text)

def main():
    args = parse_args()
    tokenizer = load_tokenizer(args)

    print(args)

    k_max = 20
    max_episode =  5000

    meta_train = MetaDataset(args, individual_dataset=args.task, split="train", tokenizer=tokenizer, k_max=k_max,  max_episodes=max_episode)
    meta_test = MetaDataset(args=args, individual_dataset=args.task, split='test', tokenizer=tokenizer, k_max=k_max,  max_episodes=max_episode)


    dir_path = f"./dataset_episodes/{args.task}/seed-{args.seed}/train"
    save_episodes_to_files(meta_train, dir_path)
    print(f"Save train episodes at {dir_path}")

    dir_path = f"./dataset_episodes/{args.task}/seed-{args.seed}/test"
    save_episodes_to_files(meta_test, dir_path)
    print(f"Save test episodes at {dir_path}")

    if not args.task.startswith("snips"):
        meta_dev = MetaDataset(args, individual_dataset=args.task, split="dev", tokenizer=tokenizer, k_max=k_max,  max_episodes=max_episode)
        dir_path = f"./dataset_episodes/{args.task}/seed-{args.seed}/dev"
        save_episodes_to_files(meta_dev, dir_path)
        print(f"Save dev episodes at {dir_path}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info("Took %5.2f seconds" % (end_time - start_time))