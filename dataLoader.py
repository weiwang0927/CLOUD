import torch
import torch.utils.data as Data
import numpy as np
from tqdm import tqdm
import random
import linecache


def create_sub_seqs(seq, max_seqs_len):
    seqs_len = len(seq)
    index = random.randint(0, seqs_len - 1)
    sub_seqs_len = random.randint(1, max_seqs_len)
    seq = seq[index: index + sub_seqs_len]
    return seq


def get_item_appear_num(itemID, list):

    sim_items = list
    nums = sim_items.count(itemID)
    return nums


def get_sim_items(sim_seq_list):

    sim_items_list = []

    for seq_id in sim_seq_list:
        row = linecache.getline(r'./Sports/train.dat', seq_id+1)
        row_items = row.strip().split(',')
        row_items = [int(i) for i in row_items]
        sim_items_list.extend(row_items)

    return sim_items_list


def raw_data_rand_modify(raw_data, item_num, modified_max_seqs_len, max_insert_num, sim_items, sim_seqs_num, plist, epoch):
    """
    item_num: item_num include total item quantity, padding, EOS token, masked token
    max_insert_num: maximum insertion per time ste
    :return:
    random_modified_data:(modified_max_seqs_len)
    l1_ground_truth:(modified_max_seqs_len)
    l2_ground_truth:(modified_max_seqs_len,max_insert_num)
    -----------------------------------------
    tips:
    1.Deleting the original sequence is to train steam to inserting the modified sequence. vice versa
    2.Available_item_pool is used to store available item_id, which can be regarded as item pool. This pool is used
    to select accessible element in the pool during insert operation
    3.The insert operation is equivalent to clicking by mistake, the delete operation is equivalent to an item not exposed to the user
    4.If one sequence only contains one element and for the last item in the sequence, it can't be deleted
    """
    current_seqs_len = len(raw_data)

    random_modified_data = []

    l1_ground_truth = torch.zeros([modified_max_seqs_len], dtype=torch.long)

    l2_ground_truth = torch.zeros([modified_max_seqs_len, max_insert_num], dtype=torch.long)

    available_item_pool = set(i for i in range(1, item_num - 2)) #0 is padding，num-1 is mask，num-2 is eos

    for item_id in raw_data:
        available_item_pool.discard(item_id)

    modified_index = 0  # raw_index represents the index of original sequence, modified_index represents the index of modified sequence

    del_seqs = []

    raw_index = 0

    
    while raw_index < len(raw_data):

        decision = random.random()

        if decision <= plist[0]: # keep

            random_modified_data.append(raw_data[raw_index])
            modified_index += 1
            raw_index += 1

        elif plist[0] < decision <= plist[1] and current_seqs_len != 1 and len(del_seqs) < max_insert_num and raw_index != len(raw_data) - 1:
            del_seqs.insert(0, raw_data[raw_index])
            current_seqs_len -= 1
            raw_index += 1


            decision = random.random()

            # enter into continuous delete mode
            while plist[0] < decision <= plist[1] and current_seqs_len != 1 and len(del_seqs) < max_insert_num and raw_index != len(raw_data) - 1:
                del_seqs.insert(0, raw_data[raw_index])
                current_seqs_len -= 1
                raw_index += 1

                decision = random.random()

            if len(del_seqs) < max_insert_num:
                del_seqs.append(item_num - 2)  # if the length of the deleted sequence is equal to max_insert_num, add EOS

                del_seqs = del_seqs + [0] * (max_insert_num - len(del_seqs))  

            del_seqs = torch.tensor(del_seqs, dtype=torch.long)

            l2_ground_truth[modified_index] = del_seqs

            l1_ground_truth[modified_index] = 2

            random_modified_data.append(raw_data[raw_index])

            modified_index += 1

            raw_index += 1

            del_seqs = []

        elif decision > plist[1] and len(available_item_pool) != 0 and current_seqs_len < modified_max_seqs_len:

            insert_item = random.sample(available_item_pool, 1)

            random_modified_data.append(insert_item[0])

            available_item_pool.remove(insert_item[0])

            l1_ground_truth[modified_index] = 1

            modified_index += 1

            current_seqs_len += 1

    random_modified_data = random_modified_data + [0] * (modified_max_seqs_len - len(random_modified_data))

    random_modified_data = torch.tensor(random_modified_data, dtype=torch.long)

    return random_modified_data, l1_ground_truth, l2_ground_truth

class TrainDataset(Data.Dataset):
    def __init__(self, dir_path, item_num, max_seqs_len, modified_max_seqs_len, max_insert_num, mask_prob, plist):
        super(TrainDataset, self).__init__()
        self.raw_data = []
        self.item_num = item_num
        self.max_seqs_len = max_seqs_len
        self.modified_max_seqs_len = modified_max_seqs_len
        self.max_insert_num = max_insert_num
        self.mask_prob = mask_prob
        self.plist = plist
        self.sim_seqs = np.load(r'./Sports/sim/0.1/sim_sqs_train.npy', allow_pickle='TRUE').item()
        self.epoch = 0

        with open(dir_path, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                seq = line.strip().split(',')
                seq = [int(i) for i in seq]
                self.raw_data.append(seq)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, index):
        '''
        seq=12345->seq=1234500000, padding_mask=1111100000, rec_loss_mask=0100100000, masked_seq=1L34L00000, l2_ground_truth=12345, insert_seqs=S1234
        '''
        seq = self.raw_data[index].copy()
        seq = create_sub_seqs(seq, self.max_seqs_len)

        padding_mask = torch.zeros([self.modified_max_seqs_len], dtype=torch.float)
        padding_mask[:len(seq)] = 1

        sim_seqs_100 = self.sim_seqs[index].copy()
        sim_seqs = random.sample(sim_seqs_100, min(10, len(sim_seqs_100)))

        sim_items = get_sim_items(sim_seqs)

        sim_seqs_num = len(sim_seqs)

        random_modified_seqs, l1_ground_truth, l2_ground_truth = raw_data_rand_modify(seq, self.item_num, self.modified_max_seqs_len, self.max_insert_num, sim_items, sim_seqs_num, self.plist, self.epoch)

        rec_loss_mask = torch.zeros([self.modified_max_seqs_len], dtype=torch.float)
        rec_loss_mask[np.random.rand(self.modified_max_seqs_len) < self.mask_prob] = 1
        rec_loss_mask *= padding_mask

        masked_seq = seq.copy()
        masked_seq = masked_seq + [0] * (self.modified_max_seqs_len - len(masked_seq))
        masked_seq = torch.tensor(masked_seq).long()
        masked_seq[rec_loss_mask == 1] = self.item_num - 1

        seq = seq + [0] * (self.modified_max_seqs_len - len(seq))
        seq = torch.tensor(seq).long()

        insert_seqs = torch.zeros([self.modified_max_seqs_len, self.max_insert_num - 1], dtype=torch.long)
        insert_seqs[:] = l2_ground_truth[:, :-1]

        if sim_seqs_num == 0:
            sim_seqs = [[0] * self.modified_max_seqs_len] * 10
        elif sim_seqs_num < 10 and sim_seqs_num > 0:
            sim_seqs = [self.raw_data[s][-self.modified_max_seqs_len:]+[0]*(self.modified_max_seqs_len-len(self.raw_data[s])) for s in sim_seqs]
            zero_seqs_num = 10 - sim_seqs_num
            sim_seqs_mask = np.array([[0] * self.modified_max_seqs_len] * zero_seqs_num)
            sim_seqs = np.concatenate((sim_seqs, sim_seqs_mask), axis=0)
        else:
            sim_seqs = [self.raw_data[s][-self.modified_max_seqs_len:] + [0] * (self.modified_max_seqs_len - len(self.raw_data[s])) for s in sim_seqs]

        sim_seqs = torch.tensor(sim_seqs).long()

        candicate_items = torch.zeros([self.item_num], dtype=torch.long)

        for item in sim_items:
            num = sim_items.count(item)
            candicate_items[item-1] = num

        zero_mark = torch.zeros([10], dtype=torch.long)
        
        for i in range(0, sim_seqs_num):
            zero_mark[i] = 1

        return seq, masked_seq, random_modified_seqs, l1_ground_truth, l2_ground_truth, insert_seqs, rec_loss_mask, sim_seqs, candicate_items, zero_mark


class ValidDataset(Data.Dataset):
    def __init__(self, valid, valid_neg, item_num, modified_max_seqs_len):
        super(ValidDataset, self).__init__()
        self.valid_data = []
        self.valid_neg_data = []
        self.item_num = item_num
        self.modified_max_seqs_len = modified_max_seqs_len
        self.sim_seqs = np.load(r'./Yelp/sim/0.1/sim_sqs_valid_10.npy', allow_pickle='TRUE').item()
        self.sim_items = np.load(r'./Yelp/sim/0.1/sim_items_valid_10.npy', allow_pickle='TRUE').item()

        with open(valid, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                seq = line.strip().split(',')
                seq = [int(i) for i in seq]
                self.valid_data.append(seq)

        with open(valid_neg, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                seq = line.strip().split(',')
                seq = [int(i) for i in seq]
                self.valid_neg_data.append(seq)

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, index):
        seq = self.valid_data[index].copy()
        masked_seq = seq.copy()

        target = masked_seq[-1]
        target = torch.tensor(target, dtype=torch.long)

        masked_seq[-1] = self.item_num - 1
        masked_seq = masked_seq + [0] * (self.modified_max_seqs_len - len(masked_seq))
        masked_seq = torch.tensor(masked_seq).long()

        leave_one_seq = seq[:-1]
        leave_one_seq = leave_one_seq + [0] * (self.modified_max_seqs_len - len(leave_one_seq))
        leave_one_seq = torch.tensor(leave_one_seq).long()

        seq = seq + [0] * (self.modified_max_seqs_len - len(seq))
        seq = torch.tensor(seq).long()

        valid_neg_seq = self.valid_neg_data[index]
        valid_neg_seq = torch.tensor(valid_neg_seq).long()

        sim_seqs = self.sim_seqs[index].copy()
        sim_seqs_num = len(sim_seqs)

        if sim_seqs_num == 0:
            sim_seqs = [[0] * self.modified_max_seqs_len] * 10
        elif sim_seqs_num < 10 and sim_seqs_num > 0:
            sim_seqs = [self.valid_data[s][-self.modified_max_seqs_len:]+[0]*(self.modified_max_seqs_len-len(self.valid_data[s])) for s in sim_seqs]      # K * 60
            zero_seqs_num = 10 - sim_seqs_num
            sim_seqs_mask = np.array([[0] * self.modified_max_seqs_len] * zero_seqs_num)
            sim_seqs = np.concatenate((sim_seqs, sim_seqs_mask), axis=0)
        else:
            sim_seqs = [self.valid_data[s][-self.modified_max_seqs_len:] + [0] * (self.modified_max_seqs_len - len(self.valid_data[s])) for s in sim_seqs]  # K * 60

        sim_seqs = torch.tensor(sim_seqs).long()

        zero_mark = torch.zeros([10], dtype=torch.long)
        
        for i in range(0, sim_seqs_num):
            zero_mark[i] = 1

        candicate_items = torch.zeros([self.item_num], dtype=torch.long)

        sim_items = self.sim_items[index].copy()
        for item in sim_items:
            num = sim_items.count(item)
            candicate_items[item-1] = num

        return seq, valid_neg_seq, masked_seq, leave_one_seq, target, sim_seqs, candicate_items, zero_mark


class TestDataset(Data.Dataset):
    def __init__(self, test, test_neg, item_num, modified_max_seqs_len):
        super(TestDataset, self).__init__()
        self.test_data = []
        self.test_neg_data = []
        self.item_num = item_num
        self.modified_max_seqs_len = modified_max_seqs_len
        self.sim_seqs = np.load(r'./Beauty/sim/0.1/sim_sqs_test_10.npy', allow_pickle='TRUE').item()
        self.sim_items = np.load(r'./Beauty/sim/0.1/sim_items_test_10.npy', allow_pickle='TRUE').item()

        with open(test, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                seq = line.strip().split(',')
                seq = [int(i) for i in seq]
                self.test_data.append(seq)

        with open(test_neg, 'r', encoding='utf-8') as input_file:
            for line in tqdm(input_file):
                seq = line.strip().split(',')
                seq = [int(i) for i in seq]
                self.test_neg_data.append(seq)

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        seq = self.test_data[index].copy()
        masked_seq = seq.copy()

        target = masked_seq[-1]
        target = torch.tensor(target, dtype=torch.long)

        masked_seq[-1] = self.item_num - 1
        real_len = len(masked_seq) - 1
        masked_seq = masked_seq + [0] * (self.modified_max_seqs_len - len(masked_seq))
        masked_seq = torch.tensor(masked_seq).long()

        leave_one_seq = seq[:-1]
        leave_one_seq = leave_one_seq + [0] * (self.modified_max_seqs_len - len(leave_one_seq))
        leave_one_seq = torch.tensor(leave_one_seq).long()


        seq = seq + [0] * (self.modified_max_seqs_len - len(seq))
        seq = torch.tensor(seq).long()

        test_neg_seq = self.test_neg_data[index]
        neg_num = len(test_neg_seq)
        neg_num = torch.tensor(neg_num).long()
        test_neg_seq = test_neg_seq + [-1] * (self.item_num - neg_num)
        test_neg_seq = torch.tensor(test_neg_seq).long()

        sim_seqs = self.sim_seqs[index].copy()
        sim_seqs_num = len(sim_seqs)

        if sim_seqs_num == 0:
            sim_seqs = [[0] * self.modified_max_seqs_len] * 10
        elif sim_seqs_num > 0 and sim_seqs_num < 10:
            sim_seqs = [self.test_data[s][-self.modified_max_seqs_len:] + [0] * (self.modified_max_seqs_len - len(self.test_data[s])) for s in sim_seqs]
            zero_seqs_num = 10 - sim_seqs_num
            sim_seqs_mask = np.array([[0] * self.modified_max_seqs_len] * zero_seqs_num)
            sim_seqs = np.concatenate((sim_seqs, sim_seqs_mask), axis=0)
        else:
            sim_seqs = [self.test_data[s][-self.modified_max_seqs_len:] + [0] * (self.modified_max_seqs_len - len(self.test_data[s])) for s in sim_seqs]

        sim_seqs = torch.tensor(sim_seqs).long()

        zero_mark = torch.zeros([10], dtype=torch.long)
        
        for i in range(0, sim_seqs_num):
            zero_mark[i] = 1

        candicate_items = torch.zeros([self.item_num], dtype=torch.long)
        
        sim_items = self.sim_items[index].copy()
        for item in sim_items:
            num = sim_items.count(item)
            candicate_items[item-1] = num

        return seq, test_neg_seq, neg_num, masked_seq, leave_one_seq, target, sim_seqs, candicate_items, zero_mark, real_len