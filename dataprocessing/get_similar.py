import numpy as np
import time
import linecache

def jaccard_sim(list1, list2):
    A = len(list(set(list1).intersection(set(list2))))
    B = len(list(set(list1).union(set(list2))))
    jac_sim = A / B
    return jac_sim

def sort_dict_get_key(dict, k):
    final_results = []
    sorted_dict = sorted([(k,v) for k,v in dict.items()], reverse=True)
    tmp_set = set()
    for item in sorted_dict:
        tmp_set.add(item[1])
    for list_ltem in sorted(tmp_set, reverse=True)[:k]:
        for dic_item in sorted_dict:
            if dic_item[1] == list_ltem:
                final_results.append(dic_item[0])
    final_results = final_results[0:k]
    return final_results


def get_sim_seqs(input_file1, input_file2):
    with open(input_file1) as f1:
        sim_dict = {}
        sim_dict_small ={}
        all_sim_list = []
        num1 = 0
        start = time.time()
        for line1 in f1.readlines():
            sim_dict.setdefault(num1, [])
            sim_dict_small.setdefault(num1, [])
            raw_data = line1.strip().split(',')
            raw_data = [int(i) for i in raw_data]
            num2 = 0

            sim_sq_dict = {}

            with open(input_file2) as f2:
                for line2 in f2.readlines():
                    if line2 == line1:
                        sim = 0.0

                    elif line2 != line1:
                        data = line2.strip().split(',')
                        data = [int(i) for i in data]
                        sim = jaccard_sim(raw_data, data)

                    if sim >= 0.1:
                        sim_sq_dict[num2] = sim
                        all_sim_list.append(sim)
                    num2 += 1
                key_list = sort_dict_get_key(sim_sq_dict, 60)
                sim_dict[num1] = key_list
                sim_dict_small[num1] = key_list[0:10]
                f2.close()
            num1 += 1
            if num1 % 100 == 0:
                now = time.time()
                print('done:{},time {}'.format(num1, now-start))
        f1.close()
    return sim_dict, all_sim_list, sim_dict_small


def get_sim_items(file1, file2):
    sim_sq_dict = np.load(file1, allow_pickle='TRUE').item()
    sim_items_dict = {}
    sim_items_dict_set = {}

    start = time.time()
    for target_sq in sim_sq_dict.keys():
        sim_items_dict.setdefault(int(target_sq), [])
        sim_items_dict_set.setdefault(int(target_sq), [])
        sq_list = list(sim_sq_dict[target_sq])

        item_lists = []

        for row_id in sq_list:
            row = linecache.getline(file2, row_id+1)
            row_items = row.strip().split(',')
            row_items = [int(i) for i in row_items]

            item_lists.extend(row_items)

        sim_items_dict[target_sq] = item_lists

        item_lists_set = list(set(item_lists))
        sim_items_dict_set[target_sq] = item_lists_set

        if target_sq % 1000 == 0:
            now = time.time()
            print('done:{},time {}'.format(target_sq, now-start))

    return sim_items_dict

































