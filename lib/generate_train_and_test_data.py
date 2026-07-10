"""
generate train and test data files. DONE (代码质量差, 不知所云)

输入原始数据 -> 中间结构 -> 训练/测试样本文件

原始 gowalla.txt
-> cut()
-> _read()
-> _gen_user_his_behave()
-> _gen_train_sample() / _gen_test_sample()
-> _split_train_sample()
-> 产出 train_instances_* / test_instances / validation_instances
"""

from __future__ import print_function

from typing import Tuple

import os
import time
import random
import multiprocessing as mp
import numpy as np

def cut(input_file, train_file, test_file, val_file, item_vec, number) -> Tuple[dict[int, int], dict[int, int]]:
    """cut the original data file into train, test, validation data set and write them into the corresponding files

    :param input_file: the file name of orignal data file
    :param train_file: the file path to write in used to trian
    :param test_file: the file path to write in used to test
    :param val_file: the file path to write in used to validate
    :param item_vec: the file path to write in used to store item vector
    :param number: the user number to testest, 测试用户数
    :return: (user_ids, item_ids),id 编码映射:
        - user_ids: 原始 user id -> 连续编号
        - item_ids: 原始 item id -> 连续编号
    """
    # user_behavior[user_id] = [line1, line2, ...] 收集每个用户全部行为
    user_behavior = dict()
    user_ids = list()
    item_ids = set()
    with open(input_file,'r') as f:
        for line in f:
            arr = line.split(',')
            if len(arr) != 5:
                break
            if int(arr[1]) not in item_ids:
                item_ids.add(int(arr[1]))
            if int(arr[0]) not in user_behavior:
                user_ids.append(int(arr[0]))
                user_behavior[int(arr[0])] = list()
            user_behavior[int(arr[0])].append(line)

    # - train/test/validation 的用户集合不重叠
    # - 同一个用户不会同时出现在 train 和 test 中
    random.shuffle(user_ids)
    test_user_ids = user_ids[:number]
    validation_user_ids=user_ids[number:number*2]
    train_user_ids = user_ids[number*2:]

    # write train data set
    with open(train_file, 'w') as f:
        for uid in train_user_ids:
            for line in user_behavior[uid]:
                f.write(line)
    
    # write test data set
    with open(test_file, 'w') as f:
        for uid in test_user_ids:
            for line in user_behavior[uid]:
                f.write(line)
    
    # write validation data set
    with open(val_file, 'w') as f:
        for uid in validation_user_ids:
            for line in user_behavior[uid]:
                f.write(line)
    
    # re label usr id and item id which make the initial id is 0
    user_ids: dict[int, int] = {id : i for i, id in enumerate(user_behavior)}
    item_ids: dict[int, int] = {id : i for i, id in enumerate(item_ids)}
    print('user number {}, item number {}'.format(len(user_ids), len(item_ids)))

    # train_user_ids={id:i for i,id in enumerate(train_user_ids)}
    # train_item_his = dict()
    # with open(_input,'r') as f:
    #     for line in f:
    #         arr = line.split(',')
    #         if len(arr) != 5:
    #             break   
    #         item = item_ids[int(arr[1])]
    #         if item not in train_item_his:
    #             train_item_his[item] = list()
    #         if int(arr[0]) not in test_user_ids and int(arr[0]) not in validation_user_ids:
    #             #print(1) 
    #             train_item_his[item].append(train_user_ids[int(arr[0])])
    # train_item_vec = np.zeros([len(item_ids), len(train_user_ids)])
    # for i in range(len(item_ids)):
    #     item_his = train_item_his[i]
    #     train_item_vec[i][item_his] = 1.0
    
    # np.save(_item_vec, train_item_vec)
    return user_ids, item_ids


def _read(raw_file, train_data_file, test_data_file, val_data_file, test_record_num):
    """读取原始数据文件, 并返回用户ID和物品ID的映射
    把原始 csv 读成内存中的 sample dict
    """
    train_data_file = 'train.dat'
    val_data_file = 'validation.dat'
    test_data_file = 'test.dat'

    path_seg = raw_file.split('/')
    prefix = ''
    if len(path_seg) > 1:
        for seg in path_seg[:-1]:
            prefix += seg + '/'
    
    user_id_map, item_id_map = cut(
        raw_file, 
        prefix + 'raw_' + train_data_file,
        prefix + 'raw_' + test_data_file,
        prefix + 'raw_' + val_data_file,
        prefix + 'train_item_vec',
        test_record_num
    )

    behavior_dict = dict()  # record the behavior type, 5 types, key is the type, value is the id of the type
    train_sample = dict()  # record user id, item id, cate_id, behavior_id, timestampe. key the liks 'USERID', value is a array.
    test_sample = dict()
    val_sample = dict()
    
    user_id = list()
    item_id = list()
    cat_id = list()
    behav_id = list()
    timestamp = list()

    start = time.time()
    itobj = zip(
        [
            prefix + 'raw_' + train_data_file, 
            prefix + 'raw_' + test_data_file, 
            prefix + 'raw_' + val_data_file
        ],
        [train_sample, test_sample,val_sample]
    )

    for filename, sample in itobj:
        with open(filename, 'r') as f:
            for line in f:
                arr = line.split(',')
                if len(arr) != 5:
                    break
                # raw id -> relabel id
                user_id.append(user_id_map[int(arr[0])])
                item_id.append(item_id_map[int(arr[1])])
                cat_id.append(int(float(arr[2])))
                if arr[3] not in behavior_dict:
                    i = len(behavior_dict)
                    behavior_dict[arr[3]] = i
                behav_id.append(behavior_dict[arr[3]])
                timestamp.append(int(arr[4]))
            sample["USERID"] = np.array(user_id)
            sample["ITEMID"] = np.array(item_id)
            sample["CATID"] = np.array(cat_id)
            sample["BEHAV"] = np.array(behav_id)
            sample["TS"] = np.array(timestamp)

            user_id = []
            item_id = []
            cat_id = []
            behav_id = []
            timestamp = []

    #write train data set
    '''
    with open(prefix+'processed_'+train_data_file, 'w') as f:
        for user_id,item_id,cat_id,behav_id,timestamp in zip(train_sample["USERID"],train_sample["ITEMID"],
                                                             train_sample["CATID"],train_sample["BEHAV"],train_sample["TS"]):

            f.write(str(user_id)+','+str(item_id)+','+str(cat_id)+','+str(behav_id)+','+str(timestamp)+'\n')

    with open(prefix+'processed_'+test_data_file, 'w') as f:
        for user_id,item_id,cat_id,behav_id,timestamp in zip(test_sample["USERID"],test_sample["ITEMID"],
                                                             test_sample["CATID"],test_sample["BEHAV"],test_sample["TS"]):

            f.write(str(user_id)+','+str(item_id)+','+str(cat_id)+','+str(behav_id)+','+str(timestamp)+'\n')

    with open(prefix+'processed_'+validation_data_file, 'w') as f:
        for user_id,item_id,cat_id,behav_id,timestamp in zip(validation_sample["USERID"],validation_sample["ITEMID"],
                                                             validation_sample["CATID"],validation_sample["BEHAV"],validation_sample["TS"]):

            f.write(str(user_id)+','+str(item_id)+','+str(cat_id)+','+str(behav_id)+','+str(timestamp)+'\n')
    '''
    print("Read data done, {} train records, {} test records"", elapsed: {}".format(len(train_sample["USERID"]), len(test_sample["USERID"]), time.time() - start))
    os.remove(prefix + 'raw_' + train_data_file)
    os.remove(prefix + 'raw_' + test_data_file)
    os.remove(prefix + 'raw_' + val_data_file)
    # train_sample record user id, item id, cate_id, behavior_id, timestampe. key the liks 'USERID', value is a array.
    # test_sample is like train_sample
    # behavior_dict record the behavior type, 5 types, key is the type, value is the id of the type
    return behavior_dict, train_sample, test_sample, val_sample, len(user_id_map), len(item_id_map)


def _gen_user_history_behavior(train_sample: dict) -> dict[str, list[tuple[str, float]]]:
    """
    Generate user history behavior, key is userid and value is a list [(itemId, timestamp)], list is ascendent by timestamp
    """
    user_his_behav = dict()
    iterobj = zip(train_sample["USERID"], train_sample["ITEMID"], train_sample["TS"])
    for user_id, item_id, ts in iterobj:
        if user_id not in user_his_behav:
            user_his_behav[user_id] = list()
        user_his_behav[user_id].append((item_id, ts))
    
    for _, value in user_his_behav.items():
        value.sort(key=lambda x: x[1])
    return user_his_behav


def _partial_gen_train_sample(
        users,
        user_his_behav, 
        filename,
        seq_len,
        min_seq_len, 
        pipe
    ):
    #filename is the file to be written into, i.e. sample file
    stat = dict() # record the frequency of the each item 's appearance
    count = 0
    with open(filename, 'w') as f:
        for user in users:
            value = user_his_behav[user]    # the clicked item id for user
            count = len(value)              # the item number of the user to click
            if count < min_seq_len:
                continue
            arr = [-1 for i in range(seq_len - min_seq_len)] + [v[0] for v in value]
            
            # 滑动窗口
            for i in range(len(arr) - seq_len + 1):
                sample = arr[i: i + seq_len]
                f.write('{}|'.format(user))                                         # sample id
                f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))      # item ids
                f.write("{}".format(sample[-1]))                                    # label, no ts
                f.write('\n')
                count += 1
                if sample[-1] not in stat:
                    stat[sample[-1]] = 0
                stat[sample[-1]] += 1
    pipe.send((stat, count))


def _gen_train_sample(
        train_sample,
        train_instances_file,
        test_sample=None,
        val_sample=None,
        train_sample_seg_cnt=400,
        parall=5,
        seq_len=70,
        min_seq_len=6
    ):
    user_his_behav = _gen_user_history_behavior(train_sample)#
    print("user_his_behav len: {}".format(len(user_his_behav)))

    users = list(user_his_behav.keys())
    process = []
    pipes = []
    job_size = int(len(user_his_behav) / parall)

    if len(user_his_behav) % parall != 0:
        parall += 1

    for i in range(parall):
        a, b = mp.Pipe()
        pipes.append(a)
        p = mp.Process(
            target=_partial_gen_train_sample,
            args=(
                users[i * job_size: (i + 1) * job_size], 
                user_his_behav,
                '{}.part_{}'.format(train_instances_file, i),
                seq_len,
                min_seq_len,
                b
            )
        )
        process.append(p)
        p.start()

    stat = dict()  # record the frequency of the each item 's appearance
    t = 0
    for pipe in pipes:
        (st, count) = pipe.recv()
        t += count
        for k, v in st.items():
            if k not in stat:
                stat[k] = 0
            stat[k] += v

    for p in process:
        p.join()
    print('total instances is {}'.format(t))
    
    # Merge partial files
    with open(train_instances_file, 'w') as f:
        for i in range(parall):
            filename = '{}.part_{}'.format(train_instances_file, i)
            with open(filename, 'r') as f1:
                f.write(f1.read())
            os.remove(filename)

        if test_sample is not None:
            user_his_behav = _gen_user_history_behavior(test_sample)
            for user, value in user_his_behav.items():
                if len(value)/2 + 1 < min_seq_len:
                    continue
                mid = int(len(value) / 2 + 1)
                left = value[:mid]      #[-seq_len + 1:]
                arr = [-1 for i in range(seq_len - min_seq_len)] + [v[0] for v in left]
                for i in range(len(arr) - seq_len + 1):
                    sample = arr[i: i + seq_len]
                    f.write('{}|'.format(user))
                    f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                    f.write("{}".format(sample[-1]))
                    f.write('\n')
        
        if val_sample is not None:
            user_his_behav = _gen_user_history_behavior(val_sample)
            for user, value in user_his_behav.items():
                if len(value) / 2 + 1 < min_seq_len:
                    continue
                mid = int(len(value) / 2 + 1)
                left = value[:mid]  # [-seq_len + 1:]
                arr = [-1 for i in range(seq_len - min_seq_len)] + [v[0] for v in left]
                for i in range(len(arr) - seq_len + 1):
                    sample = arr[i: i + seq_len]
                    f.write('{}|'.format(user))
                    f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                    f.write("{}".format(sample[-1]))
                    f.write('\n')

    # Split train sample to segments
    _split_train_sample(train_instances_file, train_sample_seg_cnt)
    return stat


def _split_train_sample(train_instances_file, train_sample_seg_cnt=400) -> None:
    """Split train sample to segments
    :param train_instances_file: train instances file
    :param train_sample_seg_cnt: train sample seg cnt
    :return: None
    """
    segment_filenames = []
    segment_files = []
    for i in range(train_sample_seg_cnt):
        filename = "{}_{}".format(train_instances_file, i)
        segment_filenames.append(filename)
        segment_files.append(open(filename, 'w'))

    with open(train_instances_file, 'r') as fi:
        for line in fi:
            i = random.randint(0, train_sample_seg_cnt - 1) # train_sample_seg_cnt is 400
            segment_files[i].write(line)

    for f in segment_files:
        f.close()

    # Remove the original file
    os.remove(train_instances_file)

    # Shuffle
    num = 0
    for fn in segment_filenames:
        lines = []
        with open(fn, 'r') as f:
            for line in f:
                lines.append(line)
        random.shuffle(lines)
        num += len(lines)
        with open(fn, 'w') as f:
            for line in lines:
                f.write(line)
    print('number of training instance is {}'.format(num))


def _gen_test_sample(
        test_sample,
        test_instances_file,
        seq_len=70,
        min_seq_len=6
    ):
    # user_his_behav is a dict, key is userid and value is a list [(itemId, timestamp)], list is ascendent by timestamp
    user_history_behavior: dict[str, list[tuple[str, float]]] = _gen_user_history_behavior(test_sample)
    with open(test_instances_file, 'w') as f:
        for user, value in user_history_behavior.items():
            if len(value) / 2 + 1 < min_seq_len:
                continue

            mid = int(len(value) / 2 + 1)
            left = value[:mid][-seq_len + 1:]
            left = [-1 for i in range(seq_len - len(left) - 1)] + [l[0] for l in left]
            right = value[mid:]
            f.write('{}|'.format(user))  # sample id
            f.write("{}|".format(",".join([str(v) for v in left])))
            f.write('{}'.format(','.join([item[0] for item in right])))  # test_unit_id is ‘test_unit_id’
            f.write('\n')


if __name__=="__main__":
    behavior_dict, train_sample, test_sample=_read('data/mock/mock.dat', 'train.dat', 'test.dat', 20)#20 is the test users
    # write the training instance into different train_sample_seg_cnt files， avoid that a file is too large
    #stat record the click frequency of each item
    #seq_len=20 min that 19 behaviors and one label
    stat = _gen_train_sample(train_sample,'./data/train_instances', train_sample_seg_cnt=10, parall=3,seq_len=20, min_seq_len=3)
    _gen_test_sample(test_sample,'./data/test_instances', seq_len=20, min_seq_len=3)
    _init_tree(train_sample, test_sample, stat, kv_file=None)
