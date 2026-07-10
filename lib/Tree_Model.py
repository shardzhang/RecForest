"""
Tree Model

负责 item/path 双向映射
"""

import torch
import math
import numpy as np
from lib.KmeansTree import ConstructKmeansTree

class Tree:
    def __init__(
        self,
        data=None,
        max_iters=100,
        feature_ratio=0.8,
        item_num=1000, 
        k=2,
        init_way='random',
        construct=True,
        parall=4,
        device='cuda'
    ):
        self.item_num = item_num    # 物品总个数
        self.k = k                  # 树的分叉树
        self.device = device        # 设备
        if construct:               # 当前是否构建树?
            self.build_tree(data=data, max_iters=max_iters, init_way=init_way, feature_ratio=feature_ratio, parall=parall)

    # fixme
    def build_tree(
            self, 
            data=None,
            max_iters=100,
            init_way='random',
            feature_ratio=0.8,
            parall=4
        ):
        """Tree 类里真正构造树映射的方法. 职责是生成两张核心表: 
        item_to_code: item_id -> path
        code_to_item: leaf_code -> item_id

        :param data: item embeding or matrix used for kmeans cluster, [item_num, emb_size]

        """
        self.tree_height = math.ceil(math.log(self.item_num, self.k))               # 树高
        num_all_leaf_code = self.k ** self.tree_height                              # the number of leaf node
        self.code_to_item = torch.zeros((num_all_leaf_code,), dtype=torch.int64)    # node code to node
        self.item_to_code = {item_id : [] for item_id in range(self.item_num)}      # record the path of item, one item can have multiple paths

        """
        情况 A: init_way == 'random'
        它随机给每个 item 分一个路径。
        逻辑上相当于: 
            打乱 item 顺序
            依次把 item 填到树叶上
            同时生成: 
                该 item 的 path
                该 leaf 对应的 item
        这是最简单的树构造方式。
        """
        if init_way.lower() == 'random':
            num_k_code_item = (num_all_leaf_code - self.item_num) // (self.k - 1)
            item_seq = np.arange(self.item_num)
            np.random.shuffle(item_seq)
            
            start_code = 0
            for i, item_id in enumerate(item_seq):
                path = [start_code % (self.k ** (j + 1)) // (self.k ** j) for j in range(self.tree_height-1,-1,-1)]
                if i < num_k_code_item:
                    for j in range(self.k):
                        path[-1] = j
                        self.item_to_code[item_id].append(torch.LongTensor(path))
                    self.code_to_item[start_code:start_code+self.k] = item_id
                    start_code = start_code + self.k
                else:
                    self.item_to_code[item_id].append(torch.LongTensor(path))
                    self.code_to_item[start_code:start_code+1] = item_id
                    start_code = start_code + 1
            
            item_id = item_seq[-1]
            while start_code < num_all_leaf_code:
                path = [start_code % (self.k ** (j+1)) // (self.k ** j) for j in range(self.tree_height-1,-1,-1)]
                self.item_to_code[item_id].append(torch.LongTensor(path))
                self.code_to_item[start_code]=item_id
                start_code = start_code + 1
        
        elif init_way.lower() == 'datakm' or init_way.lower() == 'embkm':
            """
            情况 B: init_way == 'datakm' 或 'embkm'
            它用数据 / embedding 做 KMeans 风格聚类建树
            核心步骤: 
            1. 用 ConstructKmeansTree 递归聚类
            2. 让相似 item 更可能落在相近路径上
            3. 再把聚类结果转成: item_to_code
            """
            print('start to construct')
            print(f"debug. data.shape: {data.shape}") # [item_num, emb_size]
            assert data is not None
            assert data.shape[0] == self.item_num

            constructer = ConstructKmeansTree(parall=parall)
            
            # 
            if data.shape[0] < num_all_leaf_code:
                # [num_all_leaf_code, ]
                index = torch.cat([
                        torch.arange(self.item_num),
                        torch.randint(
                            low=0,
                            high=self.item_num,
                            size=(num_all_leaf_code - self.item_num,)
                        )
                    ],
                    dim=0
                )

                # TODO: ?
                index[:] = index[torch.randperm(index.nelement())]
                assert len(index) == num_all_leaf_code
                
                print('start to construct')
                item_ids, leaf_node_codes = constructer.train(
                    data[index], 
                    self.k, 
                    max_iters=max_iters, 
                    feature_ratio=feature_ratio
                )
                start_code, end_code = leaf_node_codes.min().item(), leaf_node_codes.max().item()
                assert start_code == (self.k**self.tree_height - 1) / (self.k - 1) and end_code == (self.k**(self.tree_height + 1) - 1) / (self.k - 1) - 1
                
                print(len(index), len(item_ids), len(leaf_node_codes))
                for i, j, code in zip(range(len(index)), item_ids, leaf_node_codes):
                    assert i == j.item()
                    real_item_id = index[i].item()                    
                    reverse_path, tc = [], code.item()
                    assert tc >= start_code and tc <= end_code
                    for _ in range(self.tree_height):
                        reverse_path.append((tc - 1) % self.k)
                        tc = (tc - 1) // self.k
                    self.item_to_code[real_item_id].append(torch.LongTensor(reverse_path[::-1]))
                    self.code_to_item[code.item() - start_code] = real_item_id
        
        self.card = torch.zeros(self.tree_height)
        for i in range(self.tree_height):
            self.card[i] = self.k ** (self.tree_height - i - 1)
        self.card = self.card.to(self.device)
        self.code_to_item = self.code_to_item.to(self.device)


    def read_tree(self, code_to_item_file, item_to_code_file, k=4):
        """read tree from file
        :param code_to_item_file: the file path of code_to_item
        :param item_to_code_file: the file path of item_to_code
        :param k: the branch number of each tree
        :return: self
        """
        # =========== code_to_item ==========
        # 32488
        # 33907
        # 38082
        # 46357
        # 7218
        # 叶子节点编号 -> item id
        # (104976,)
        self.code_to_item = torch.tensor(np.load(code_to_item_file)).to(self.device)
        print(f"debug. self.code_to_item.shape: {self.code_to_item.shape}")
        self.item_num = self.code_to_item.max().item() + 1
        
        #========= item_to_code ============
        # [6, 14, 12, 12]
        # [11, 17, 11, 5]
        # [11, 16, 3, 8]
        # [0, 4, 17, 17]
        # [17, 12, 6, 9]
        # item id -> path
        # (71436, 4)
        item_to_code_mat = torch.tensor(np.load(item_to_code_file)).long()
        assert self.item_num == item_to_code_mat.shape[0]
        self.tree_height = item_to_code_mat.shape[-1]
        self.item_to_code = item_to_code_mat.to(self.device)
        print(f"debug. self.item_to_code.shape: {self.item_to_code.shape}")
        self.k = k

        # 把树路径path当成k进制数, self.card[i]表示第i位上的位权
        # 用途: 把path token序列压成一个唯一leaf code, 用于从code_to_item反查 item
        self.card = torch.zeros(self.tree_height).to(self.device)
        for i in range(self.tree_height):
            self.card[i] = self.k ** (self.tree_height - i - 1)
        return self


    def label_to_path(self, batch_y):
        """given the item id, obtain the path sequence
        :param batch_y: [batch_size, 1], it contains item ids
        :return: [batch_size, 1, tree_height]
        """
        print(self.item_to_code[batch_y].shape)
        if isinstance(self.item_to_code, dict):
            labels = batch_y.view(-1).cpu().tolist()
            return torch.stack([self.item_to_code[label][0] for label in labels], dim=0).to(self.device)
        return self.item_to_code[batch_y.to(self.device)]

    
    def path_to_label(self, batch_pred_seq):
        """translate the path sequence to item
        1. 路径每一位乘上对应权重
        2. 全部加起来, 得到唯一整数 code
        3. 用这个 code 去查 code_to_item[code] -> item_id

        :param batch_pred_seq: [batch_size, topk, tree_height]
        :return: [batch_size, topk, ]
        """
        print(self.card)
        code = ((batch_pred_seq * self.card).sum(-1)).long()
        return self.code_to_item[code]
