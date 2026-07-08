"""
DIN Trainer. DONE
"""

import torch
import numpy as np
from lib.DIN_Model import DeepInterestNetwork


class DINTrain:
    """
    方法名   前向次数	                            峰值显存	                     GPU 利用率	   适用场景
    方法 1	 batch_size 次	                        ~7.05 GB	                    中	         CPU / batch ≤ 1
    方法 2	 1 次	                                ~7.05 GB × batch_size	        最高	     仅 batch ≤ 1
    方法 3	 item_num / item_bs 次	                batch × item_bs × 69 × 384 × 4	高	         ✅ 通用 GPU 方案    
    方法 4	 batch_size × item_num / item_bs 次	    ~0.2 GB	                        低	         不推荐
    """
    def __init__(self,
        item_num: int = 100,
        sample_negative_num: int = 60,
        emb_dim: int = 96,
        device: str = 'cpu',
        feature_groups: list[int] = [20,20,10,10,2,2,2,1,1,1],
        sum_pooling: bool = False
    ):
        """初始化DIN Trainer
        :param item_num: item number
        :param sample_negative_num: negative sample number
        :param emb_dim: embedding dimension
        :param device: device
        :param feature_groups: feature groups
        :param sum_pooling: sum pooling
        :param optimizer: optimizer
        """
        self.item_num = item_num
        self.device = device
        self.N = sample_negative_num

        self.DINModel = DeepInterestNetwork(
            item_num=item_num, 
            embedding_dim=emb_dim,
            feature_groups=feature_groups, 
            sum_pooling=sum_pooling
        ).to(self.device)

        # optimizer
        params = [
            {"params": self.DINModel.item_embedding.parameters(), "weight_decay": 1e-5},
            {"params": [p for n,p in self.DINModel.named_parameters() if "item_embedding" not in n], "weight_decay": 0.0}
        ]
        self.optimizer = torch.optim.Adam(params, lr=1e-3)
        
        # the number of batches that have been trained, which is used to calculate the learning rate
        self.batch_num = 0


    def update_learning_rate(
        self, 
        iter: int, 
        learning_rate_base: float = 1e-3, 
        warmup_steps: int = 5000,
        decay_rate: float = 1./3, 
        learning_rate_min: float = 1e-6
    ):
        """ Learning rate with linear warmup and exponential decay.

        :param t: the number of batches that have been trained
        :param learning_rate_base: the base learning rate after warmup
        :param warmup_steps: the number of steps for linear warmup
        :param decay_rate: the decay rate for exponential decay
        :param learning_rate_min: the minimum learning rate after decay
        :return: the learning rate for the current batch
        """
        lr = learning_rate_base * np.minimum(
            (iter + 1.0) / warmup_steps,
            np.exp(decay_rate * ((warmup_steps - iter - 1.0) / warmup_steps)),
        )
        lr = np.maximum(lr, learning_rate_min)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


    def uniform_sampled_softmax(self, batch_users: torch.Tensor, batch_labels: torch.Tensor, N: int) -> torch.Tensor:
        """sampled softmax loss
        TODO: 是否可以直接调用tf.nn.sampled_softmax_loss来计算损失?

        :param: batch_users: (batch_size, seq_len)
        :param: batch_labels: (batch_size, 1)
        :param: N: negative sample number
        :return: loss. scalar
        """
        batch_size = batch_users.shape[0]        

        # generate samples, the first column is positive labels, and the rest are negative labels
        # (batch_size, N + 1)
        labels = torch.full((batch_size, N + 1), -1, device=self.device, dtype=torch.int64)
        labels[:, 0:1] = batch_labels      # positive labels
        labels[:, 1:] = torch.randint(     # random negative labels
            low=0, 
            high=self.item_num, 
            size=(batch_size, N), 
            device=self.device,
        )

        # the effective index of samples, which is the index of items that are not filled with the absence (item_num)
        # (batch_size, N + 1)
        effective_index = torch.full(labels.shape, True, device=self.device, dtype=torch.bool)
        # the first column of samples is positive labels, so the effective index of the first column is always True
        # the effective index of the rest columns is True if the negative label is not equal to the positive label, otherwise False
        effective_index[:, 1:] = labels[:, 0:1] != labels[:, 1:]

        # compute the log of sampling probability, which is used to correct the bias caused by negative sampling
        # (batch_size, N + 1)
        log_q_matrix = torch.full(labels.shape, 0.0, device=self.device, dtype=torch.float32)

        # 负采样概率: q = 每个用户的有效负样本数 / (总物品数 - 1)
        # 每个用户的有效的所有负样本共享同一个log(q)
        log_q_matrix[:, 1:][effective_index[:, 1:]] = torch.log(
            effective_index[:, 1:].sum(-1).view(batch_size, 1) * 1.0 / (self.item_num - 1)
        ).expand(batch_size, N)[effective_index[:, 1:]]

        # (batch_size, N + 1)
        user_index = torch.arange(batch_size, device=self.device).view(-1, 1).expand(labels.shape)
        # (batch_size * effective_index_len, )
        # 用和原张量同形状的布尔张量做索引，会把原张量中所有True位置的元素展平成一维张量
        user_index = user_index[effective_index]

        # (batch_size * effective_index_len, )
        labels = labels[effective_index]
        # (batch_size * effective_index_len, 1)
        labels = labels.view(-1, 1)

        # (batch_size * effective_index_len, seq_len)
        # 把每个用户的特征重复取出effective_index_len次, 拼成一个更长的矩阵, 和samples中的标签一一对应
        batch_users = batch_users[user_index]

        # (batch_size, N + 1)
        # the preference score of the positive and negative samples, which is computed by DINModel, and the bias caused by negative sampling is corrected with log_q_matrix
        o_pi = torch.full(log_q_matrix.shape, -1.0e9, device=self.device, dtype=torch.float32)

        # (batch_size * effective_index_len, )
        logits = self.DINModel(batch_user=batch_users, batch_label=labels)[:, 0]
        # 布尔掩码索引赋值: 只操作两个张量中掩码为True的位置
        o_pi[effective_index] = logits - log_q_matrix[effective_index]

        # compute the sampled softmax loss with log-sum-exp trick for numerical stability
        # (batch_size, ) -> scalar
        return (-o_pi[:, 0] + torch.logsumexp(o_pi, dim=1)).mean(-1)


    def update_DIN(self, batch_users, batch_labels):
        """ update the parameters of DINModel with sampled softmax loss
        :param batch_users: (batch_size, seq_len)
        :param batch_labels: (batch_size, 1)
        :return: loss. scalar
        """
        self.batch_num += 1 # global_step
        loss = self.uniform_sampled_softmax(
            batch_users.to(self.device), 
            batch_labels.to(self.device).view(-1, 1), 
            self.N
        )
        self.optimizer.zero_grad()  # clean the gradient
        loss.backward()             # compute the gradient
        self.optimizer.step()       # update the parameters
        self.update_learning_rate(self.batch_num)
        return loss


    def calculate_preference(self, batch_user: torch.Tensor, batch_items: torch.Tensor) -> torch.Tensor:
        """calculate the preference score of batch_user to batch_items with DINModel
        
        :param batch_user: (batch_size, seq_len)
        :param batch_items: (batch_size, 1)
        :return: (batch_size, 1)
        """
        return self.DINModel(batch_user, batch_items)


    @torch.no_grad()
    def predict_for_user(self, test_instances: torch.Tensor, topk: int = 10) -> torch.Tensor:
        """ predict the top-k items with the highest preference score for each user in test_instances
        
        逐用户循环，对所有 item 一次前向。最直观，适合 CPU 或 batch_size 很小的场景。

        什么时候用?
        - CPU 上 (无 kernel launch 开销差异)
        - batch_size 较大(≥4)且显存有限
        - 每次 for i in range(batch_size) repeat 后 user_seq 被及时回收，峰值只有一份 [item_num, seq_len]

        :param test_instances: (batch_size, seq_len)
        :param topk: the number of items to recommend for each user
        :return: (batch_size, item_num) the preference scores for each user
        """
        self.DINModel.eval()
        
        # (batch_size, seq_len)
        test_instances: torch.Tensor = test_instances.to(self.device)
        batch_size: int = test_instances.shape[0]
        
        # 1. 构建一个包含所有物品ID的张量
        # (item_num, )
        all_items: torch.Tensor = torch.arange(self.item_num, device=self.device)
        
        # 2. 计算每个用户对所有物品的偏好得分
        # 暴力全量排序 Brute-force Top-K. 推荐系统里最标准、最准确的评估方式
        scores = []
        for i in range(batch_size):
            # (1, seq_len) -> (item_num, seq_len)
            user_seq: torch.Tensor = test_instances[i:i+1].expand(all_items.shape[0], -1)
            # (item_num, 1) -> (item_num, )
            score: torch.Tensor = self.calculate_preference(user_seq, all_items.view(-1, 1)).view(-1)
            scores.append(score)
        
        # (batch_size, item_num)
        scores = torch.stack(scores)

        self.DINModel.train()
        return scores


    @torch.no_grad()
    def predict_merge_all(self, test_instances: torch.Tensor, topk: int = 10) -> torch.Tensor:
        """predict the top-k items with the highest preference score for each user in test_instances
        
        全量合并: 
        一次大矩阵前向打满 GPU，显存需求极高
        一次前向计算全部 batch_size x item_num 对。GPU 利用率最高，但 71436 x 69 x 384 x 4 ≈ 7.05 GB 起步，batch_size > 1 即爆 8 GB 显存，实用性最低

        什么时候用?
        - GPU
        - batch_size x item_num x seq_len x 字节 < 可用显存 x 安全系数

        :param test_instances: (batch_size, seq_len)
        :param topk: the number of items to recommend for each user
        :return: (batch_size, item_num) the preference scores for each user
        """
        self.DINModel.eval()

        # (batch_size, seq_len)
        test_instances: torch.Tensor = test_instances.to(self.device)
        batch_size: int = test_instances.shape[0]
        seq_len: int = test_instances.shape[1]
        item_num: int = self.item_num
        
        # 1. 构建一个包含所有物品ID的张量
        # (item_num, )
        all_items: torch.Tensor = torch.arange(item_num, device=self.device)

        # 2. 一次计算 batch_size * item_num 个 (user, item) 对
        # [batch_size, item_num, seq_len]
        user_seqs: torch.Tensor = test_instances.unsqueeze(1).expand(-1, item_num, -1)
        # [batch_size, item_num, 1]
        items: torch.Tensor = all_items.view(1, -1, 1).expand(batch_size, -1, -1)
        # [batch_size, item_num]
        scores: torch.Tensor = self.calculate_preference(
            user_seqs.reshape(-1, seq_len),   # [batch_size * item_num, seq_len]
            items.reshape(-1, 1)              # [batch_size * item_num, 1]
        ).view(batch_size, -1)                # [batch_size, item_num]

        self.DINModel.train()
        return scores


    @torch.no_grad()
    def predict_item_chunk(self, test_instances: torch.Tensor, topk: int = 10, item_bs: int = 2000) -> torch.Tensor:
        """ predict the top-k items with the highest preference score for each user in test_instances
        
        batch-first 分块合并: 
        按 item 分块、user 合并到 batch，最优 GPU 方案
        按 item 分块, user 维合并到 batch 维。36 次前向，每次 32000 个 (user, item) 对，峰值约 3.4 GB（batch=16），GPU 实用方案

        比全合并版安全得多，同时 kernel launch 次数从 batch_size 次 (for i 循环版) 降到 item_num / item_bs 次

        :param test_instances: (batch_size, seq_len)
        :param topk: the number of items to recommend for each user
        :return: (batch_size, item_num) the preference scores for each user
        """
        self.DINModel.eval()

        # (batch_size, seq_len)
        test_instances: torch.Tensor = test_instances.to(self.device)
        batch_size = test_instances.shape[0]
        seq_len = test_instances.shape[1]
        item_num = self.item_num

        # (item_num, )
        all_items: torch.Tensor = torch.arange(item_num, device=self.device)

        # [batch_size, item_num] 全量分数矩阵
        scores = torch.zeros(batch_size, item_num, device=self.device)

        for start in range(0, item_num, item_bs):
            end = min(start + item_bs, item_num)
            chunk_size = end - start
            # (batch_size, chunk_size, 1)
            items = all_items[start:end].view(1, -1, 1).expand(batch_size, chunk_size, -1)
            # (batch_size, chunk_size, seq_len)
            user = test_instances.unsqueeze(1).expand(-1, chunk_size, -1)
            # (batch_size, chunk_size)
            scores[:, start:end] = self.calculate_preference(
                user.reshape(-1, seq_len),   # [batch_size * chunk_size, seq_len]
                items.reshape(-1, 1)         # [batch_size * chunk_size, 1]
            ).view(batch_size, -1)           # [batch_size, chunk_size]
        
        self.DINModel.train()
        return scores
    

    @torch.no_grad()
    def predict_user_chunk(self, test_instances: torch.Tensor, topk: int = 10, item_bs: int = 2000) -> torch.Tensor:
        """复刻手动拆分逻辑

        user 级分块:
        双层嵌套分块，GPU 利用率低，不推荐
        嵌套 for user → for item。576 次前向，每次 2000 对，GPU 利用率低，等价于 notebook 5000-step 代码的直接封装，不推荐。

        :return: (batch_size, item_num) the preference scores for each user
        """
        self.DINModel.eval()

        # (batch_size, seq_len)
        val_batch: torch.Tensor = test_instances.to(self.device)
        batch_size = val_batch.shape[0]
        seq_len = val_batch.shape[1]
        item_num = self.item_num
        
        # the preference matrix, which is used to store the preference score of each user for each item
        # (batch_size, item_num). 
        scores: torch.Tensor = torch.full((batch_size, item_num), -1.0e9, dtype=torch.float32)
        
        # (item_num, 1)
        all_items: torch.Tensor = torch.arange(item_num, device=self.device).view(-1, 1)

        for i, user in enumerate(val_batch):
            start_id = 0
            while start_id < item_num:
                # (item_bs, 1)
                part_labels: torch.Tensor = all_items[start_id : start_id + item_bs, :]
                # (1, f_num) -> (item_bs, f_num)
                user_history: torch.Tensor = user.to(self.device).expand(len(part_labels), val_batch.shape[1])

                scores[i, start_id : start_id + item_bs] = self.calculate_preference(
                    user_history,   # (item_bs, f_num)
                    part_labels     # (item_bs, 1)
                ).view(1, -1)
                start_id = start_id + item_bs

        self.DINModel.train()
        return scores
