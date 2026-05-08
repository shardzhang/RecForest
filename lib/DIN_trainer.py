
import torch
import numpy as np
from lib.DIN_Model import DeepInterestNetwork

class DINTrain:
    def __init__(self,
        item_num=100,
        sample_negative_num=60,
        emb_dim=96,
        device='cpu',
        feature_groups=[20,20,10,10,2,2,2,1,1,1],
        sum_pooling=False,
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3, amsgrad=True)
    ):
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
        self.optimizer = optimizer(self.DINModel.parameters())
        
        # the number of batches that have been trained, which is used to calculate the learning rate
        self.batch_num = 0

    def update_learning_rate(
        self, 
        t, 
        learning_rate_base=1e-3, 
        warmup_steps=5000,
        decay_rate=1./3, 
        learning_rate_min=1e-6
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
            (t + 1.0) / warmup_steps,
            np.exp(decay_rate * ((warmup_steps - t - 1.0) / warmup_steps)),
        )
        lr = np.maximum(lr, learning_rate_min)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def uniform_sampled_softmax(self, batch_users, batch_labels, N):
        """sampled softmax loss
        TODO: 是否可以直接调用tf.nn.sampled_softmax_loss来计算损失?

        :param: batch_users: (batch_size, f_num)
        :param: batch_labels: (batch_size, 1)
        :param: N: negative sample number
        :return: loss. scalar
        """
        batch_size = batch_users.shape[0]        

        # generate samples, the first column is positive labels, and the rest are negative labels
        # (batch_size, N + 1)
        samples = torch.full((batch_size, N + 1), -1, device=self.device, dtype=torch.int64)
        samples[:, 0:1] = batch_labels      # positive labels
        samples[:, 1:] = torch.randint(     # random negative labels
            low=0, 
            high=self.item_num, 
            size=(batch_size, N), 
            device=self.device,
        )

        # the effective index of samples, which is the index of items that are not filled with the absence (item_num)
        # (batch_size, N + 1)
        effective_index = torch.full(samples.shape, True, device=self.device, dtype=torch.bool)
        # the first column of samples is positive labels, so the effective index of the first column is always True
        # the effective index of the rest columns is True if the negative label is not equal to the positive label, otherwise False
        effective_index[:, 1:] = samples[:, 0:1] != samples[:, 1:]

        # compute the log of sampling probability, which is used to correct the bias caused by negative sampling
        # (batch_size, N + 1)
        log_q_matrix = torch.full(samples.shape, 0.0, device=self.device, dtype=torch.float32)

        # 负采样概率: q = 每个用户的有效负样本数 / (总物品数 - 1)
        # 每个用户的有效的所有负样本共享同一个log(q)
        log_q_matrix[:, 1:][effective_index[:, 1:]] = torch.log(
            effective_index[:, 1:].sum(-1).view(batch_size, 1) * 1.0 / (self.item_num - 1)
        ).expand(batch_size, N)[effective_index[:, 1:]]

        # (batch_size, N + 1)
        user_index = torch.arange(batch_size, device=self.device).view(-1, 1).expand(samples.shape)
        # (batch_size * effective_index_len, )
        # 用和原张量同形状的布尔张量做索引，会把原张量中所有True位置的元素展平成一维张量
        user_index = user_index[effective_index]

        # (batch_size * effective_index_len, )
        samples = samples[effective_index]
        # (batch_size * effective_index_len, 1)
        samples = samples.view(-1, 1)

        # (batch_size * effective_index_len, f_num)
        # 把每个用户的特征重复取出effective_index_len次, 拼成一个更长的矩阵, 和samples中的标签一一对应
        batch_users = batch_users[user_index]

        # (batch_size, N + 1)
        # the preference score of the positive and negative samples, which is computed by DINModel, and the bias caused by negative sampling is corrected with log_q_matrix
        o_pi = torch.full(log_q_matrix.shape, -1.0e9, device=self.device, dtype=torch.float32)

        # (batch_size * effective_index_len, 1)
        logits = self.DINModel(batch_user=batch_users, batch_label=samples)[:, 0]
        # 布尔掩码索引赋值: 只操作两个张量中掩码为True的位置
        o_pi[effective_index] = logits - log_q_matrix[effective_index]

        # compute the sampled softmax loss with log-sum-exp trick for numerical stability
        # (batch_size, ) -> scalar
        return (-o_pi[:, 0] + torch.logsumexp(o_pi, dim=1)).mean(-1)

    def update_DIN(self, batch_users, batch_labels):
        """ update the parameters of DINModel with sampled softmax loss
        :param batch_users: (batch_size, f_num)
        :param batch_labels: (batch_size, 1)
        :return: loss. scalar
        """
        self.batch_num += 1
        loss = self.uniform_sampled_softmax(
            batch_users.to(self.device), 
            batch_labels.to(self.device).view(-1, 1), 
            self.N
        )
        loss.backward()             # compute the gradient
        self.optimizer.step()       # update the parameters
        self.optimizer.zero_grad()  # clean the gradient
        self.update_learning_rate(self.batch_num)
        return loss

    def calculate_preference(self, batch_user, batch_items):
        """ calculate the preference score of batch_user to batch_items with DINModel
        :param batch_user: (batch_size, f_num)
        :param batch_items: (batch_size, 1)
        :return: (batch_size, 1)
        """
        return self.DINModel(batch_user, batch_items)

    @torch.no_grad()
    def predict(self, test_instances, topk=10):
        """ predict the top-k items with the highest preference score for each user in test_instances
        :param test_instances: (batch_size, f_num)
        :param topk: the number of items to recommend for each user
        :return: (batch_size, topk) the recommended item IDs for each user
        """
        self.DINModel.eval()

        test_instances = test_instances.to(self.device)
        batch_size = test_instances.shape[0]
        
        # 1. 构建一个包含所有物品ID的张量
        # (item_num, )
        all_items = torch.arange(self.item_num, device=self.device)
        
        # 2. 计算每个用户对所有物品的偏好得分
        # 暴力全量排序 Brute-force Top-K. 推荐系统里最标准、最准确的评估方式
        scores = []
        for i in range(batch_size):
            # (1, f_num) -> (item_num, f_num)
            user_seq = test_instances[i:i+1].repeat(all_items.shape[0], 1)
            
            # (item_num, ) -> (item_num, 1)
            items = all_items.view(-1, 1)
            
            # (item_num, 1) -> (item_num, )
            score = self.calculate_preference(user_seq, items).view(-1)
            scores.append(score)
        # (batch_size, item_num)
        scores = torch.stack(scores)
        
        # 3. 取Top-K
        topk_scores, topk_items = torch.topk(scores, k=topk, dim=-1)

        self.DINModel.train()
        return topk_items
