"""
Trm4Rec. 

tree不同, decoder 不同, 但 encoder 最终共享. encoder/decoder分工: 
- encoder: 学通用的用户兴趣表示
- decoder: 学每棵树自己的路径预测规则
"""

import numpy as np
import time
import math

import torch
import torch.nn.functional as F

from lib.HF_Model import HFTransformerModel as TransformerModel
from lib.Tree_Model import Tree

class Trm4Rec:
    def __init__(
        self,
        item_num=1000,
        user_seq_len=69,
        d_model=24,
        nhead=4,
        device='cuda',
        num_layers = 4,
        k=2,                        # k branch on each tree
        item_to_code_file=None,
        code_to_item_file=None,
        tree_has_generated=False,
        init_way='random',
        data=None,
        max_iters=100,
        feature_ratio=0.8,
        parall=4,
        position_embedding_type='absolute'
    ):
        self.item_num = item_num    # 
        self.k = k                  # k branch on each tree
        self.device = device        # cuda:0 or cpu

        # 直接加载树
        if tree_has_generated:
            self.tree = Tree(construct=False, device=self.device) 
            self.tree.read_tree(
                item_to_code_file=item_to_code_file,
                code_to_item_file=code_to_item_file,
                k=k
            )
        else:
            # 先生产树, 再保存
            self.tree = Tree(
                data=data,
                max_iters=max_iters,
                feature_ratio=feature_ratio,
                item_num=item_num,
                k=k,
                init_way=init_way,
                parall=parall,
                device=self.device
            )

            # 保存code_to_item
            np.save(code_to_item_file, self.tree.code_to_item.cpu().numpy())
            
            # 保存item_to_code
            item_to_code_mat = torch.full((item_num,self.tree.tree_height), -1, dtype=torch.int64)
            for item_id, paths in self.tree.item_to_code.items():
                assert len(paths) > 0
                item_to_code_mat[item_id] = paths[0]
            self.tree.item_to_code = item_to_code_mat.to(self.device)
            np.save(item_to_code_file, item_to_code_mat.cpu().numpy())



        """
        0 .. k-1   ->  branch token(真正要预测的路径分支)
        k          ->  start token(解码起始符)
        k+1        ->  pad token(填充符)
        """

        """
        一句话
        start token: 告诉 decoder 从哪里开始生成路径
        pad token: 让 batch 里长短不一的路径能够拼成统一形状的 tensor
        """

        """Note: start token 为什么需要?
        因为 Trm4Rec 的 decoder 是自回归式生成路径, 不是一次性输出所有 path token. 
        
        自回归生成时, decoder 需要看到“前面的 token”才能预测“下一个 token”. 但生成路径时, 一开始没有任何前缀. 

        所以用一个特殊 token start 来表示“路径开始”: 
            输入到 decoder:  [start]
            decoder 预测:    branch_1
            输入到 decoder:  [start, branch_1]
            decoder 预测:    branch_2
            输入到 decoder:  [start, branch_1, branch_2]
            decoder 预测:    branch_3
            ...
        没有 start token, decoder 就无法知道“什么时候开始生成路径”.         
        """
        self.src_voc_size = item_num + 1                # TODO: 为什么 + 1? pad_token
        self.tgt_voc_size = k + 2                       # TODO: k + 2, 为什么是+2? start_token 和 pad_token
        self.max_src_len = user_seq_len                 # 用户行为序列长度
        self.max_tgt_len = self.tree.tree_height + 1    # TODO: 为什么 + 1? start_token
        
        # Transformer模型本体
        self.trm_model = TransformerModel(
            src_voc_size=self.src_voc_size, 
            tgt_voc_size=self.tgt_voc_size, 
            max_src_len=self.max_src_len,
            max_tgt_len=self.max_tgt_len,
            d_model=d_model, 
            nhead=nhead, 
            device=device,
            num_layers=num_layers,
            position_embedding_type=position_embedding_type
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.trm_model.parameters(), lr=1e-3, amsgrad=True)
        self.batch_num = 0                  # 
        print(self.trm_model)               # 
        print(self.trm_model.parameters())  # 


    def update_learning_rate(
            self, 
            t, 
            learning_rate_base=1e-3, 
            warmup_steps=5000,
            decay_rate=1./3, 
            learning_rate_min=1e-5
        ):
        """Learning rate with linear warmup and exponential decay
        :
        """
        lr = learning_rate_base * np.minimum(
            (t + 1.0) / warmup_steps,
            np.exp(decay_rate * ((warmup_steps - t - 1.0) / warmup_steps)),
        )
        lr = np.maximum(lr, learning_rate_min)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


    def update_model(self, batch_x, batch_y):
        """"TODO: 命名不合适

        :param batch_x: (batch_size, seq_len)
        :param batch_y: (batch_size, )
        :return loss
        """
        self.batch_num += 1
        
        # [batch_size, tree_height]
        path = self.tree.item_to_code[batch_y]
        assert path.shape[0] == batch_x.shape[0] and path.shape[1] == self.tree.tree_height
        print(f"debug. path.dtype: {path.dtype}") # orch.int64
        
        output = self.trm_model(batch_x.to(self.device), path.to(self.device))
        # loss: 标量
        loss = output.loss
        print(f"debug. loss.shape: {loss.shape}")
        
        # logits: [batch_size, tree_height, tgt_voc_size]
        logits = output.logits
        print(f"debug. logits.shape: {logits.shape}")
        return loss
    

    def compute_scores(self, batch_x, batch_y):
        """计算路径级别分数
        
        当前树认为这个 candidate item 在这个 user history 下有多合理.

        每棵树都有自己的path编码和decoder -> 不同树对同一个 (user, item) 对的打分不同

        把 用户真实候选item路径 和 模型对每一层 branch 的预测 logits 做逐位置交叉熵, 得到: 
        每个用户、每个路径位置上的分类损失矩阵 [batch_size, tree_height]

        :param batch_x: [batch_size * max_len, seq_len], 用户历史item序列
        :param batch_y: [batch_size * max_len], 候选item
        :return: [batch_size * max_len, tree_height]
        """
        self.trm_model.eval()

        start_time = time.time()

        # [batch_size * max_len, seq_len]
        batch_x = batch_x.to(self.device)
        
        # [batch_size * max_len]
        batch_y = batch_y.to(self.device)
        print(f"debug. batch_x.shape: {batch_x.shape}")

        # [batch_size * max_len, tree_height]
        # 候选item的真实路径
        target = self.tree.item_to_code[batch_y]
        print(f"debug. target.shape: {target.shape}")
        
        assert target.shape[0] == batch_x.shape[0] and target.shape[1] == self.tree.tree_height
        
        print(f"debug. time1: {time.time() - start_time}")
        start_time = time.time()
        
        output = self.trm_model.trm(
            input_ids=batch_x,      # [batch_size * max_len, seq_len]
            labels=target,          # [batch_size * max_len, tree_height]
            attention_mask=(batch_x != self.trm_model.src_pad)
        )
        print(f"debug. type(output): {type(output)}")
        print(f"debug. output.logits.shape: {output.logits.shape}") # (batch_size, k + 2)

        print(f"debug. time2: {time.time() - start_time}")
        start_time = time.time()
        
        # 模型对每个用户、每个路径位置、每个候选 branch 的预测logits
        # [batch_size * max_len, tree_height, k]
        logits = output.logits[:, :, :self.tgt_voc_size - 2]
        print(f"debug. logits.shape: {logits.shape}") # (batch_size, k)

        # 每个用户的真实 item -> 被转换成真实路径(每个位置是该层应选的branch id)
        # input: [batch_size * max_len, k, tree_height]
        # target: [batch_size * max_len, tree_height]
        # loss: [batch_size * max_len, tree_height]
        loss = F.cross_entropy(
            input=logits.transpose(1, 2), 
            target=target, 
            reduction='none'
        )
        print(f"debug. loss.shape: {loss.shape}")

        """Note: 为什么要 transpose(1, 2)?
        F.cross_entropy 对多维输入的要求是: 
            input shape  = [N, C, d1, d2, ...]
            target shape = [N, d1, d2, ...]
        其中, C 必须是类别维
        """

        """Note: cross_entropy 实际在算什么?
        对每个用户、每个路径位置, 都在做一个 k 分类问题.

        比如某个位置: 
          - 模型给出 18 个 branch 的分数
          - 真实标签是 branch 12
        那这一项损失就是: 
          - log p(branch=12 | 当前用户历史, 当前路径位置)
        
        整批数据上, 它会对每个位置都算一次. 所以输出不是一个标量, 而是: 
        loss.shape == [batch_size * max_len, tree_height], 因为reduction='none'
        """

        """"Note: 为什么这里要 reduction='none'?
        如果不用它, cross_entropy 默认会把所有位置直接平均掉, 得到一个标量
        但这里作者想保留: 每个用户、每个路径位置的单独损失
        """
        print(f"debug. time3: {time.time() - start_time}")
        self.trm_model.train()

        """Note: compute_scores()意图: 给候选 item 打分

        候选 item 先变成 path temp_y, 然后模型算: 
          - 如果这个 item 真的是目标
          - 它的每层 path token 有多容易被模型预测出来

        交叉熵越小, 说明模型越认可这条路径
          - loss 小 -> score 大
          - loss 大 -> score 小
        这就把 路径预测损失 变成 候选item打分
        """
        return -loss


    @torch.no_grad()
    def predict(self, batch_x, topk=24, num_beams=100):
        """是自定义 beam search. 为每个用户生成多条候选路径, 再解码回 item id. 
        
        :param batch_x: [batch_size, seq_len], 一批用户历史序列
        :param topk: 最终每用户保留多少条候选item
        :param num_beams: 每一步保留多少条候选路径
        :return: [batch_size, topk]
        """
        self.trm_model.eval()

        #=========== 阶段 1. 初始化 ===========
        batch_x = batch_x.to(self.device)
        batch_size, seq_len = batch_x.shape[0], batch_x.shape[1]

        start_time = time.perf_counter()
        # [batch_size * num_beams, seq_len]
        # 每个用户的历史被复制num_beams份 (因为beam search维护num_beams条候选路径, 每条路径都需要一份完整的encoder输入)
        input_ids = torch.zeros((batch_size * num_beams, seq_len), device=self.device, dtype=torch.int64)
        
        # [batch_size, num_beams]
        select_index = torch.arange(batch_size).view(-1, 1).repeat(1, num_beams).to(self.device)
        # [batch_size * num_beams, seq_len]
        input_ids = batch_x[select_index.view(-1)]
        
        # 每条 beam 的累积对数分数. 初始化时只有[0](start_token) 是 0, 其他都是极小值, 确保第一轮只会从 beam 0 开始扩展
        # [batch_size, num_beams]
        pred_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float32, device=self.device)
        pred_scores[:, 0] = 0
        
        # [batch_size * num_beams, 1]
        pred = torch.full(
            (batch_size * num_beams, 1), 
            self.trm_model.trm.config.decoder_start_token_id,       # start_token
            dtype=torch.int64, 
            device=self.device
        )
        print(f"debug. init pred.shape: {pred.shape}")

        init_time = time.perf_counter() - start_time
        print(f"debug. init_time: {init_time}")
        start_time = time.perf_counter()
        
        #=========== 阶段 2. 逐位置 beam search 循环 ===========
        for j in range(self.max_tgt_len - 1):   # tree_height
            print(f"debug. loop j={j}, pred.shape={pred.shape}")
            start_time = time.perf_counter()
            
            #=========== 阶段 2.1 单步前向 ===========
            # 注意: 由于是预测阶段, 因此decoder入参是decoder_input_ids=pred, 不是labels=pred. 不会自动做 labels 右移
            output = self.trm_model.trm(
                input_ids=input_ids,            # [batch_size * num_beams, seq_len]
                decoder_input_ids=pred,         # [batch_size * num_beams, 1]
                attention_mask=(input_ids != self.trm_model.src_pad)
            )
            
            #=========== 阶段 2.2: 取最后一位logits ===========
            # [batch_size * num_beams, cur_len, tgt_voc_size]
            logits = output.logits

            # [batch_size, num_beams, k+2]
            last_token_logits = logits[:, -1, :].view(batch_size, num_beams, -1)
            
            # [batch_size, num_beams, k]
            # 最后一个位置中每个branch token的对数概率
            last_token_scores = torch.log_softmax(last_token_logits, dim=-1)[:, :, :self.tgt_voc_size - 2]
            
            #=========== 阶段 2.3 累积分数 + 路径扩展 ===========
            # [batch_size, num_beams * k], 每条旧 beam 分裂出 k 条新路径
            pred_scores = (pred_scores.view(batch_size, num_beams, 1) + last_token_scores).view(batch_size, -1)
            
            # 每条旧路径复制 k 份, 每份追加一个新的 branch token
            # [batch_size * num_beams, 1, 1] -> [batch_size * num_beams, k, 1] -> [batch_size, num_beams * k, 1]
            pred = pred.view(batch_size * num_beams, 1, -1).repeat(1, self.tgt_voc_size - 2, 1).view(batch_size, num_beams * (self.tgt_voc_size - 2), -1)
            
            # [batch_size, num_beams * k, 1]
            pred_last_token = torch.arange(
                0, self.tgt_voc_size - 2, device=self.device
            ).repeat(batch_size * num_beams).view(batch_size, -1, 1)

            # [batch_size, num_beams * k, cur_len]
            pred = torch.cat([pred, pred_last_token], dim=-1)
            
            #=========== 阶段 2.4 裁剪 beam ===========
            # 最后一步: 从 num_beams * k 条候选里保留分数最高的 topk 条 
            if pred.shape[-1] == self.max_tgt_len:  # tree_height
                # values:  [batch_size, topk]   # topk个最高分数
                # index:   [batch_size, topk]   # 对应位置索引
                pred_scores, index = pred_scores.topk(topk)
                # [batch_size, topk, cur_len]
                index = index.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
                # [batch_size * topk, cur_len]
                pred = pred.gather(dim=1, index=index).view(batch_size * topk, -1)    
            else:
                # 非最后一步: 从 num_beams * k 条候选里保留分数最高的 num_beams 条
                # values:  [batch_size, num_beams]   # topk个最高分数
                # index:   [batch_size, num_beams]   # 对应位置索引
                pred_scores, index = pred_scores.topk(num_beams)
                # [batch_size, num_beams, cur_len]
                index = index.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
                # [batch_size * num_beams, cur_len]
                pred = pred.gather(dim=1, index=index).view(batch_size * num_beams, -1)
            compute_time += time.perf_counter() - start_time
        print(f"debug. final pred.shape={pred.shape}")
        start_time = time.perf_counter()
        
        #=========== 阶段 3: 解码 ===========
        # [batch_size, topk, self.max_tgt_len]
        # 去掉每个路径的第一个 token（start）
        all_pred = pred.view(batch_x.shape[0], topk,self.max_tgt_len)[:, :, 1:] 

        # [batch_size, topk * tree_num]
        # path token 序列 → leaf code → item id
        label = self.decode(all_pred)
        
        decode_time = time.perf_counter() - start_time
        print(f"debug. decode_time: {decode_time}")
        
        self.trm_model.train()
        return label
    
    
    @torch.no_grad()
    def predict_hf(self, batch_x, topk=24, num_beams=100, batch_size=50):
        """
        predict_hf 和 predict 做的是同一件事（beam search 生成路径 + 解码回 item id），区别在于：
        - predict：完全手写的 beam search
        - predict_hf：先用 Hugging Face 的 generate() 做初选，再自己补一轮 beam 扩展和重排

        :param batch_x: [batch_size, seq_len], 一批用户历史序列
        :param topk: 最终每用户保留多少条候选item
        :param num_beams: 每一步保留多少条候选路径
        :param batch_size: 拆 batch 用
        :return: [batch_size, topk]
        """
        self.trm_model.eval()

        # 当用户数超过 batch_size 时，拆成多个子 batch 循环处理。all_pred 用于收集各子 batch 的结果。
        num_batch = int(math.ceil(batch_x.shape[0] / batch_size))
        
        all_pred = []
        for i in range(num_batch): 
            x = batch_x[i * batch_size : (i + 1) * batch_size].to(self.device)
            model_kwargs = {
                "input_ids": x,
                "attention_mask": torch.FloatTensor((x != self.trm_model.src_pad).cpu().numpy()).to(self.device)
            }
            print(model_kwargs)
            
            """
            参数	        含义
            num_beams=100	beam search beam 数
            num_return_sequences=100	返回 num_beams 条路径
            do_sample=False	不走随机采样，取分数最高的
            max_length=4	最大生成长度（tree_height，不含 start）
            """

            # pred：[batch_size * num_beams, max_length]，路径 token 序列
            # pred_scores：[batch_size * num_beams]，每条路径的累积分数
            pred, pred_scores = self.trm_model.trm.generate(
                num_beams=num_beams,
                **model_kwargs,
                num_return_sequences=num_beams,
                do_sample=False, 
                max_length=self.max_tgt_len - 1
            )

            """
            generate() 的问题：只返回 beam 最终路径，缺少 beam 扩展信息
            Hugging Face 的 generate() 只返回 num_beams 条最终路径，但我们需要的是 从这 num_beams 条路径再扩展一次，从 num_beams × k 条候选里重新选 topk。所以后面都是自己补的工作。
            """

            # 重算最后一帧分数
            # 这一步是为了拿到每个位置的 branch token 对数概率，generate() 不直接暴露这些信息。
            last_token_logits = self.trm_model.trm(
                input_ids=model_kwargs['input_ids'].repeat_interleave(num_beams, dim=0), 
                attention_mask=model_kwargs['attention_mask'].repeat_interleave(num_beams, dim=0), 
                decoder_input_ids=pred
            ).logits[:, -1, 0:] # 只保留前 k 个 branch token（去掉 start 和 pad）
            

            # 手动 beam 扩展 + 重排
            input_batch_size = model_kwargs['input_ids'].shape[0]
            pred = pred.view(input_batch_size, num_beams, -1)
            pred_scores = pred_scores.view(input_batch_size, num_beams)
            
            last_token_logits = last_token_logits.view(input_batch_size, num_beams,-1)
            
            last_token_logits = torch.log_softmax(last_token_logits, dim=-1)[:, :, :self.tgt_voc_size - 2]
            
            pred_scores = pred_scores.view(input_batch_size, num_beams, 1) + last_token_logits
            
            pred_last_token = (torch.arange(0, self.tgt_voc_size - 2)).repeat(input_batch_size * num_beams).to(self.device)
            
            pred = torch.cat([pred.repeat_interleave(self.tgt_voc_size - 2, dim=1), pred_last_token.view(input_batch_size, -1, 1)], dim=-1)
            
            pred_scores = pred_scores.view(input_batch_size, -1)
            
            index = pred_scores.argsort(dim=-1, descending=True)[:, :topk].unsqueeze(-1).expand(-1, -1, self.max_tgt_len)
            
            pred = pred.gather(dim=1, index=index).view(input_batch_size * topk, -1)
            
            all_pred.append(pred.cpu())
        
        all_pred = torch.cat(all_pred, dim=0)
        
        all_pred = all_pred.view(batch_x.shape[0], topk, self.max_tgt_len)[:, :, 1:]
        
        label = self.decode(all_pred)   # [batch_size, topk * tree_num]
        
        self.trm_model.train()

        return label


    def decode(self, all_pred):
        """translate Decimal into tree_num-ary, i.e. find the result on each tree

        :param all_pred: [batch_size, topk, tree_height], eliminate the starting symbol
        :return: [batch_size, topk]
        """
        batch_size = all_pred.shape[0]
        topk = all_pred.shape[1]
        
        start_position = (torch.log(all_pred + 1.0) / torch.log(self.tree_num)).ceil() - 1
        return self.tree.path_to_label(all_pred).view(batch_size, topk)
