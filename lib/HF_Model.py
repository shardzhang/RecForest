"""
HFTransformerModel
"""

import torch
import torch.nn as nn
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertModel

class HFTransformerModel(nn.Module):
    def __init__(
        self,
        src_voc_size: int = 1000,   # item_num + 1. pad_token
        tgt_voc_size: int = 20,     # k + 2. start_token, pad_token
        max_src_len: int = 69,      # seq_len
        max_tgt_len = 5,            # tree_height + 1. start_token
        d_model = 24,
        nhead = 4,
        device = 'cuda',
        num_layers = 6,
        intermediate_size=1024,
        position_embedding_type='absolute'
    ):
        super(HFTransformerModel, self).__init__()

        self.src_voc_size = src_voc_size    # item_num + 1. pad_token
        self.tgt_voc_size = tgt_voc_size    # k + 2. start_token, pad_token
        self.max_src_len = max_src_len      # seq_len
        self.max_tgt_len = max_tgt_len      # tree_height + 1. start_token
        self.src_pad = src_voc_size - 1     # item_num
        self.tgt_pad = tgt_voc_size - 1     # k + 2 - 1
        self.device = device                # 

        config_encoder = BertConfig(
            vocab_size=src_voc_size,                            # item_num + 1. pad_token
            hidden_size=d_model, 
            num_hidden_layers=num_layers, 
            num_attention_heads=nhead,
            intermediate_size=intermediate_size, 
            pad_token_id=self.src_pad,                          # item_num
            position_embedding_type=position_embedding_type,
            max_position_embeddings=max_src_len + 1             # 位置 embedding 表的大小
        )

        config_decoder = BertConfig(
            vocab_size=tgt_voc_size,                            # k + 2. start_token, pad_token
            hidden_size=d_model, 
            num_hidden_layers=num_layers, 
            num_attention_heads=nhead,
            intermediate_size=intermediate_size, 
            pad_token_id=self.tgt_pad,                          # k + 2 - 1
            position_embedding_type=position_embedding_type,
            max_position_embeddings=max_tgt_len + 1             # 位置 embedding 表的大小
        )

        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=config_encoder, decoder_config=config_decoder)
        print(config.decoder.is_decoder)  # True, decoder 侧的 self-attention 是 causal（因果式）, 不会看到未来的 token
        print(config.encoder.is_decoder)  # False, encoder 侧的 self-attention 是 bidirectional（双向）, 符合预期
        print(f"debug. config: {config}")
        """
        debug. config: EncoderDecoderConfig {
            "decoder": {
                "_name_or_path": "",
                "add_cross_attention": true,            # decoder 能通过 cross-attention 看到 encoder 的输出
                "architectures": null,
                "attention_probs_dropout_prob": 0.1,
                "bos_token_id": null,
                "chunk_size_feed_forward": 0,
                "classifier_dropout": null,
                "dtype": null,
                "eos_token_id": null,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 96,
                "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
                },
                "initializer_range": 0.02,
                "intermediate_size": 1024,
                "is_decoder": true,                     # decoder 侧的 self-attention 是 causal（因果式）, 不会看到未来的 token
                "is_encoder_decoder": false,
                "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1
                },
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 6,
                "model_type": "bert",
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "output_attentions": false,
                "output_hidden_states": false,
                "pad_token_id": 19,
                "position_embedding_type": "absolute",
                "problem_type": null,
                "return_dict": true,
                "tie_word_embeddings": true,
                "type_vocab_size": 2,
                "use_cache": true,
                "vocab_size": 20
            },
            "decoder_start_token_id": null,             # config 里默认是空, 后面代码第 68 行手动设置了 self.trm.config.decoder_start_token_id = tgt_voc_size - 2
            "encoder": {
                "_name_or_path": "",
                "add_cross_attention": false,
                "architectures": null,
                "attention_probs_dropout_prob": 0.1,
                "bos_token_id": null,
                "chunk_size_feed_forward": 0,
                "classifier_dropout": null,
                "dtype": null,
                "eos_token_id": null,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 96,
                "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
                },
                "initializer_range": 0.02,
                "intermediate_size": 1024,
                "is_decoder": false,
                "is_encoder_decoder": false,
                "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1
                },
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 70,
                "model_type": "bert",
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "output_attentions": false,
                "output_hidden_states": false,
                "pad_token_id": 71436,
                "position_embedding_type": "absolute",
                "problem_type": null,
                "return_dict": true,
                "tie_word_embeddings": true,
                "type_vocab_size": 2,
                "use_cache": true,
                "vocab_size": 71437
            },
            "is_encoder_decoder": true,
            "model_type": "encoder-decoder",
            "pad_token_id": null,                           # 顶层config没设置, 靠第 74 行手动补的 self.trm.config.pad_token_id = self.tgt_pad
            "transformers_version": "5.12.1"
        }
        """
        
        # TODO: 为什么是一个encoder-decoder结构的模型?
        # 标准的 seq2seq 结构
        # encoder(history) -> 上下文表示 -> decoder(生成 path sequence) -> item
        # 所以你天然需要一个 seq2seq 模型: 
        # - encoder 读用户历史
        # - decoder 生成路径
        self.trm = EncoderDecoderModel(config=config)

        """
        如果不设置这个值会怎样？
        训练时, Hugging Face 靠它来对 labels 做右移, 生成 decoder 的输入序列
            labels              = [5, 2, 17, 9]      # 训练目标
            decoder_input_ids   = [k, 5, 2, 17]      # decoder 看到的输入(k = decoder_start_token_id)
        推理 / 生成时, decoder 把 decoder_start_token_id 作为第一个输入 token, 开始逐位生成路径
        """
        self.trm.config.decoder_start_token_id = tgt_voc_size - 2   # fixme: 为什么要设置decoder_start_token_id?

        """
        EncoderDecoderModel 的 config 不会自动从 decoder config 继承 pad_token_id, 所以需要手动写这一行.
        否则 loss 计算时会包含 pad 位置, 导致指标虚低
        """
        self.trm.config.pad_token_id = self.tgt_pad                 # fixme: 什么用途? 是否重复设置了?
    

    def forward(self, batch_x, batch_y):
        """
        batch_x:(batch_size, seq_len)
        batch_y:(batch_size, tree_height)
        """
        x, y = batch_x, batch_y
        
        #(batch_size, seq_len)
        src_key_padding_mask = _generate_pad_mask(x, self.src_pad).to(self.device)
        #(batch_size, tree_height)
        tgt_key_padding_mask = _generate_pad_mask(y, self.tgt_pad).to(self.device)
        
        """
        # 当你只传 labels 不传 decoder_attention_mask 时，内部自动做：
        decoder_attention_mask = (labels != pad_token_id)
        """
        output = self.trm(
            input_ids=x,                                    #(batch_size, seq_len)
            attention_mask=src_key_padding_mask,            #(batch_size, seq_len)
            labels=y,                                       #(batch_size, tree_height)
            decoder_attention_mask=tgt_key_padding_mask     #(batch_size, tree_height)
        )
        """
        output.logits
        shape: [batch_size, tree_height, tgt_voc_size]
        含义: decoder 在每个位置上, 对 tgt_voc_size 个候选 token 的预测分数
        后面会从这里取前 k 列做 cross_entropy
        
        output.loss
        shape: 标量
        含义: decoder 每个位置上的 cross-entropy 损失的平均值
        训练时直接拿它 .backward()
        
        在训练 / 打分里的用法差异
        训练update_model: 取 output.loss, 不自己算 loss
        打分compute_scores: 用 output.logits, 手动算 F.cross_entropy(..., reduction='none'), 保留每个位置的 loss, 再 -loss 作为 score
        所以同一个 output 在两条路径里取了不同的字段
        """

        """
        enc_out = trm.encoder(input_ids=x)
        enc_out字段: 
        - last_hidden_state	[batch_size, seq_len, d_model]	encoder 最后一层的全部位置输出
        - pooler_output	[batch_size, d_model]	[CLS] 位置过 Linear + Tanh

        关键点: pooler_output 确实被计算了(shape [2, 16]), 但没有被 EncoderDecoderModel 的通路消费。decoder 只使用 last_hidden_state。

        计算层面
        EncoderDecoderModel 调用 self.encoder(...) 时, BertModel 确实计算了 pooler_output([CLS] 位置过 Linear+Tanh)。这部分计算是浪费的, 但它确实在跑。

        消费层面
        EncoderDecoderModel 默认的 forward() 通路只取了 encoder_outputs.last_hidden_state 传给 decoder, 没有读取 encoder_outputs.pooler_output。所以它被计算了, 但没有被使用。

        更准确的说法
        一定会被计算(因为 BertModel 自带它)
        一定会被浪费(因为 EncoderDecoderModel 不消费它)
        但你可以主动取到它——只要在 forward 后手动取: 
        output = self.trm(input_ids=x, labels=y)
        pooled = self.trm.encoder(input_ids=x).pooler_output
        所以结论是: 
        默认的 EncoderDecoderModel 不能用 BertPooler, 但你可以自己绕过默认 forward 去拿到它。
        """
        return output


    # TODO
    #def generate(self, num_beams=40, topk=24, max_length=7, **kwargs):
        

def _generate_pad_mask(x, pad):
    """生成padding mask
    :param x: [batch_size, seq_length]
    :param pad: int, padding token id
    :return: [batch_size, seq_length]
    """
    # mask=(x != pad).float()
    # mask=torch.where(x!=pad,torch.ones(x.shape), torch.zeros(x.shape), device=device,dtype=torch.float32)
    mask = torch.FloatTensor((x != pad).cpu().numpy())  # TODO: 为什么.numpy()?
    return mask