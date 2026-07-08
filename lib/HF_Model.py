"""
HFTransformerModel
"""

import torch
import torch.nn as nn
import math
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertModel
import numpy as np

class HFTransformerModel(nn.Module):
    def __init__(
        self,
        src_voc_size: int = 1000, #item_num+1
        tgt_voc_size: int = 4,
        max_src_len: int = 69,
        max_tgt_len = 15,
        d_model = 24,
        nhead = 4,
        device = 'cuda',
        num_layers = 6,
        intermediate_size=1024,
        position_embedding_type='absolute'
    ):
        """HFTransformerModel
        :param src_voc_size: int, 源序列的词汇表大小
        :param tgt_voc_size: int, 目标序列的词汇表大小
        :param max_src_len: int, 最大源序列长度
        :param max_tgt_len: int, 最大目标序列长度
        :param d_model: int, 模型维度
        :param nhead: int, 多头注意力头数
        :param device: str, 设备类型
        :param num_layers: int, 层数
        :param intermediate_size: int, 中间层维度
        :param position_embedding_type: str, 位置编码类型
        :return: None
        """
        super(HFTransformerModel, self).__init__()

        self.src_voc_size = src_voc_size
        self.tgt_voc_size = tgt_voc_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.device = device
        self.src_pad = src_voc_size - 1
        self.tgt_pad = tgt_voc_size - 1

        config_encoder = BertConfig(
            vocab_size=src_voc_size, 
            hidden_size=d_model, 
            num_hidden_layers=num_layers, 
            num_attention_heads=nhead,
            intermediate_size=intermediate_size, 
            pad_token_id=self.src_pad,
            position_embedding_type=position_embedding_type,
            max_position_embeddings=max_src_len+1
        )
        config_decoder = BertConfig(
            vocab_size=tgt_voc_size, 
            hidden_size=d_model, 
            num_hidden_layers=num_layers, 
            num_attention_heads=nhead,
            intermediate_size=intermediate_size, 
            pad_token_id=self.tgt_pad,
            position_embedding_type=position_embedding_type,
            max_position_embeddings=max_tgt_len+1
        )
        config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
        self.trm = EncoderDecoderModel(config=config)
        self.trm.config.decoder_start_token_id = tgt_voc_size - 2
        self.trm.config.pad_token_id = self.tgt_pad
        #print(config)
    
    def forward(self, batch_x, batch_y):
        """
        batch_x: [batch_size, seq_length]
        batch_y: [batch_size, seq_length]
        """
        x, y = batch_x, batch_y
        src_key_padding_mask = _generate_pad_mask(x, self.src_pad).to(self.device)
        tgt_key_padding_mask = _generate_pad_mask(y, self.tgt_pad).to(self.device)
        return self.trm(
            input_ids=x, 
            attention_mask=src_key_padding_mask, # [batch_size, seq_length]
            labels=y,
            decoder_attention_mask=tgt_key_padding_mask
        )
        #return output

    #def generate(self, num_beams=40, topk=24, max_length=7, **kwargs):
        

def _generate_pad_mask(x, pad):
    """生成padding mask
    :param x: [batch_size, seq_length]
    :param pad: int, padding token id
    :return: [batch_size, seq_length]
    """
    # mask=(x != pad).float()
    # mask=torch.where(x!=pad,torch.ones(x.shape), torch.zeros(x.shape), device=device,dtype=torch.float32)
    mask = torch.FloatTensor((x != pad).cpu().numpy())
    return mask