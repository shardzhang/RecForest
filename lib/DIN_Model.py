"""
Deep Interest Network (DIN). DONE
"""

from __future__ import annotations
import torch.nn as nn
import torch
from typing import Sequence

class DeepInterestNetwork(nn.Module):
    def __init__(
        self, 
        item_num=100,
        embedding_dim=96,
        feature_groups=[20,20,10,10,2,2,2,1,1,1],
        sum_pooling=False,
    ):
        """初始化DIN Model

        :param item_num: item number
        :param embedding_dim: embedding dimension
        :param feature_groups: feature groups
        :param sum_pooling: sum pooling
        """ 
        super().__init__()

        self.item_num = item_num
        self.embed_dim = embedding_dim
        self.sum_pooling = sum_pooling
        self.item_embedding = EmbeddingLayer(item_num, embedding_dim)
        self.attention_unit = LocalActivationUnit(
            hidden_dims=[64, 16], 
            bias=[True, True], 
            embedding_dim=embedding_dim, 
            batch_norm=False,
        )

        if sum_pooling:    
            self.fc_layer = FullyConnectedLayer(
                input_dim=2 * embedding_dim,
                hidden_dims=[200, 80, 1],
                bias=[True, True, True],
                activation='relu',  # TODO: dice activation function
                sigmoid=False,
            )
        else:                               
            self.fc_layer = FullyConnectedLayer(
                input_dim=(len(feature_groups) + 1) * embedding_dim,
                hidden_dims=[200, 80, 1],
                bias=[True, True, True],
                activation='relu', # TODO: dice activation function
                sigmoid=False,
            )

            # window matrix for each window's weight sum
            # shape: (len(feature_groups), sum(feature_groups))
            window_matrix = torch.zeros(len(feature_groups), sum(feature_groups))
            start_index = 0
            for i, dim in enumerate(feature_groups):
                window_matrix[i, start_index : start_index + dim] = 1.0
                start_index += dim
            self.register_buffer('window_matrix', window_matrix)

    def forward(self, batch_user, batch_label):
        """item_num fill with the absence in batch_user

        :param batch_user: (batch_size, f_num)
        :param batch_label: (batch_size, 1)
        :return: (batch_size, 1)
        """

        # (批次大小, 序列长度)
        batch_size, f_num = batch_user.shape

        # get the effective index of batch_user, which is the index of items that are not filled with the absence (item_num)
        # (batch_size, f_num)
        effective_index = batch_user < self.item_num
        
        # label expand to the same shape as batch_user for indexing, then get the effective labels
        # 布尔索引: 用和原张量同形状的布尔张量做索引，会把原张量中所有True位置的元素展平成一维张量
        # 和 indices[gather_idx]（花式索引，输出形状 = gather_idx 形状）不同，布尔索引一定返回一维——所有被 True 选中的元素被展平成一个一维列表，不保留原始形状。
        # (batch_size * effective_index_len, )
        effective_labels = batch_label.expand(batch_user.shape)[effective_index]

        if self.sum_pooling:
            # (batch_size, f_num)
            weight_matrix = torch.zeros((batch_user.shape), device=batch_user.device, dtype=torch.float32)
            
            # (batch_size * effective_index_len, 1)
            attention_unit = self.attention_unit(
                self.item_embedding(batch_user[effective_index]),   # (batch_size * effective_index_len, embed_dim)
                self.item_embedding(effective_labels),              # (batch_size * effective_index_len, embed_dim)
            )
            weight_matrix[effective_index] = attention_unit.view(-1)

            # compute the weighted sum of the embeddings of the items in batch_user, where the weights are given by attention_unit, and the absence (item_num) does not contribute to the sum because its embedding is a zero vector
            # (batch_size, embed_dim)
            user_emb = torch.matmul(
                weight_matrix.view(batch_size, 1, f_num), 
                self.item_embedding(batch_user)             # (batch_size, f_num, embed_dim)
            ).squeeze(1)
            
            # (batch_size, embed_dim)
            item_emb = self.item_embedding(batch_label.view(-1))
            
            # (batch_size, 1)
            return self.fc_layer(torch.cat([user_emb, item_emb], dim=1))
        else:
            # (batch_size * f_num, embed_dim)
            pre_linear_part_inputs = torch.zeros(
                (batch_size * f_num, self.embed_dim),
                device=batch_user.device,
                dtype=torch.float32
            )

            # (batch_size * effective_index_len, 1)
            attention_unit = self.attention_unit(
                self.item_embedding(batch_user[effective_index]),   # (batch_size * effective_index_len, embed_dim)
                self.item_embedding(effective_labels)               # (batch_size * effective_index_len, embed_dim)
            )

            # (batch_size * effective_index_len, embed_dim) = (batch_size * effective_index_len, embed_dim) x (batch_size * effective_index_len, 1)
            pre_linear_part_inputs[effective_index.view(-1)] = self.item_embedding(batch_user[effective_index]) * attention_unit

            # (batch_size * f_num, embed_dim) -> (batch_size, f_num, embed_dim))
            pre_linear_part_inputs = pre_linear_part_inputs.view(batch_size, f_num, self.embed_dim)

            # (batch_size, len(feature_groups), embed_dim) = (len(feature_groups), f_num) x (batch_size, f_num, embed_dim)
            # (batch_size, len(feature_groups), embed_dim) -> (batch_size, len(feature_groups) * embed_dim)
            user_emb = torch.matmul(self.window_matrix, pre_linear_part_inputs).view(batch_size, -1)
            
            # (batch_size, embed_dim)
            item_emb = self.item_embedding(batch_label.view(-1))

            # (batch_size, 1)
            return self.fc_layer(torch.cat([user_emb, item_emb], -1))


def get_activation(name: str, num_features: int = None, dice_dim: int = None) -> nn.Module:
    """Get the activation function by name.

    :param name: Activation name (case-insensitive).
        Supported: ``relu``, ``gelu``, ``silu`` / ``swish``,
        ``leaky_relu``, ``prelu``, ``tanh``, ``dice``.
    :return: An nn.Module representing the activation function
    """
    name = name.lower().replace("-", "_")
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "prelu":
        return nn.PReLU()
    if name == "tanh":
        return nn.Tanh()
    # if name == "dice":
        # return Dice(num_features, dim=dice_dim)
    raise ValueError(f"Unsupported activation: {name}")

class FullyConnectedLayer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list[int], 
        bias: list[bool], 
        batch_norm: bool = True,
        dropout: float = 0.1, 
        activation: str = 'relu', 
        sigmoid: bool = False
    ):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_dims) >= 1 and len(bias) >= 1, "hidden_size and bias must be non-empty lists"
        assert len(bias) == len(hidden_dims), "bias must have the same length as hidden_size"
        self.sigmoid = sigmoid

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim
        
        self.fc: nn.Sequential = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            return self.output_layer(self.fc(x)) 
        else:
            return self.fc(x)

class LocalActivationUnit(nn.Module):
    def __init__(
        self, 
        hidden_dims: list[int] = [80, 40], 
        bias: list[bool] = [True, True],
        embedding_dim: int = 4, 
        batch_norm: bool = False
    ):
        super(LocalActivationUnit, self).__init__()
        
        self.fc1 = FullyConnectedLayer(
            input_dim=4 * embedding_dim,
            hidden_dims=hidden_dims,
            bias=bias,
            batch_norm=batch_norm,
            activation='relu',
        )

        self.fc2 = FullyConnectedLayer(
            input_dim=hidden_dims[-1],
            hidden_dims=[1],
            bias=[True],
            batch_norm=batch_norm,
            activation='relu',
        )

    def forward(self, user_behavior: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """
        :param user_behavior: (batch_size, f_num, embed_dim)
        :param queries: (batch_size, f_num, embed_dim)
        :return: (batch_size, f_num, 1)
        """
        attention_output = self.fc2(self.fc1(
            torch.cat([
                queries, 
                user_behavior, 
                queries - user_behavior, 
                queries * user_behavior
            ], 
            dim=-1)
        ))
        return attention_output

class Dice(nn.Module):
    """TODO 
    Dice activation function
    :param num_features: number of features
    :param dim: dimension of the feature
    :return: Dice activation function
    """
    def __init__(self, num_features, dim=2):
        self.num_features = num_features
        self.dim = dim
        super(Dice, self).__init__()

        assert dim == 2 or dim == 3
    
        self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        
        if self.dim == 3:
            #self.alpha = torch.zeros((num_features, 1)).cuda()
            self.alpha=nn.Parameter(torch.rand((num_features, 1)))
        elif self.dim == 2:
            #self.alpha = torch.zeros((num_features,)).cuda()
            self.alpha=nn.Parameter(torch.rand((num_features,)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 3:
            # x is [batch_size,time_seq_len,hidden_size]
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        
        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        
        return out

class EmbeddingLayer(nn.Module):
    def __init__(self, item_num: int, embedding_dim: int):
        super(EmbeddingLayer, self).__init__()
        """初始化Embedding Layer
        :param item_num: item number
        :param embedding_dim: embedding dimension
        """
        # 词表 71437 = 71436 真实 item + 1 个 padding id
        self.embed = nn.Embedding(item_num + 1, embedding_dim, padding_idx=item_num)
        # normal weight initialization
        self.embed.weight.data.normal_(0., 0.0001)
        # TODO: regularization
        self.regularization = nn.Dropout(0.1)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.regularization(self.embed(x))
