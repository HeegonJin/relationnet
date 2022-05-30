import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

class tf_relation_net(nn.Module):
    def __init__(self, num_models, input_size, encode_size, hidden_size, num_classes, num_layers, num_heads):
        super(tf_relation_net, self).__init__()
        
        # define the properties
        self.input_size = input_size
        self.encode_size = encode_size
        self.hidden_size = hidden_size
        self.num_models = num_models
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_layer = nn.ModuleList([nn.Linear(input_size[i], encode_size) for i in range(self.num_models)]) # intput fully connected layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.encode_size, nhead=self.num_heads, dim_feedforward=self.hidden_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        
        self.cls_token =  nn.Parameter(torch.randn(1, encode_size))
        self.dist_token = nn.Parameter(torch.randn(1, encode_size))
        self.cls_head = nn.Sequential(
            nn.LayerNorm(encode_size),
            nn.Linear(encode_size, self.num_classes)
        )
        self.dist_head = nn.Sequential(
            nn.LayerNorm(encode_size),
            nn.Linear(encode_size, self.num_classes)
        )
    
    def forward(self, inputs):
        B, N, D = inputs.shape[0], self.num_models+2, self.encode_size
        embed_inputs = torch.empty((B, N, D)).to(inputs.device)
        
        cls_tokens = repeat(self.cls_token, '() d -> b d', b = B)
        dist_tokens = repeat(self.dist_token, '() d -> b d', b = B)

        embed_inputs[:, 0] = cls_tokens
        embed_inputs[:, 1] = dist_tokens
        
        idx = 0
        for i_model in range(self.num_models):
            embed_inputs[:, i_model+2] = self.embedding_layer[i_model](inputs[:, idx:idx+self.input_size[i_model]])
            idx += self.input_size[i_model]
        outputs = self.transformer_encoder(embed_inputs)
        cls_output, dist_output = outputs[:, 0], outputs[:, 1]
        cls_output = self.cls_head(cls_output)
        dist_output = self.dist_head(dist_output)
        
        return cls_output, dist_output