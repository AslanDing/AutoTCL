import torch
import torch.nn as nn
from models.informer.embed import DataEmbedding
from models.informer.attn import ProbAttention,FullAttention,AttentionLayer
from models.informer.encoder import Encoder,EncoderLayer,ConvLayer

class Informer(nn.Module):
    def __init__(self, enc_in=1,d_model=512,
                 factor=5,  n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True):
        super(Informer, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.mapping = nn.Linear(512,feature_size).cuda()


    def forward(self, x_enc,
                enc_self_mask=None):
        emb_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(emb_out, attn_mask=enc_self_mask)
        # mean_pooling = torch.mean(enc_out,1)
        return enc_out
