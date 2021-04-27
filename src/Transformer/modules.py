'''
    | CLS1 | Img1 | img2 |
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, n_embs, img_embs, n_hidden, n_head, n_block, vocab_size, dropout, max_len, CLS):
        super(Model, self).__init__()

        # initilize parameter
        self.n_embs = n_embs
        self.img_embs = img_embs
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.n_block = n_block
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout = dropout
        self.CLS = CLS

        self.word_embedding = nn.Embedding(vocab_size, n_embs)
        self.img_project = nn.Sequential(nn.Linear(img_embs, n_hidden), nn.Sigmoid())
        self.position_encoding = PositionalEmb(n_hidden, dropout, 3)
        self.img_encoder = Transformer(n_embs=n_embs, dim_ff=n_hidden, n_head=n_head, n_block=2, dropout=dropout)
        self.decoder = TransformerDecoder(dim=self.n_hidden, dim_ff=self.n_hidden, n_head=n_head, n_block=1, dropout=dropout)
        self.output_layer = nn.Linear(self.n_hidden, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0)

    def encode_img(self, img1, img2):
        img1 = img1.view(img1.size(0), img1.size(1), -1)
        img2 = img2.view(img2.size(0), img2.size(1), -1)
        Img = torch.cat((img1, img2), dim=2)
        Img = Img.transpose(1, 2)
        Img = self.img_project(Img)
        CLS = torch.tensor([[self.CLS]])
        CLS = CLS.repeat(Img.size(0), 1)
        CLS = self.word_embedding(CLS.cuda())
        Img = torch.cat((CLS, Img), dim=1)
        Img = self.position_encoding(Img)
        Img = self.img_encoder(Img)
        G = Img[:, 0, :].unsqueeze(1)
        L = Img[:, 1:, :]
        return G, L

    def encode_text(self, Text):
        batch_size = Text.size(0)
        text_embs = self.word_embedding(Text)
        CLS = torch.mean(text_embs, dim=1, keepdim=True)
        #CLS = torch.LongTensor(batch_size, 1).fill_(self.CLS).cuda()
        #CLS = self.word_embedding(CLS)
        text_embs = torch.cat((CLS, text_embs), dim=1)
        T = self.text_encoder(text_embs)
        CLS_T = T[:, 0, :].unsqueeze(1)
        return CLS_T, T[:, 1:, :]

    def decode(self, Diff, Des, M, mask):
        embs = self.word_embedding(Des)
        embs = torch.cat((Diff, embs), dim=1)
        out = self.decoder(embs, M, mask)
        out = self.output_layer(out[:, 1:, :])
        return out

    def forward(self, Img1, Img2, Cap):
        Diff, Imgs = self.encode_img(Img1, Img2)
        #T, Text = self.encode_text(Cap)

        mask = Variable(subsequent_mask(Cap.size(0), Cap.size(1)), requires_grad=False).cuda()
        T_img = self.decode(Diff, Cap[:,:-1], Imgs, mask=mask)
        #T_mm = self.decode(Diff, Cap[:,:-1], Text, mask=mask)
        #T_text = self.decode(T, Cap[:,:-1], Imgs, mask=mask)
        Cap = Cap.t()
        outs_img = T_img.transpose(0, 1)
        #outs_text = T_text.transpose(0, 1)
        #outs_mm = T_mm.transpose(0, 1)
        loss_img = self.criterion(outs_img.contiguous().view(-1, self.vocab_size), Cap[1:].contiguous().view(-1))
        #loss_text = self.criterion(outs_text.contiguous().view(-1, self.vocab_size), Cap[1:].contiguous().view(-1))
        #loss_mm = self.criterion(outs_mm.contiguous().view(-1, self.vocab_size), Cap[1:].contiguous().view(-1))
        loss = loss_img #+ loss_mm # + loss_mm
        return loss

    def generate(self, Img1, Img2, beam_size=1):
        Diff, Imgs = self.encode_img(Img1.unsqueeze(0), Img2.unsqueeze(0))
        Cap = Variable(torch.ones(1, 1).long()).cuda()
        Cap = self.beam_search(Diff, Imgs, beam_size)
        return Cap.squeeze()

    def beam_search(self, Diff, Imgs, beam_size):
        LENGTH_NORM = True
        batch_size = Imgs.size(0)
        startTokenArray = Variable(torch.ones(batch_size, 1).long()).cuda()
        backVector = torch.LongTensor(beam_size).cuda()
        torch.arange(0, beam_size, out=backVector)
        backVector = backVector.unsqueeze(0).repeat(batch_size, 1)
        backVector = Variable(backVector)

        tokenArange = torch.LongTensor(self.vocab_size).cuda()
        torch.arange(0, self.vocab_size, out=tokenArange)
        tokenArange = Variable(tokenArange)

        beamTokensTable = torch.LongTensor(batch_size, beam_size, self.max_len).fill_(2) # end Token
        beamTokensTable = Variable(beamTokensTable.cuda())
        backIndices = torch.LongTensor(batch_size, beam_size, self.max_len).fill_(-1)
        backIndices = Variable(backIndices.cuda())

        aliveVector = beamTokensTable[:, :, 0].eq(2).unsqueeze(2)

        for i in range(self.max_len-1):
            if i == 0:
                Des = startTokenArray
                out = self.decode(Diff, Des, Imgs, Variable(subsequent_mask(Des.size(0), Des.size(1)+1)).cuda())
                probs = out[:, -1]
                topProbs, topIdx = probs.topk(beam_size, dim=1)
                beamTokensTable[:, :, 0] = topIdx.data
                ProbSums = topProbs
            else:
                Des = beamTokensTable[:, :, :i].squeeze(0)
                out = self.decode(Diff, Des, Imgs.repeat(beam_size, 1, 1), Variable(subsequent_mask(Des.size(0), Des.size(1)+1)).cuda())
                probCurrent = out[:, -1,:].view(batch_size, beam_size, self.vocab_size)
                if LENGTH_NORM:
                    probs = probCurrent * (aliveVector.float() / (i+1))
                    coeff_ = aliveVector.eq(0).float() + (aliveVector.float() * i / (i+1))
                    probs += ProbSums.unsqueeze(2) * coeff_
                else:
                    probs = probCurrent * (aliveVector.float())
                    probs += ProbSums.unsqueeze(2)
                
                mask_ = aliveVector.eq(0).repeat(1, 1, self.vocab_size)
                mask_[:, :, 0] = 0 
                minus_infinity_ = torch.min(probs).item()

                probs.data.masked_fill_(mask_.data, minus_infinity_)
                probs = probs.view(batch_size, -1)

                tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).repeat(batch_size, beam_size, 1)
                tokensArray.masked_fill_(aliveVector.eq(0), 2)
                tokensArray = tokensArray.view(batch_size, -1)
                backIndexArray = backVector.unsqueeze(2).repeat(1, 1, self.vocab_size).view(batch_size, -1)

                topProbs, topIdx = probs.topk(beam_size, dim=1)
                ProbSums = topProbs
                beamTokensTable[:, :, i] = tokensArray.gather(1, topIdx)
                backIndices[:, :, i] = backIndexArray.gather(1, topIdx)

            aliveVector = beamTokensTable[:, :, i:i + 1].ne(2)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = i
            if aliveBeams == 0:
                break

        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data

        RECOVER_TOP_BEAM_ONLY = True
        tokenIdx = finalLen
        backID = backIndices[:, :, tokenIdx]
        tokens = []
        while tokenIdx >= 0:
            tokens.append(beamTokensTable[:, :, tokenIdx].gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, tokenIdx].gather(1, backID)
            tokenIdx = tokenIdx - 1

        tokens.append(startTokenArray.unsqueeze(2).repeat(1, beam_size, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLen = tokens.ne(2).long().sum(dim=2)

        if RECOVER_TOP_BEAM_ONLY:
            tokens = tokens[:, 0]
            seqLen = seqLen[:, 0]
            
        return Variable(tokens)

   
class TransformerDecoder(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout, n_block):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(dim, dim_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(dim)

    def forward(self, x, v, mask):
        for layer in self.layers:
            x = layer(x, v, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attn = SpecialAttention(dim, n_head, dropout)
        self.attn = MultiHeadAttention(dim, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(3)])

    def forward(self, x, v, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, lambda x: self.attn(x, v, v))
        return self.sublayer[2](x, self.feed_forward)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        assert dim % n_head == 0
        self.d_k = dim // n_head
        self.n_head = n_head
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(weights, dim=-1)
        if dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, dim_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class Transformer(nn.Module):
    def __init__(self, n_embs, dim_ff, n_head, dropout, n_block):
        super(Transformer, self).__init__()
        self.norm = LayerNorm(n_embs)
        self.layers = nn.ModuleList([AttnBlock(n_embs, dim_ff, n_head, dropout) for _ in range(n_block)])

    def forward(self, x):
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)
        return x

class AttnBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(AttnBlock, self).__init__()
        self.attn = SpecialAttention(dim, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(2)])

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)

class SpecialAttention(nn.Module):
    def __init__(self, dim, n_head, dropout):
        super(SpecialAttention, self).__init__()
        assert dim % n_head == 0
        self.d_k = dim // n_head
        self.n_head = n_head
        self.q_global = nn.Linear(dim, dim)
        self.k_global = nn.Linear(dim, dim)
        self.v_global = nn.Linear(dim, dim)

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear= nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(weights, dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        V = torch.matmul(attn_weights, value)
        return V

    def forward(self, Q, K, V, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = Q.size(0)
        Q_CLS = Q[:, 0, :].unsqueeze(1)
        Q_Feat = Q[:, 1:, :]
        K_CLS = K[:, 0, :].unsqueeze(1)
        K_Feat = K[:, 1:, :]
        V_CLS = V[:, 0, :].unsqueeze(1)
        V_Feat = V[:, 1:, :]

        CLS_Q = self.q_global(Q_CLS).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        CLS_K = self.k_global(K_CLS).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        CLS_V = self.v_global(V_CLS).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        Feat_Q = self.q_linear(Q_Feat).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        Feat_K = self.k_linear(K_Feat).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        Feat_V = self.v_linear(V_Feat).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        Q = torch.cat((CLS_Q, Feat_Q), dim=2)
        K = torch.cat((CLS_K, Feat_K), dim=2)
        V = torch.cat((CLS_V, Feat_V), dim=2)
        x = self.attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)
        x = self.out_linear(x)
        return x

class SpecialAttention(nn.Module):
    def __init__(self, dim, n_head, dropout):
        super(SpecialAttention, self).__init__()
        assert dim % n_head == 0
        self.d_k = dim // n_head
        self.n_head = n_head
        self.q_global = nn.Linear(dim, dim)
        self.k_global = nn.Linear(dim, dim)
        self.v_global = nn.Linear(dim, dim)

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_linear= nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(weights, dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        V = torch.matmul(attn_weights, value)
        return V

    def forward(self, Q, K, V, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = Q.size(0)
        Q_CLS = Q[:, 0, :].unsqueeze(1)
        Q_Feat = Q[:, 1:, :]
        K_CLS = K[:, 0, :].unsqueeze(1)
        K_Feat = K[:, 1:, :]
        V_CLS = V[:, 0, :].unsqueeze(1)
        V_Feat = V[:, 1:, :]

        CLS_Q = self.q_global(Q_CLS).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        CLS_K = self.k_global(K_CLS).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        CLS_V = self.v_global(V_CLS).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        Feat_Q = self.q_linear(Q_Feat).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        Feat_K = self.k_linear(K_Feat).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        Feat_V = self.v_linear(V_Feat).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
        Q = torch.cat((CLS_Q, Feat_Q), dim=2)
        K = torch.cat((CLS_K, Feat_K), dim=2)
        V = torch.cat((CLS_V, Feat_V), dim=2)
        x = self.attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)
        x = self.out_linear(x)
        return x


def subsequent_mask(batch, size):
    attn_shape = (batch, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    mask = torch.from_numpy(subsequent_mask) == 0
    return mask

class PositionalEmb(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEmb, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.nn.Embedding(max_len, d_model)

    def forward(self, x):
        batchSize = x.size(0)
        patchLen = (x.size(1)-1) // 2
        CLS = torch.LongTensor(batchSize, 1).fill_(0)
        img1 = torch.LongTensor(batchSize, patchLen).fill_(1)
        img2 = torch.LongTensor(batchSize, patchLen).fill_(2)
        img_position = Variable(torch.cat((CLS, img1, img2), dim=-1)).cuda()
        pe_embs = self.pe(img_position)
        x = x + pe_embs
        x = self.dropout(x)
        return x