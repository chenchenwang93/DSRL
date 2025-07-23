import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads
        # together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-8  # 防止log(0)

    def forward(self, features, labels):
        """
        features: [B, D] 特征矩阵
        labels: [B] 类别标签
        """
        # 计算余弦相似度矩阵
        normalized_features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(normalized_features, normalized_features.T)

        # 构造对比对掩码
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 计算对比损失
        similarity_matrix = similarity_matrix / self.temperature
        log_prob = F.log_softmax(similarity_matrix, dim=1)
        pos_mask = mask - torch.eye(mask.size(0), dtype=torch.float32).to(device)
        pos_mask = pos_mask.float().to(device)

        # 计算每个样本的负采样损失
        loss = -torch.sum(log_prob * pos_mask, dim=1) / pos_mask.sum(dim=1)
        loss = loss.mean()

        return loss


class DSRL(nn.Module):
    def __init__(self, config):
        super(DSRL, self).__init__()
        self.config = config
        self.dim_f = config.dim_f
        self.alpha = config.alpha
        self.lam1 = config.lam1
        self.lam2 = config.lam2
        self.device = config.device

        self.mhatt = MultiHeadAttention(2, self.dim_f, 64, 64, dropout=0.1)    # H=2/4  k=64/56  v=64/56
        # self.softmax = nn.Softmax(dim=-1)
        self.linear_a2x = nn.Linear(1*self.dim_f, self.dim_f)
        self.linear_x2a = nn.Linear(self.dim_f, 1*self.dim_f)
        self.linear_f = nn.Linear(self.dim_f, self.dim_f)
        # self.linear_b2x = nn.Linear(self.dim_f, self.dim_f)
        # self.linear_x2b = nn.Linear(self.dim_f, self.dim_f)
        # self.dropout = nn.Dropout(dropout)
        # self.proj = nn.Linear(self.dk, self.dk)
        # self.layer_norm = nn.LayerNorm(self.dk)
        self.layer_norm1 = nn.LayerNorm(self.dim_f)
        # self.layer_norm2 = nn.LayerNorm(self.dim_f)
        # self.mlp = Mlp(in_features=self.dim_f, hidden_features=1024, act_layer=nn.GELU, drop=0.)
        # self.W_0 = nn.Linear(self.dk, self.dk + self.dim_f)
        # self.layer_norm3 = nn.LayerNorm(self.dk, eps=1e-3)  # eps=1e-3
        # self.batch_norm = nn.BatchNorm1d(self.dk, eps=1e-03, momentum=0.1)
        # self.batch_norm1 = nn.BatchNorm1d(self.dim_f, eps=1e-03, momentum=0.1)
        # self.batch_norm2 = nn.BatchNorm1d(self.dk + self.dim_f, eps=1e-03, momentum=0.1)
        self.contrastive = SupervisedContrastiveLoss(temperature=temp)
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0)  #
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, self.dim_f))
        # self.log_softmax = nn.LogSoftmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_a2x.weight)
        nn.init.xavier_uniform_(self.linear_x2a.weight)
        nn.init.xavier_uniform_(self.linear_f.weight)
        # nn.init.xavier_uniform_(self.linear_b2x.weight)
        # nn.init.xavier_uniform_(self.linear_x2b.weight)
        nn.init.constant_(self.linear_a2x.bias, 0)
        nn.init.constant_(self.linear_x2a.bias, 0)
        nn.init.constant_(self.linear_f.bias, 0)
        # nn.init.constant_(self.linear_b2x.bias, 0)
        # nn.init.constant_(self.linear_x2b.bias, 0)

    def forward(self, batch_label, batch_feature, batch_att_b, batch_att_p):

        f = torch.cat((batch_att_b.reshape(batch_att_b.shape[0], 1, batch_att_b.shape[1]), batch_att_p), 1)
        # batch_att_b = self.layer_norm(batch_att_b)
        # batch_att_p = self.layer_norm1(batch_att_p)
        f = self.layer_norm1(f)
        a, _ = self.mhatt(f, f, f)
        a = self.adaptive_pool(a)
        a = a.squeeze()
        # a = a + batch_att_b
        # a = self.layer_norm1(a)
        # a = a.reshape(a.size(0), a.size(1)*a.size(2))   # concatenate
        # a = a.mean(dim=1).squeeze()
        # a = a / a.norm(dim=-1, keepdim=True)


        # classification loss
        x1 = self.linear_a2x(a)
        a1 = self.linear_x2a(batch_feature)
        # x1b = self.linear_b2x(batch_att_b)
        # b1 = self.linear_x2b(batch_feature)
        # loss1 = torch.norm(batch_feature - x1).pow(2)
        # loss2 = torch.norm(a - a1).pow(2)
        # loss_reg = (loss1 + self.alpha * loss2) / batch_feature.shape[0]
        loss_reg = F.mse_loss(batch_feature, x1, reduction='mean') + self.alpha * F.mse_loss(a, a1, reduction='mean')

        pred = torch.matmul(batch_feature, a.T)
        ground_truth = torch.arange(len(pred), dtype=torch.long, device=self.device)
        # for k in range(len(ground_truth)):
        #     index = np.where(batch_label == batch_label[k])[0]
        #     ground_truth[index] = k
        loss_ce = self.cross_entropy(pred, ground_truth)

        batch_att_pf = batch_att_p.reshape(batch_att_p.size(0) * batch_att_p.size(1), batch_att_p.size(2))
        batch_att_pf = self.linear_f(batch_att_pf)
        labels = torch.repeat_interleave(ground_truth, repeats=batch_att_p.size(1))
        loss_con = self.contrastive(batch_att_pf, labels)
        loss = loss_con + self.lam1 * loss_ce + self.lam2 * loss_reg
        return loss


if __name__ == '__main__':
    pass
