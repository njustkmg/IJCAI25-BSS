import torch
import torch.nn as nn
import torch.nn.functional as F

class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
    # def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True): # resnet18/34
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output


class DeepFusion(nn.Module):
    def __init__(self, dim_A, dim_B, num_classes):
        super(DeepFusion, self).__init__()
        self.fc_A = nn.Linear(dim_A, 1024)  # 处理模态A的特征
        self.fc_B = nn.Linear(dim_B, 1024)  # 处理模态B的特征
        self.fc_combined = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, features_A, features_B):
        # 分别处理两个模态的特征
        features_A = self.fc_A(features_A)
        features_B = self.fc_B(features_B)
        # 特征融合
        combined_features = torch.cat([features_A, features_B], dim=1)
        # 分类
        logits = self.fc_combined(combined_features)
        return features_A, features_B, logits

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key, value):
        # 交叉注意力机制
        attn_output, _ = self.attention(query, key, value)
        # 线性层和残差连接
        output = self.norm(self.linear(attn_output) + query)
        return output

class TwoLayerCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, num_classes):
        super().__init__()
        self.layer1 = CrossAttentionLayer(dim, num_heads)
        self.layer2 = CrossAttentionLayer(dim, num_heads)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, features_A, features_B):
        # 假设 features_A 和 features_B 的形状都是 [batch_size, seq_len, dim]
        # 交叉注意力层1，以模态A为query，模态B为key和value
        features_AB = self.layer1(features_A.permute(1, 0, 2), features_B.permute(1, 0, 2), features_B.permute(1, 0, 2))
        # 交叉注意力层2，以模态B为query，更新后的模态A（features_AB）为key和value
        features_BA = self.layer2(features_B.permute(1, 0, 2), features_AB, features_AB)
        # 将注意力输出的特征平均池化作为分类的输入
        pooled_output = features_BA.permute(1, 0, 2).mean(dim=1)
        logits = self.classifier(pooled_output)
        return features_A, features_B, logits


class AdvancedAdapterModule(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=6, dropout_rate=0.5):
        super(AdvancedAdapterModule, self).__init__()
        # 从512维升维到1024
        self.fc_expand = nn.Linear(input_dim, hidden_dim)
        # 再从1024降维回512
        self.fc_compress = nn.Linear(hidden_dim, input_dim)
        # 应用Dropout
        self.dropout = nn.Dropout(dropout_rate)
        # 最终的分类器
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, feature1, feature2):
        # 对两个输入特征求和
        combined_feature = torch.cat((feature1, feature2), dim=1)

        # 升维
        expanded_feature = F.relu(self.fc_expand(combined_feature))
        # 应用Dropout
        expanded_feature = self.dropout(expanded_feature)
        # 降维
        compressed_feature = self.fc_compress(expanded_feature)

        # 应用残差连接，将降维后的特征与原始特征相加
        combined_with_residual = combined_feature + compressed_feature

        # 最终的分类结果
        output = self.classifier(combined_with_residual)
        return output