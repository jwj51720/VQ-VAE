import torch
import torch.nn as nn


class VQ(nn.Module):
    def __init__(self, D, B=128, K=512, beta=0.25):
        super(VQ, self).__init__()
        self.beta = beta  # commitment
        self.k = K  # Embedding size
        self.d = D  # Embedding dimension (256 * 32 * 32)
        self.b = B  # Batch size
        self.codebook = nn.Embedding(self.d, self.k)
        self.embedding = self.codebook.weight

    def forward(self, input):
        # flatten 된 Encoder의 Output이 입력된다는 전제 (B, D = WHC)
        assert input.shape[0] == self.b

        # Euclidean dist (L2): sqrt(A^2 + B^2 - 2AB)
        A_2 = torch.sum(input**2, dim=1, keepdim=True)  # [B, 1]
        B_2 = torch.sum(self.embedding**2, dim=0, keepdim=True)  # [1, K]
        AB = torch.matmul(input, self.embedding)  # [B, K]
        dist = torch.sqrt(A_2 + B_2 - 2 * AB)

        # 최소 dist 지점을 선택 -> codebook에 접근할 idx
        idx = torch.argmin(dist, dim=1)  # [B]
        vq = self.embedding.t()[idx]  # [B, D]

        # loss 2번, 3번 항 계산
        loss = 0

        # z = 32 x 32 x 1 형태로 변환
        vq = vq.view(self.b, -1, 32, 32)  # [B, 256, 32, 32]

        return vq, loss
