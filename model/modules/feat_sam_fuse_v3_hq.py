import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskCrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4, reduction=2):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim_q // num_heads) ** -0.5
        
        self.q_proj = nn.Conv2d(dim_q, dim_q, kernel_size=1)
        self.k_proj = nn.Conv2d(dim_kv, dim_q, kernel_size=1)
        self.v_proj = nn.Conv2d(dim_kv, dim_q, kernel_size=1)
        self.out_proj = nn.Conv2d(dim_q, dim_q, kernel_size=1)
        
        self.red = reduction
        self.reduction = nn.AvgPool2d(reduction) if self.red > 1 else nn.Identity()
        
    
    def forward(self, x, z, m):
        B, C, H, W = x.shape

        if self.red > 1:
            Hr, Wr = H // self.reduction.kernel_size, W // self.reduction.kernel_size
        else:
            Hr, Wr = H, W
            
        print("input:", m.shape, x.shape, z.shape)
        x = self.reduction(x)
        print("x reduced:", m.shape, x.shape, z.shape)
        z = self.reduction(z)
        print("z reduced:", m.shape, x.shape, z.shape)
        m_resized = F.interpolate(m, size=(Hr, Wr), mode='bilinear', align_corners=False)
        
        q = self.q_proj(x).reshape(B, self.num_heads, C // self.num_heads, Hr * Wr).permute(0, 1, 3, 2)
        k = self.k_proj(z).reshape(B, self.num_heads, C // self.num_heads, Hr * Wr).permute(0, 1, 3, 2)
        v = self.v_proj(z).reshape(B, self.num_heads, C // self.num_heads, Hr * Wr).permute(0, 1, 3, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).permute(0, 1, 3, 2).reshape(B, C, Hr, Wr)
        out = self.out_proj(out)
        
        out = out * m_resized
        return F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

class SAMFuser(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = nn.Conv2d(128, 96, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1)
        self.enc4 = nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=1)
        
        self.attn1 = MaskCrossAttention(96, 96, reduction=2)
        self.attn2 = MaskCrossAttention(192, 192, reduction=4)
        self.attn3 = MaskCrossAttention(384, 384, reduction=8)
        self.attn4 = MaskCrossAttention(768, 768, reduction=8)
        
        self.dec1 = nn.ConvTranspose2d(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(384, 192, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec4 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
        
        self.learnable_weight = nn.Parameter(torch.randn(1, 1, 1, 1))
        
    def forward(self, x, z, m):
        # print(m.shape, x.shape, z[0].shape, z[1].shape, z[2].shape, z[3].shape)
        # torch.Size([5, 1, 256, 448]) torch.Size([5, 128, 60, 108]) torch.Size([5, 96, 64, 112]) torch.Size([5, 192, 32, 56]) torch.Size([5, 384, 16, 28]) torch.Size([5, 768, 8, 14])
        
        # padding x
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = 8
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)
        x_shortcut = x

        print(m.shape, x.shape, z[0].shape, z[1].shape, z[2].shape, z[3].shape)
        
        # Encoding
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.enc3(x2))
        x4 = F.relu(self.enc4(x3))
        
        # Multi-scale Feature Enhancement
        x1 = x1 + self.attn1(x1, z[0], m)
        x2 = x2 + self.attn2(x2, z[1], m)
        x3 = x3 + self.attn3(x3, z[2], m)
        x4 = x4 + self.attn4(x4, z[3], m)
        
        # Decoding
        d1 = F.relu(self.dec1(x4) + x3)
        d2 = F.relu(self.dec2(d1) + x2)
        d3 = F.relu(self.dec3(d2) + x1)
        d4 = self.dec4(d3)
        # d1 = F.relu(self.dec2(x3) + x2)
        # d2 = F.relu(self.dec3(d1) + x1)
        # d3 = self.dec4(d2)
        
        d4 = d4 * self.learnable_weight + x_shortcut
        # d3 = d3 * self.learnable_weight + x_shortcut
        
        # unpadding
        d4 = d4[:, :, :h_ori, :w_ori]
        # d3 = d3[:, :, :h_ori, :w_ori]

        return d4
        # return d3

# # Example Usage
# x = torch.randn(5, 128, 64, 112)
# z = [torch.randn(5, 96, 64, 112), torch.randn(5, 192, 32, 56), torch.randn(5, 384, 16, 28), torch.randn(5, 768, 8, 14)]
# m = torch.randn(5, 1, 64, 112)
# model = SparseUNet()
# output = model(x, z, m)
