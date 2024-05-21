# %%
import torch
import torch.nn as nn

# %%
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x

# %%
class AdaptiveReweight(nn.Module):
    def __init__(self, channel, reduction=4,momentum=0.1,index=0):
        self.channel=channel
        super(AdaptiveReweight, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LayerNorm([channel // reduction]),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.register_buffer('running_scale', torch.zeros(1))
        self.momentum=momentum
        self.ind=index
        

    def forward(self, x):
        b, c, _, _ = x.size()
        _x=x.view(b,c,-1)
        x_var=_x.var(dim=-1)

        y = self.fc(x_var).view(b, c)

        if self.training:
            scale=x_var.view(-1).mean(dim=-1).sqrt()
            self.running_scale.mul_(1. - self.momentum).add_(scale.data*self.momentum)
        else:
            scale=self.running_scale
        inv=(y/scale).view(b,c,1,1)
        return inv.expand_as(x)*x  
    
class CE(nn.Module):
    def __init__(self, num_features, pooling=False, num_groups=1, num_channels=64, T=3, dim=4, eps=1e-5, momentum=0,
                    *args, **kwargs):
        super(CE, self).__init__()
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.dim = dim

        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
            num_groups)
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features

        self.AR=AdaptiveReweight(num_features)
        self.pool=None
        if pooling:
            self.pool=nn.MaxPool2d(2,stride=2)
    
        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))

        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
        self.x_weight = nn.Parameter(torch.zeros(1))
        print(self.num_channels)

    def forward(self, X):
        N,C,H,W=X.size()
        xin=self.AR(X)
        x_pool=self.pool(X) if self.pool is not None else X
        
        x_pool=x_pool.transpose(0, 1).contiguous().view(self.num_groups, self.num_channels, -1)
        x = X.transpose(0, 1).contiguous().view(self.num_groups, self.num_channels, -1)
        _, d, m = x.size()
        
        if self.training:
            mean = x_pool.mean(-1, keepdim=True)
            
            xc = x_pool - mean
            
            P = [None] * (self.T + 1)
            P[0] = torch.eye(d,device=X.device).expand(self.num_groups, d, d)
            Sigma = torch.baddbmm(alpha=self.eps, input=P[0], beta=1. / m, batch1=xc, batch2=xc.transpose(1, 2))

            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            Sigma_N = Sigma * rTr
            for k in range(self.T):
                mat_power3=torch.matmul(torch.matmul(P[k],P[k]),P[k])
                P[k + 1] = torch.baddbmm(alpha=1.5, input=P[k], beta=-0.5, batch1=mat_power3, batch2=Sigma_N)
            
            wm = P[self.T]  

            self.running_mean.mul_(1. - self.momentum).add_(mean.data*self.momentum)
            self.running_wm.mul_((1. - self.momentum)).add_(self.momentum * wm.data)
        else:
            xc = x - self.running_mean
            wm = self.running_wm

        xn = wm.matmul(x)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()

        x_weight=torch.sigmoid(self.x_weight)
        return x_weight*Xn+(1-x_weight)*xin
    

class WISDM_CE(nn.Module):
    def __init__(self, num_class=6):
        super(WISDM_CE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        # self.relu3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.ce3 = CE(num_features=256, pooling=False, num_channels=256)

        self.fc = nn.Linear(2304, num_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu3(out)
        out = self.pool3(out)
        out = self.ce3(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda(0)
        # out = F.normalize(out.cuda(0))

        return out

# %%

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=in_channels, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

def get_bn(channels):
    return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        if type(kernel_size) == tuple:
            padding = (kernel_size[0] // 2,kernel_size[1] // 2)
        elif type(kernel_size) == int:
            padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        if type(kernel_size) == tuple:
            padding = (kernel_size[0] // 2,kernel_size[1] // 2)
        elif type(kernel_size) == int:
            padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
   
        if type(kernel_size) == tuple:
            padding = (kernel_size[0] // 2,kernel_size[1] // 2)
        elif type(kernel_size) == int:
            padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2 if type(small_kernel) == int else (small_kernel[0]//2,small_kernel[1]//2), groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'): 
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ELK(nn.Module):
    def __init__(self, in_channels, dw_channels, block_lk_size, small_kernel,small_kernel_merged=True):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.large_kernel = ReparamLargeKernelConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                                  stride=1, groups=dw_channels, small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        self.lk_nonlinear = nn.ReLU()
        self.pw2 = conv_bn(dw_channels, dw_channels, 1, 1, 0, groups=1)
        self.prelkb_bn = get_bn(dw_channels)
       

        self.short = nn.Sequential()
        if (in_channels != dw_channels):
            self.short = nn.Sequential(
                 nn.Conv2d(in_channels,dw_channels,1),
                 nn.ReLU(),
            )


    def forward(self, x):
        out = self.pw1(x)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)     
        return  self.short(x) + (out)
    
class ELK_CNN(nn.Module):
    def __init__(self, train_shape, category):
        super(ELK_CNN, self).__init__()
       
        self.layer = nn.Sequential(
            nn.Conv2d(1,64, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            ELK(64,128,(31,1),(3,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,256, (6, 1), (2, 1), (1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.ada_pool = nn.AdaptiveAvgPool2d((1, train_shape[-1]))
        self.fc = nn.Linear(256*train_shape[-1], category)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer(x)
        x = self.ada_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

# %%


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()

        # print(channel_in, channel_out,  kernel, stride, bias,'channel_in, channel_out, kernel, stride, bias')
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(6, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(128),
        )
        self.ca1 = ChannelAttention(128)
        self.sa1 = SpatialAttention()

        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(256),
        )
        self.ca2 = ChannelAttention(256)
        self.sa2 = SpatialAttention()

        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(384),
            nn.ReLU(True)
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(6, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(384),
        )
        self.ca3 = ChannelAttention(384)
        self.sa3 = SpatialAttention()

        self.fc = nn.Sequential(
            nn.Linear(8064, 6)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)
        h1 = self.Block1(x)
        # print(h1.shape)
        r = self.shortcut1(x)
        # print(r.shape)
        h1 = self.ca1(h1) * h1
        h1=  self.sa1(h1) * h1
        h1 = h1 + r
        # print(h1.shape)
        h2 = self.Block2(h1)
        # print(h2.shape)
        r = self.shortcut2(h1)
        # print(r.shape)
        h2 = self.ca2(h2) * h2
        h2 = self.sa2(h2) * h2
        h2 = h2 + r
        # print(h2.shape)
        h3 = self.Block3(h2)
        # print(h3.shape)
        r = self.shortcut3(h2)
        # print(r.shape)
        h3 = self.ca3(h3) * h3
        h3 = self.sa3(h3) * h3
        h3 = h3 + r
        x = h3.view(h3.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        x = nn.LayerNorm(x.size())(x.cpu())
        x = x.cuda()

        return x


# Model

