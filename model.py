import torch;from torch import nn
from Utils.utils import upsample

class ReluLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return torch.relu(self.linear(input))

class LeakyReluLayer(nn.Module):
    def __init__(self, in_features, out_features, coef=1e-3, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.LeakyReLU(coef)

    def forward(self, input):
        return self.relu(self.linear(input))
    
class LF_PINE(nn.Module):
    def __init__(self,in_features,x_res,z_res,hidden_features=64,hidden_layers=1,out_features=1):
                
        super().__init__()

        self.z_res = z_res; self.x_res = x_res
        self.in_features = in_features

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((self.z_res,self.x_res,self.x_res,in_features))*2 -1),requires_grad = True)
        
        self.net = []
        self.net.append(LeakyReluLayer(in_features, hidden_features))

        if hidden_layers>0: 
            for i in range(hidden_layers): self.net.append(LeakyReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        self.net.append(nn.LeakyReLU(1e-3))

        self.net = nn.Sequential(*self.net)

    def forward(self, zrange):
        output = self.net(self.table[zrange,...])
        return output

class LF_PINE_c2f(nn.Module):
    def __init__(self,in_features,x_res,coarse_zres,z_res,hidden_features=64,hidden_layers=1,out_features=1):
                
        super().__init__()

        self.z_res = z_res; self.x_res = x_res
        self.in_features = in_features

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((coarse_zres,self.x_res,self.x_res,in_features))*2 -1),requires_grad = True)

        self.finetable = nn.parameter.Parameter(1e-4 * (torch.rand((self.z_res,self.x_res,self.x_res,in_features))*2 -1),requires_grad = True)
        
        self.net = []
        self.net.append(LeakyReluLayer(in_features, hidden_features))

        if hidden_layers>0: 
            for i in range(hidden_layers): self.net.append(LeakyReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        self.net.append(nn.LeakyReLU(1e-3))

        self.net = nn.Sequential(*self.net)

    def coarse2fine(self):
        for i in range(self.in_features):self.finetable[...,i] = nn.parameter.Parameter(upsample(self.table[...,i],dims=3,size=[self.z_res,self.x_res,self.x_res]))

        self.table.detach_()
        self.table.grad = None

    def forward(self, z_start, z_end, isfine=False):
        if isfine: output = self.net(self.finetable[z_start:z_end,...])
        else: output = self.net(self.table[z_start:z_end,...])
        return output
    
class LF_PINE_DAO(nn.Module):
    def __init__(self,hash_table_length,in_features,z_res,hidden_features=64,hidden_layers=1,out_features=1):
                
        super().__init__()

        self.table = nn.parameter.Parameter(1e-4 * (torch.rand((hash_table_length,in_features))*2 -1),requires_grad = True)
        self.z_res = z_res
        self.xy_res = int(hash_table_length/z_res)

        self.net = []
        self.net.append(LeakyReluLayer(in_features, hidden_features))

        if hidden_layers>0: 
            for i in range(hidden_layers): self.net.append(LeakyReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        self.net.append(nn.LeakyReLU(1e-3))

        self.net = nn.Sequential(*self.net)

    def forward(self, z_start, z_end):

        start = z_start*self.xy_res
        end = z_end*self.xy_res

        output = self.net(self.table[start:end,:])
        return output

class ZernikeFitter(nn.Module):
    def __init__(self,polymode,zernike,psf_res):
                
        super().__init__()
        self.zernike_coef = nn.parameter.Parameter(1e-4*(torch.rand(len(polymode),1,1)*2-1),requires_grad = True)
        self.zernike_base = upsample(zernike,size=psf_res).squeeze(0)

    def forward(self):
        phase = torch.sum(self.zernike_coef*self.zernike_base,dim=0)
        return phase