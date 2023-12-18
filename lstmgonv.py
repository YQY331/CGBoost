import torch.nn as nn
import torch,dgl,math
import dgl.function as fn
import dgl.nn.functional as dglF
from dgl.nn.pytorch.glob import SetTransformerEncoder,SetTransformerDecoder
import dgl.nn.pytorch.conv as conv

class crossgatconv(nn.Module):
    def __init__(self,inchannel,outchannel,nhead=8,acfun="sigmoid"):
        super(crossgatconv,self).__init__()
        self.nhead=nhead if outchannel%8==0 else 1
        self.mapping1=nn.Linear(inchannel,outchannel)
        self.mapping2=nn.Linear(inchannel,outchannel)
        self.res=nn.Linear(inchannel,outchannel)
        self.m=nn.Parameter(torch.ones([3,self.nhead,outchannel//self.nhead]))
        self.dropout=nn.Dropout(p=0.1)
        acfun_dict={"sigmoid":nn.Sigmoid(),"elu":nn.ELU(),"leaky_relu":nn.LeakyReLU(),"relu":nn.ReLU()}
        self.acfun=acfun_dict[acfun]
    def layer_normal(self,x,dim=None):
        if dim is None:
            std,mean=torch.std_mean(x)
        else:
            std,mean=torch.std_mean(x,dim,keepdim=True)
        return (x-mean)/(std+1e-05)
    def minmax_norm(self,x,dim=-1):
        min_,max_=torch.min(x,dim=dim,keepdim=True)[0],torch.max(x,dim=dim,keepdim=True)[0]
        x=(x-min_)/(max_-min_+1e-05)
        return x
    def message_passing(self,edges):
        return {"l":self.acfun(edges.src["l"]+edges.dst["l"]),"r":torch.sigmoid(edges.src["r"]+edges.dst["r"])}
    def forward(self,g,ndata):
        with g.local_scope():
            g=dgl.add_self_loop(g)
            ndata=self.minmax_norm(ndata)
            g.ndata["l"]=self.mapping1(ndata).reshape(g.num_nodes(),self.nhead,-1)
            g.ndata["r"]=self.mapping2(ndata).reshape(g.num_nodes(),self.nhead,-1)
            g.apply_edges(self.message_passing)
            g.edata["l"]=dglF.edge_softmax(g,g.edata["l"])
            g.edata["r"] = dglF.edge_softmax(g, g.edata["r"])
            g.update_all(fn.src_mul_edge("r","l","tmp"),fn.sum("tmp","ft_r"))
            g.update_all(fn.src_mul_edge("ft_r", "r", "tmp"), fn.sum("tmp", "ft_l"))
            h=self.m[0]*g.ndata["ft_l"]
            h=h.reshape(g.num_nodes(),-1)+self.res(ndata)
        return h
class cross2gatconv(nn.Module):
    def __init__(self,inchannel,outchannel,nhead=8):
        super(cross2gatconv,self).__init__()
        self.nhead=nhead if outchannel%8==0 else 1
        self.mapping1=nn.Linear(inchannel,outchannel)
        self.mapping2=nn.Linear(inchannel,outchannel)
        self.res=nn.Linear(inchannel,outchannel)
        self.m=nn.Parameter(torch.ones([3,self.nhead,outchannel//self.nhead]))
        self.dropout=nn.Dropout(p=0.1)
    def layer_normal(self,x,dim=None):
        if dim is None:
            std,mean=torch.std_mean(x)
        else:
            std,mean=torch.std_mean(x,dim,keepdim=True)
        return (x-mean)/(std+1e-05)
    def minmax_norm(self,x,dim=-1):
        min_,max_=torch.min(x,dim=dim,keepdim=True)[0],torch.max(x,dim=dim,keepdim=True)[0]
        x=(x-min_)/(max_-min_+1e-05)
        return x
    def message_passing(self,edges):
        return {"l":torch.sigmoid(edges.src["l"]+edges.dst["l"])}
    def message_passing2(self,edges):
        return {"r":torch.sigmoid(edges.src["r"]+edges.dst["r"])}
    def forward(self,g,ndata):
        with g.local_scope():
            g=dgl.add_self_loop(g)
            ndata=self.minmax_norm(ndata)
            g.ndata["l"]=self.mapping1(ndata).reshape(g.num_nodes(),self.nhead,-1)
            g.ndata["r"]=self.mapping2(ndata).reshape(g.num_nodes(),self.nhead,-1)
            g.apply_edges(self.message_passing)
            g.edata["l"]=dglF.edge_softmax(g,g.edata["l"])
            g.update_all(fn.src_mul_edge("l", "l", "tmp"), fn.sum("tmp", "ft_r"))
            g.apply_edges(self.message_passing2)
            g.edata["r"] = dglF.edge_softmax(g, g.edata["r"])
            g.update_all(fn.src_mul_edge("ft_r", "r", "tmp"), fn.sum("tmp", "ft_l"))
            h=self.m[0]*g.ndata["ft_l"]#+self.m[1]*g.ndata["ft_r"]
            h=h.reshape(g.num_nodes(),-1)+self.res(ndata)
        return h
class ParCgnn(nn.Module):
    def __init__(self,inchannel,outchannel,kernelsize=9):
        super(ParCgnn,self).__init__()
        self.inchannel,self.outchannel,self.kernelsize=inchannel,outchannel,kernelsize
        self.linear=nn.Linear(inchannel,outchannel)
        self.weight=nn.Parameter(torch.randn([outchannel,kernelsize]))
        self.bias=nn.Parameter(torch.randn([outchannel]))
        self.res = nn.Linear(inchannel, outchannel)
    def generate_pe(self,pos_num,d_num):
        pe = torch.zeros([pos_num,d_num])
        pos = torch.arange(0, pos_num).reshape(-1, 1)
        d_i = torch.arange(0, d_num+1, 2).reshape(1, -1)
        pe[:, 0::2] = torch.sin(pos / (torch.pow(10000, d_i /d_num)))[:,:(pe[:, 0::2].shape[-1])]
        pe[:, 1::2] = torch.cos(pos / (torch.pow(10000, d_i / d_num)))[:,:(pe[:, 1::2].shape[-1])]
        return pe
    def reduce_expandweight(self,nodes):
        feature = torch.sort(nodes.mailbox["tmp"],dim=1)[0]  # n,neighbor_num,c
        node_num, neighbor_num = feature.shape[0], feature.shape[1]
        import torch.nn.functional as F
        local_ex_pe=F.interpolate(self.pe.unsqueeze(dim=-1).permute(2,1,0),size=neighbor_num).permute(0,2,1)#1,c,neighbor_num->1,neighbornum,c
        local_weight=F.interpolate(self.weight.unsqueeze(dim=0),size=neighbor_num).permute(0,2,1)#1,c,neighbor_num->1,neighbornum,c
        feature=feature+local_ex_pe
        feature=torch.fft.fft(feature,dim=1)
        local_weight=torch.fft.fft(local_weight,dim=1)
        feature=feature*torch.conj(local_weight)
        feature=torch.fft.ifft(feature,dim=1).real
        feature=feature.sum(dim=1)+self.bias
        return {"h":feature}


    def forward(self,g,ndata):
        self.pe=self.generate_pe(self.kernelsize,self.outchannel).to(ndata.device)#k,c'
        with g.local_scope():
            g=dgl.add_self_loop(g)
            g.ndata["feat"]=self.linear(ndata)
            g.update_all(fn.copy_u("feat","tmp"),self.reduce_expandweight)
            h=g.ndata["h"]+self.res(ndata)
        return h
class gatblock(nn.Module):
    def __init__(self,inchannel,outchannel,acfun="None"):
        super(gatblock,self).__init__()
        if outchannel%8==0:
            self.conv=conv.GATConv(inchannel,outchannel//8,8,feat_drop=0.0,attn_drop=0.0,residual=True,allow_zero_in_degree=True)
        else:
            self.conv = conv.GATConv(inchannel, outchannel,1,residual=True,allow_zero_in_degree=True)
    def forward(self,g,ndata):
        with g.local_scope():
            #g=dgl.add_self_loop(g)
            h=self.conv(g,ndata)
            h=h.reshape(g.num_nodes(),-1)
        return h
class gatv2block(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(gatv2block,self).__init__()
        if outchannel%8==0:
            self.conv=conv.GATv2Conv(inchannel,outchannel//8,8,feat_drop=0.0,attn_drop=0.0,residual=True,allow_zero_in_degree=True)
        else:
            self.conv = conv.GATv2Conv(inchannel, outchannel,1,residual=True,allow_zero_in_degree=True)
    def forward(self,g,ndata):
        with g.local_scope():
            h=self.conv(g,ndata)
            h=h.reshape(g.num_nodes(),-1)
        return h
class gcnblock(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(gcnblock,self).__init__()
        self.conv1=conv.GraphConv(inchannel,outchannel,allow_zero_in_degree=True)
    def forward(self,g,ndata):
        with g.local_scope():
            #g=dgl.add_self_loop(g)
            h=self.conv1(g,ndata)#n,c
        return h
class ginblock(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(ginblock,self).__init__()
        self.lin=nn.Linear(inchannel,outchannel)
        self.conv1=conv.GINConv(self.lin,"mean")
    def forward(self,g,ndata):
        with g.local_scope():
            #g=dgl.add_self_loop(g)
            h=self.conv1(g,ndata)#n,c
        return h
class pnablock(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(pnablock,self).__init__()
        self.conv=conv.PNAConv(inchannel,outchannel,['mean', 'max', 'mean'], ['identity', 'amplification'], 2.5)
    def forward(self,g,ndata):
        with g.local_scope():
            g=dgl.add_self_loop(g)
            h=self.conv(g,ndata)
        return h
