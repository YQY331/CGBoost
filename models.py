import dgl,torch.nn as nn,torch
class GNN_C(nn.Module):
    def __init__(self,inchannel,classes,gnnblock,modenode=False,hiddensize=128,nlayer=4,p=0.0):
        super(GNN_C,self).__init__()
        self.nlayer=nlayer
        self.conv1=gnnblock(inchannel,hiddensize)
        for layer in range(nlayer-2):
            setattr(self,"conv1_{}".format(layer),gnnblock(hiddensize,hiddensize))
        # self.conv1_1 = gnnblock(hiddensize,hiddensize)
        # self.conv1_2= gnnblock(hiddensize,hiddensize)
        self.conv2=gnnblock(hiddensize,classes)
        self.elu=nn.ELU()
        self.modenode=modenode
        self.dropout=nn.Dropout(p=p)
    def layer_normal(self, x, dim=-1):
        if dim is None:
            std, mean = torch.std_mean(x)
        else:
            std, mean = torch.std_mean(x, dim, keepdim=True)
        return (x - mean) / (std + 1e-05)
    def forward(self, g, ndata):#Normal graph classification and terminal node classification by adding terminal nodes
        with g.local_scope():
            h = self.elu(self.layer_normal(self.conv1(g, ndata)))
            h = self.dropout(h)
            for layer in range(self.nlayer-2):
                h = self.elu(self.layer_normal(getattr(self,"conv1_{}".format(layer))(g, h)))
                h = self.dropout(h)
            # h = self.elu(self.layer_normal(self.conv1_1(g, h)))
            # feature= self.elu(self.layer_normal(self.conv1_2(g, h)))
            feature=h
            h = self.conv2(g, feature)
            g.ndata["h"]=h
            g.ndata["f"]=feature
            if self.modenode is False:
                h=dgl.readout_nodes(g,"h",op="max")#readout operation
            else:#
                gs=dgl.unbatch(g)
                tmph=[]
                for tmpg in gs:
                    tmph.append(tmpg.ndata["h"][-1])
                h=torch.stack(tmph,dim=0)
            h=torch.softmax(h,dim=-1)
            feature=dgl.readout_nodes(g, "f", op="max")
        return h,feature
class GNN(nn.Module):
    def __init__(self,inchannel,classes,gnnblock,modenode=False,hiddensize=128,nlayer=4,p=0.0):
        super(GNN,self).__init__()
        self.nlayer=nlayer
        self.conv1=gnnblock(inchannel,hiddensize)
        for layer in range(nlayer-2):
            setattr(self,"conv1_{}".format(layer),gnnblock(hiddensize,hiddensize))
        self.conv2=gnnblock(hiddensize,classes)
        self.elu=nn.ELU()
        self.modenode=modenode
        self.dropout=nn.Dropout(p=p)
    def layer_normal(self, x, dim=-1):
        if dim is None:
            std, mean = torch.std_mean(x)
        else:
            std, mean = torch.std_mean(x, dim, keepdim=True)
        return (x - mean) / (std + 1e-05)
    def forward(self, g, ndata):#Normal graph classification and terminal node classification by adding terminal nodes
        with g.local_scope():
            h = self.elu(self.layer_normal(self.conv1(g, ndata)))
            h = self.dropout(h)
            for layer in range(self.nlayer-2):
                h = self.elu(self.layer_normal(getattr(self,"conv1_{}".format(layer))(g, h)))
                h = self.dropout(h)
            feature=h
            h = self.conv2(g, feature)
            h=torch.softmax(h,dim=-1)
        return h,feature
class GNN_complex(nn.Module):
    def __init__(self,inchannel,classes,gnnblock,modenode=False,hiddensize=128,nlayer=4,p=0.0):
        super(GNN_complex,self).__init__()
        self.nlayer=nlayer
        setattr(self,"conv{}".format(0),gnnblock(inchannel,hiddensize))
        for layer in range(nlayer)[1:-1]:
            setattr(self,"conv{}".format(layer),gnnblock(hiddensize,hiddensize))
        setattr(self,"conv{}".format(nlayer-1),gnnblock(hiddensize,classes))
        self.elu=nn.ELU()
        self.modenode=modenode
        self.dropout=nn.Dropout(p=p)
    def layer_normal(self, x, dim=None):
        if dim is None:
            std, mean = torch.std_mean(x)
        else:
            std, mean = torch.std_mean(x, dim, keepdim=True)
        return (x - mean) / (std + 1e-05)
    def forward(self, g, ndata):#Normal graph classification and terminal node classification by adding terminal nodes
        with g.local_scope():
            h = self.elu(self.layer_normal(getattr(self,"conv{}".format(0))(g, ndata)))
            h = self.dropout(h)
            for layer in range(self.nlayer)[1:-1]:
                h = self.elu(self.layer_normal(getattr(self,"conv{}".format(layer))(g, h)))
                h = self.dropout(h)
            g.ndata["h"]=h
            h = dgl.readout_nodes(g, "h", op="max")
            tmpg=dgl.knn_graph(h,k=int(len(h)*0.8))
            feature=h
            h =getattr(self,"conv{}".format(self.nlayer-1))(tmpg, feature)
            h=torch.softmax(h,dim=-1)
        return h,feature
class GNN_complex2(nn.Module):
    def __init__(self,inchannel,classes,gnnblock,modenode=False,hiddensize=128,nlayer=4,p=0.0,acfun="sigmoid"):
        super(GNN_complex2,self).__init__()
        self.nlayer=nlayer
        #relation
        setattr(self,"conv{}".format(0),gnnblock(inchannel,hiddensize,acfun=acfun))
        for layer in range(nlayer)[1:-1]:
            setattr(self,"conv{}".format(layer),gnnblock(hiddensize,hiddensize,acfun=acfun))
        setattr(self,"conv{}".format(nlayer-1),gnnblock(hiddensize,classes,acfun=acfun))
        #node
        setattr(self, "conv{}_1".format(0), gnnblock(inchannel, hiddensize,acfun=acfun))
        for layer in range(nlayer)[1:-1]:
            setattr(self, "conv{}_1".format(layer), gnnblock(hiddensize, hiddensize,acfun=acfun))
        setattr(self, "conv{}_1".format(nlayer - 1), gnnblock(hiddensize, classes,acfun=acfun))
        self.elu=nn.ELU()
        self.modenode=modenode
        self.dropout=nn.Dropout(p=p)
        self.weight=nn.Parameter(torch.tensor([0.5]))
    def layer_normal(self, x, dim=None):
        if dim is None:
            std, mean = torch.std_mean(x)
        else:
            std, mean = torch.std_mean(x, dim, keepdim=True)
        return (x - mean) / (std + 1e-05)
    def forward_bak(self, g, ndata):#Normal graph classification and terminal node classification by adding terminal nodes
        with g.local_scope():
            def mapfun(argtuple):
                g,ndata=argtuple
                #node
                g.ndata["feat"]=ndata
                readouts=[]
                # relation first layer pixel result
                h = self.elu(self.layer_normal(getattr(self, "conv{}".format(0))(g, ndata)))
                h = self.dropout(h)
                g.ndata["tmp"] = h
                readout = dgl.readout_nodes(g, "tmp", op="max");readouts.append(readout)
                for index,layer in enumerate(range(self.nlayer)[1:-1]):
                    # relation，pixelInterrelation
                    h = self.elu(self.layer_normal(getattr(self, "conv{}".format(layer))(g, h)))
                    h = self.dropout(h)
                    g.ndata["tmp"] = h
                    readout = dgl.readout_nodes(g, "tmp", op="max");readouts.append(readout)
                # relation，The last layer of pixel relationships
                h = getattr(self, "conv{}".format(self.nlayer - 1))(g, h)
                g.ndata["h"] = h
                h = dgl.readout_nodes(g, "h", op="max");readouts.append(h)
                return readouts
            if g.num_nodes()<2000:
                argtuples=[[g,ndata]]
            else:
                gs=dgl.unbatch(g)
                argtuples=[[g,g.ndata["feat"]] for g in gs]
            from multiprocessing.dummy import Pool
            import multiprocessing as mp
            pool=Pool(mp.cpu_count()-2)
            result=list(pool.map(mapfun,argtuples))
            pool.close()
            eachlayer_readout=zip(*result)
            readouts=[]
            for layer_readout in eachlayer_readout:
                print(f"stack before:{layer_readout[0].shape}")
                readouts.append(torch.cat(layer_readout,dim=0))#readout:1,h,cat:n,h

            # Make a diagram of the slope unit nodes
            onefeat = dgl.readout_nodes(g, "feat", op="mean")
            tmpg = dgl.knn_graph(onefeat, len(onefeat))  # Create sample diagram
            # First layer slope unit results
            h1 = self.elu(self.layer_normal(getattr(self, "conv{}_1".format(0))(tmpg, onefeat)))
            h1 = self.dropout(h1)
            for index,layer in enumerate(range(self.nlayer)[1:-1]):
                #node，Slope unit interrelations
                print(f"readout:{readouts[layer-1].shape},h1:{h1.shape}")
                h1 = self.elu(self.layer_normal(getattr(self, "conv{}_1".format(layer))(tmpg, h1+readouts[layer-1])))
                h1 = self.dropout(h1)
            #node，The relation between the last layer of slope units
            h1 = getattr(self, "conv{}_1".format(self.nlayer - 1))(tmpg, h1+readouts[-2])

            h=self.weight*readouts[-1]+(1-self.weight)*h1
            h=torch.softmax(h,dim=-1)
        return h
    def process_huge_g(self,gs):#todo:Sampling pixels in very large slope units
        import dgl
        glist = dgl.unbatch(gs)
        hugenodeindexs = torch.where(gs.batch_num_nodes() >= 2000)[0]
        for hnindex in hugenodeindexs.cpu().tolist():
            indexs = [i for i in range(glist[hnindex].num_nodes())]  # The ramp unit is too large and the pixels are sampled
            import random
            selectids = random.sample(indexs, 400)
            removeids = list(set(indexs) - set(selectids))
            g = dgl.remove_nodes(glist[hnindex], removeids)
            # g=dgl.rand_graph(400,1000)
            # for key,data in g.ndata.items():
            #     g.ndata[key]=glist[hnindex].ndata[key][selectids]
            glist[hnindex] = g
        gs = dgl.batch(glist)
        return gs
    def forward(self, g, ndata,outpixelpred=False):#Normal graph classification and terminal node classification by adding terminal nodes
        g = self.process_huge_g(g)  # todo:Sampling pixels in very large slope units
        with g.local_scope():
            #node
            if ndata is None:
                ndata = g.ndata["feat"]
            else:
                g.ndata["feat"] = ndata
            onefeat=dgl.readout_nodes(g,"feat",op="mean")
            tmpg=dgl.knn_graph(onefeat,len(onefeat))#Create sample diagram
            #First layer slope unit results
            h1= self.elu(self.layer_normal(getattr(self, "conv{}_1".format(0))(tmpg, onefeat)))
            h1= self.dropout(h1)
            #realtionFirst layer pixel result
            h = self.elu(self.layer_normal(getattr(self,"conv{}".format(0))(g, ndata)))
            h = self.dropout(h)
            g.ndata["tmp"]=h
            readout=dgl.readout_nodes(g,"tmp",op="max")
            for index,layer in enumerate(range(self.nlayer)[1:-1]):
                #node，Slope unit interrelations
                h1 = self.elu(self.layer_normal(getattr(self, "conv{}_1".format(layer))(tmpg, h1+readout)))
                h1 = self.dropout(h1)
                #relation，pixel interrelations
                h = self.elu(self.layer_normal(getattr(self,"conv{}".format(layer))(g, h)))
                h = self.dropout(h)
                g.ndata["tmp"] = h
                readout = dgl.readout_nodes(g, "tmp", op="max")
            #node，The relations between the last layer of slope units
            h1 = getattr(self, "conv{}_1".format(self.nlayer - 1))(tmpg, h1+readout)
            #relation，The last layer of pixel relations
            pixelpred =getattr(self,"conv{}".format(self.nlayer-1))(g, h)
            g.ndata["h"]=pixelpred
            h=dgl.readout_nodes(g,"h",op="max")

            h=self.weight*h+(1-self.weight)*h1
            h=torch.softmax(h,dim=-1)
        if outpixelpred is True:
            return h,torch.softmax(pixelpred,dim=-1)
        else:
            return h
    def pixel_backward(self,g,ndata,pixelalbel=None):
        criterion=nn.CrossEntropyLoss()
        if pixelalbel is None:
            pixelalbel=g.ndata["pixelalbel"].flatten().long()
        pred,pixelpred=self.forward(g,ndata,True)
        loss=criterion(pixelpred,pixelalbel)
        loss.backward()
    def fix_pixelbranch(self):
        for name,data in self.named_parameters():
            if "_" not in name:#pixelbranch parameters
                data.requires_grad=False
    def notfix(self):
        for name,data in self.named_parameters():
            data.requires_grad=True
class GNN_complex2std(nn.Module):
    def __init__(self,inchannel,classes,gnnblock,modenode=False,hiddensize=128,nlayer=4,p=0.0):
        super(GNN_complex2std,self).__init__()
        self.nlayer=nlayer
        #relation
        setattr(self,"conv{}".format(0),gnnblock(inchannel,hiddensize))
        for layer in range(nlayer)[1:-1]:
            setattr(self,"conv{}".format(layer),gnnblock(hiddensize,hiddensize))
        setattr(self,"conv{}".format(nlayer-1),gnnblock(hiddensize,classes))
        #node
        setattr(self, "conv{}_1".format(0), gnnblock(inchannel*2, hiddensize))
        for layer in range(nlayer)[1:-1]:
            setattr(self, "conv{}_1".format(layer), gnnblock(hiddensize, hiddensize))
        setattr(self, "conv{}_1".format(nlayer - 1), gnnblock(hiddensize, classes))
        self.elu=nn.ELU()
        self.modenode=modenode
        self.dropout=nn.Dropout(p=p)
        self.weight=nn.Parameter(torch.tensor([0.5]))
    def layer_normal(self, x, dim=None):
        if dim is None:
            std, mean = torch.std_mean(x)
        else:
            std, mean = torch.std_mean(x, dim, keepdim=True)
        return (x - mean) / (std + 1e-05)
    def process_huge_g(self,gs):#todo:Sampling pixels in very large slope units
        import dgl
        glist = dgl.unbatch(gs)
        hugenodeindexs = torch.where(gs.batch_num_nodes() >= 2000)[0]
        for hnindex in hugenodeindexs.cpu().tolist():
            indexs = [i for i in range(glist[hnindex].num_nodes())]  # The ramp unit is too large and the pixels are sampled
            import random
            selectids = random.sample(indexs, 400)
            removeids = list(set(indexs) - set(selectids))
            g = dgl.remove_nodes(glist[hnindex], removeids)
            # g=dgl.rand_graph(400,1000)
            # for key,data in g.ndata.items():
            #     g.ndata[key]=glist[hnindex].ndata[key][selectids]
            glist[hnindex] = g
        gs = dgl.batch(glist)
        return gs
    def forward(self, g, ndata,outpixelpred=False):#Normal graph classification and terminal node classification by adding terminal nodes
        g = self.process_huge_g(g)  # todo:Sampling pixels in very large slope units
        with g.local_scope():
            #node
            if ndata is None:
                ndata=g.ndata["feat"]
            else:
                g.ndata["feat"]=ndata
            onefeat=dgl.readout_nodes(g,"feat",op="mean")
            from functions import readout_nodes
            import torch
            onefeat_std=readout_nodes(g,"feat","std")
            onefeat=torch.cat([onefeat,onefeat_std],dim=-1)
            tmpg=dgl.knn_graph(onefeat,len(onefeat))#Create sample diagram
            #First layer slope unit results
            h1= self.elu(self.layer_normal(getattr(self, "conv{}_1".format(0))(tmpg, onefeat)))
            h1= self.dropout(h1)
            #realtion First layer pixel result
            h = self.elu(self.layer_normal(getattr(self,"conv{}".format(0))(g, ndata)))
            h = self.dropout(h)
            g.ndata["tmp"]=h
            readout=dgl.readout_nodes(g,"tmp",op="max")
            for index,layer in enumerate(range(self.nlayer)[1:-1]):
                #node，Slope unit interrelations
                h1 = self.elu(self.layer_normal(getattr(self, "conv{}_1".format(layer))(tmpg, h1+readout)))
                h1 = self.dropout(h1)
                #relation，pixel interrelations
                h = self.elu(self.layer_normal(getattr(self,"conv{}".format(layer))(g, h)))
                h = self.dropout(h)
                g.ndata["tmp"] = h
                readout = dgl.readout_nodes(g, "tmp", op="max")
            #node，The relations between the last layer of slope units
            h1 = getattr(self, "conv{}_1".format(self.nlayer - 1))(tmpg, h1+readout)
            #relation，The last layer of pixel relations
            pixelpred =getattr(self,"conv{}".format(self.nlayer-1))(g, h)
            g.ndata["h"]=pixelpred
            h=dgl.readout_nodes(g,"h",op="max")

            h=self.weight*h+(1-self.weight)*h1
            h=torch.softmax(h,dim=-1)
        if outpixelpred is True:
            return h,torch.softmax(pixelpred,dim=-1)
        else:
            return h
    def pixel_backward(self,g,ndata,pixelalbel=None):
        criterion=nn.CrossEntropyLoss()
        if pixelalbel is None:
            pixelalbel=g.ndata["pixelalbel"].flatten().long()
        pred,pixelpred=self.forward(g,ndata,True)
        loss=criterion(pixelpred,pixelalbel)
        loss.backward()
    def fix_pixelbranch(self):
        for name,data in self.named_parameters():
            if "_" not in name:#pixel branch parameters
                data.requires_grad=False
    def notfix(self):
        for name,data in self.named_parameters():
            data.requires_grad=True
class cnn_lstm(nn.Module):
    def __init__(self,inchannel,classes):
        super(cnn_lstm,self).__init__()
        self.conv=nn.Sequential(nn.Upsample(size=[15,15]),nn.Conv2d(inchannel,64,3,1,bias=False),nn.ReLU(),nn.MaxPool2d(2,1),nn.Dropout(0.1),
                                nn.Conv2d(64,128,3,1,bias=False),nn.ReLU(),nn.MaxPool2d(2,1),nn.Dropout(0.1),
                                nn.Conv2d(128,256,3,1,bias=False),nn.ReLU(),nn.MaxPool2d(2,1),nn.Dropout(0.1),
                                nn.Conv2d(256,512,3,1,bias=False),nn.ReLU(),nn.MaxPool2d(2,1),nn.Dropout(0.1))
        self.flattn=nn.Flatten(2,-1)#b,c,1
        self.lstm1=nn.LSTM(input_size=512,hidden_size=256,num_layers=1,batch_first=True,bidirectional=True,bias=False)
        self.dropout2=nn.Dropout(0.2)
        self.lstm2= nn.LSTM(input_size=256*2, hidden_size=128, num_layers=1, batch_first=True,bidirectional=True,bias=False)
        self.mlp=nn.Sequential(nn.Flatten(),nn.Linear(9*128*2,128,bias=False),nn.Linear(128,64,bias=False))
        self.convone=nn.Sequential(nn.Linear(64,classes,bias=False),nn.Softmax(dim=-1))
    def forward(self,x):
        x=self.conv(x.float())
        x=self.flattn(x).permute(0,2,1)#batch,9,512
        x,tmptuple=self.lstm1(x)
        x=self.dropout2(x)
        x,_=self.lstm2(x)
        x=self.dropout2(x)
        feature=self.mlp(x)
        x=self.convone(feature)
        return x,feature
class cnn2d(nn.Module):
    def __init__(self):
        super(cnn2d,self).__init__()
        self.conv=nn.Sequential(nn.Upsample(size=[24,24]),nn.Conv2d(1,20,3,1,bias=False),nn.Dropout(),nn.Upsample(size=[11,11]),
                                nn.Conv2d(20,15,3,1,bias=False),nn.Dropout(),nn.Upsample(size=[4,4]),nn.Flatten(),
                                nn.Linear(15*4*4,78,bias=False))
        self.convone=nn.Sequential(nn.Linear(78,2,bias=False),nn.Softmax(dim=-1))
    def forward(self,x):#m,d
        #onehotEncoding
        import torch.nn.functional as F,torch
        x = torch.clamp(x, 0, 12)#remove the largest and negative values from x
        max_=torch.max(x).int().item()#8
        #print("value max",max_)
        d=x.shape[-1];batch=x.shape[0]#35
        #print("feature dim",d)
        max_=max(max_,d)
        onehotdata=torch.zeros([batch,1,max_,max_],device=x.device)

        x=F.one_hot(x.unsqueeze(dim=1).long(),max_)#batch,1,d,max_
        onehotdata[:,:,:d,:]=onehotdata[:,:,:d,:]+x
        # right=torch.arange(0,max_,device=x.device).reshape(1,1,-1)
        # onehotdata=(x.unsqueeze(dim=-1)==right).float().unsqueeze(dim=1)#m,d,1==1,1,d  m,1,d,d
        # if max_-d>0:
        #     addhotdata=torch.zeros([batch,1,max_-d,max_],device=x.device)
        #     onehotdata=torch.cat([onehotdata,addhotdata],dim=-2)
        #compute
        feature=self.conv(onehotdata)#m,78
        x=self.convone(feature)
        return x,feature
class DNN(nn.Module):
    def __init__(self,inchannel,classes):
        super(DNN,self).__init__()
        self.mlp=nn.Sequential(nn.Linear(inchannel,10),nn.ReLU(),nn.Linear(10,15),nn.ReLU(),nn.Linear(15,6),nn.ReLU()
                               )
        self.convone=nn.Sequential(nn.Linear(6,classes),nn.Softmax(dim=-1))
    def forward(self,x):
        feature=self.mlp(x)
        x=self.convone(feature)
        return x,feature

