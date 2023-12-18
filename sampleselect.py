from dgl.data import DGLDataset
class LSDataset(DGLDataset):
    def __init__(self,tifpath,samplexcelpath=None,possheet=None,negsheet=None,kscale=0.5,addmodenode=False,possheet2=None,negsheet2=None):
        """
        Select samples to create graph data based on positive and negative sample indexes
        :param tifpath:
        :param samplexcelpath:
        :param possheet:
        :param negsheet:
        :param kscale:
        :param addmodenode:
        :param possheet2:
        :param negsheet2:
        """
        super(LSDataset,self).__init__("ok")
        from functions import readTiff
        import pandas as pd,random,numpy as np
        """random_trainpos","random_testpos","random_trainneg","random_testneg","trainneg","testneg","trainpos","testpos"""
        self.kscale=kscale;self.addmodenode=addmodenode
        self.data,self.geotran,self.proj=readTiff(tifpath)
        self.data[-1] = (self.data[-1] > 0) * self.data[-1]  # su>0
        print("slopunit unique",np.unique(self.data[-1]))
        self.slopunit = (1000 * self.data[-1]).astype(np.int)  # h,w
        print(np.unique(self.slopunit))
        if possheet is not None and negsheet is not None:
            if possheet2 is None and negsheet2 is None:
                posindexs=pd.read_excel(samplexcelpath,sheet_name=possheet).values[:,0].tolist()
                pos_label=[1]*len(posindexs)
                negindexs = pd.read_excel(samplexcelpath, sheet_name=negsheet).values[:, 0].tolist()
                neg_lable=[0]*len(negindexs)
                self.sampleindexs=np.array(posindexs+negindexs)#todo:setIt will reorder, it is best not to use it
                self.label=np.array(pos_label+neg_lable)
                indexs=[i for i in range(len(self.sampleindexs))]
                random.shuffle(indexs)#Shuffle excel positive and negative samples
                random.shuffle(indexs)
                self.sampleindexs=self.sampleindexs[indexs]
                self.label = self.label[indexs]
                #uniquesu=np.unique(self.slopunit).tolist()
                # for index in self.sampleindexs.tolist():
                #     if index not in uniquesu:
                #         print(index)
            else:
                sampleindexs = np.unique(self.slopunit)[1:].tolist()
                posindexs = pd.read_excel(samplexcelpath, sheet_name=possheet).values[:, 0].tolist()
                negindexs = pd.read_excel(samplexcelpath, sheet_name=negsheet).values[:, 0].tolist()
                posindexs2 = pd.read_excel(samplexcelpath, sheet_name=possheet2).values[:, 0].tolist()
                negindexs2 = pd.read_excel(samplexcelpath, sheet_name=negsheet2).values[:, 0].tolist()
                otherindexs=list(set(sampleindexs)-set(posindexs)-set(negindexs)-set(posindexs2)-set(negindexs2))
                self.sampleindexs = np.array(otherindexs)
                indexs = [i for i in range(len(self.sampleindexs))]
                random.shuffle(indexs)  # Shuffle excel positive and negative samples
                self.sampleindexs = self.sampleindexs[indexs]
                self.label=None
        else:
            self.sampleindexs=np.unique(self.slopunit)[1:]
            indexs = [i for i in range(len(self.sampleindexs))]
            random.shuffle(indexs)  # Shuffle excel positive and negative samples
            self.sampleindexs = self.sampleindexs[indexs]
            self.label=None#This label is unavailable
    def OutLinkSU(self,SuOutlist,outfullpath):#list or numpy,str
        import numpy as np
        from functions import writeTiff
        result=np.zeros_like(self.slopunit,dtype=np.float)
        slopunit=self.slopunit
        def mapfun(argtuple):
            su, proba=argtuple
            result[slopunit==su]=proba
            print(su)
        from multiprocessing.dummy import Pool
        pool=Pool(processes=10)
        list(pool.map(mapfun,SuOutlist))
        pool.close()
        writeTiff(outfullpath,result,self.geotran,self.proj)
        print("large result write finish")
    def process(self):
        pass
    def __len__(self):
        return len(self.sampleindexs)

    def padd(self,x_min, x_max, width):
        x_width = x_max - x_min
        print(x_width)
        if (15- x_width) % 2 == 0:
            xlp = xrp = (15- x_width) // 2
        else:
            xlp = (15- x_width) // 2
            xrp = xlp + 1
        print(xrp, xlp)
        if x_min - xlp < 0:  # Left to end
            left = x_min;
            right = x_max + (15- x_width)
            print("o", left, right)
        elif x_max + xrp >= width:  # Right to end
            left = x_min - (15- x_width);
            right = x_max
        else:
            left = x_min - xlp;
            right = x_max + xrp
        return left, right
    def __getitem__(self, item):
        import torch,dgl,numpy as np,math
        import torch.nn.functional as F
        from functions import clamp
        su_index=self.sampleindexs[item]#slope unit index
        labels=self.data[-2]
        pixellabels=(self.data[-3]>=0) * (self.data[-3]<=1)*self.data[-3]#The label is neither greater than 1 nor less than 0, and is an integer
        if self.label is not None:
            label=torch.LongTensor([self.label[item]])#1
        else:
            label=torch.mode(torch.tensor(labels[self.slopunit==su_index]).long())

        mask=self.slopunit==su_index
        feat=torch.tensor(clamp(self.data[:35,mask]),dtype=torch.float32).permute(1,0)#m, d, normalized feat
        pixellabels=torch.tensor(pixellabels[mask],dtype=torch.long).reshape(-1,1)#Regularization

        onefeat = torch.mode(feat, dim=0)[0]  # d,Find the mode
        print(self.data.shape,su_index,mask.sum())
        feat=torch.cat([onefeat.unsqueeze(dim=0),feat],dim=0)
        pixellabels=torch.cat([label.unsqueeze(dim=0),pixellabels],dim=0)
        m=len(feat)
        g=dgl.knn_graph(feat,k=math.ceil(self.kscale*m))
        g.ndata["feat"]=feat
        print("feat's shape:",feat.shape)
        print("pixelalbel'shape:",pixellabels.shape)
        g.ndata["pixelalbel"]=pixellabels.float()
        if self.addmodenode is True:
            #Add mode node
            g=dgl.add_nodes(g,1,{"feat":onefeat.unsqueeze(dim=0)})
            g=dgl.add_edges(g,torch.arange(0,m),torch.full([m],fill_value=m).long())

        xs,ys=np.where(mask)
        x_min,x_max=np.min(xs),np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        xl,xr=self.padd(x_min,x_max,self.data.shape[1])
        yl,yr=self.padd(y_min,y_max,self.data.shape[2])
        rectanglefeat=torch.tensor(self.data[:35,xl:xr,yl:yr],dtype=torch.float32).unsqueeze(dim=0)
        gridfeat=F.interpolate(rectanglefeat,size=[16,16]).squeeze(dim=0)
        su_index=torch.LongTensor([su_index])#1
        #print(gridfeat.shape)
        return onefeat,gridfeat,g,label,su_index
class pkldataset(DGLDataset):
    def __init__(self,pklfullpath):
        """
        Read batch data of pkl type
        :param pklfullpath:
        """
        super(pkldataset,self).__init__("ok2")
        import pickle
        with open(pklfullpath,'rb') as f:
            self.testpacks = pickle.load(f)
    def process(self):
        pass
    def __getitem__(self, item):
        onefeats, retangelfeats, gs, labels, slopunits=self.testpacks[item]
        return onefeats.squeeze(dim=0),retangelfeats.squeeze(dim=0),gs,labels.squeeze(dim=0),slopunits.squeeze(dim=0)
    def __len__(self):
        return len(self.testpacks)

