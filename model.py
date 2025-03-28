import torch
import torch.nn as nn
from torchvision.models import swin_s, Swin_S_Weights,SwinTransformer
from attention import MultiHeadAttention

class BasicModel(nn.Module):
    def __init__(self,chioce_matrix,embedding_size=[6,113,38,115,124,120,14],num_heads=5):
        super(BasicModel, self).__init__()
        self.chioce_matrix=chioce_matrix
        self.embedding_1 = nn.Embedding(embedding_size[0], 1000)
        self.embedding_2 = nn.Embedding(embedding_size[1], 1000)
        self.embedding_3 = nn.Embedding(embedding_size[2], 1000)
        self.embedding_4 = nn.Embedding(embedding_size[3], 1000)
        self.embedding_5 = nn.Embedding(embedding_size[4], 1000)
        self.embedding_6 = nn.Embedding(embedding_size[5], 1000)
        self.embedding_7 = nn.Embedding(embedding_size[6], 1000)
        self.att1 = MultiHeadAttention(model_dim=1000, num_heads=num_heads)
    #Evacuation_Zone,YearBuilt,EstimatedValue_level
    def forward(self,Evacuation_Zone,YearBuilt,EstimatedValue_level,dist_track_line,dist_track_landfall,wind_mean,flood_mean):
        e1 = self.embedding_1(Evacuation_Zone)
        e2 = self.embedding_2(YearBuilt)
        e3 = self.embedding_3(EstimatedValue_level)
        e4 = self.embedding_4(dist_track_line)
        e5 = self.embedding_5(dist_track_landfall)
        e6 = self.embedding_6(wind_mean)
        e7 = self.embedding_7(flood_mean)
        a_e=torch.cat((e1, e2, e3, e4, e5, e6, e7), dim=1)[:,self.chioce_matrix,:]
        e,_=self.att1(a_e,a_e,a_e)
        e=torch.mean(e,dim=1)
        return e
class FModel(nn.Module):
    def __init__(self,embedding_size=[6,113,38,115,124,120,14],chioce_matrix=[1,1,1,1,0,1,0],num_classes=3,num_heads=5):
        super(FModel, self).__init__()
        if chioce_matrix == None:
            self.chioce_matrix = list(range(0, len(embedding_size)))
        else:
            self.chioce_matrix = []
            for i in range(len(chioce_matrix)):
                if chioce_matrix[i] != 0:
                    self.chioce_matrix.append(i)
        self.embedding=BasicModel(embedding_size=embedding_size,num_heads=num_heads,chioce_matrix=self.chioce_matrix)
        self.linear = nn.Linear(1000, num_classes)
    def forward(self,Evacuation_Zone,YearBuilt,EstimatedValue_level,dist_track_line,dist_track_landfall,wind_mean,flood_mean):
        e=self.embedding(Evacuation_Zone,YearBuilt,EstimatedValue_level,dist_track_line,dist_track_landfall,wind_mean,flood_mean)
        x = self.linear(e)
        return x
'''
class MixModel(nn.Module) means that we use the swin_s model and the FModel to train the model. The FModel aim to fuse the Evacuation_Zone,YearBuilt,EstimatedValue_level.
'''
class MixModel(nn.Module):

    def __init__(self,embedding_size=[6,113,38,115,124,120,14],chioce_matrix=[1,1,1,1,0,1,0],FModel_path=None,num_classes=3,ratio=0.8,pretrain_path="./swin_s-5e29d889.pth",chioce="None",num_heads=5):
        super(MixModel, self).__init__()
        self.swim=swin_s()
        if chioce_matrix==None:
            self.chioce_matrix=list(range(0, len(embedding_size)))
        else:
            self.chioce_matrix=[]
            for i in range(len(chioce_matrix)):
                if chioce_matrix[i]!=0:
                    self.chioce_matrix.append(i)
        self.swim.load_state_dict(torch.load(pretrain_path))
        self.embedding=FModel(embedding_size=embedding_size,chioce_matrix=self.chioce_matrix)
        if FModel_path!=None:
            self.embedding.load_state_dict(torch.load(FModel_path))
        self.dropout = nn.Dropout(p=0.25, inplace=False)
        self.choice=chioce
        self.ratio=ratio
        self.linear = nn.Linear(1000, num_classes)


    def forward(self,img,Evacuation_Zone,YearBuilt,EstimatedValue_level,dist_track_line,dist_track_landfall,wind_mean,flood_mean):
        x = self.swim(img)
        e = self.embedding.embedding(Evacuation_Zone,YearBuilt,EstimatedValue_level,dist_track_line,dist_track_landfall,wind_mean,flood_mean)
        x = x*self.ratio+(1-self.ratio)*e
        x = self.dropout(x)
        x = self.linear(x)
        return x
