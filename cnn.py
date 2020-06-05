import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
class CNN(nn.Module):

    def __init__(self, latent_dim):
        super(CNN, self).__init__()
        self.latent_dim = latent_dim

        self.embed = nn.Embedding(23, 12)

        #Encode Layer
        self.conv0 = nn.Conv1d(12, 8, 5, padding=2)
        self.conv1 = nn.Conv1d(8, 6, 5, padding=2)#self.conv(30, 15, 5)
        self.conv2 = nn.Conv1d(6, 4, 5, padding=2)#self.conv(15, 8, 5)
        self.conv3 = nn.Conv1d(6, 4, 5, padding=2)#self.conv(8, 4, 5)
        self.conv4 = nn.Conv1d(6, 4, 5, padding=2)

        self.conv_mid = nn.Conv1d(4,4,5,padding=2)

        #Decode Layer
        self.conv5 = nn.Conv1d(4, 8, 5, padding=2)#self.conv(4, 8, 5)
        self.conv6 = nn.Conv1d(8, 16, 5, padding=2)#self.conv(8, 14, 5)
        self.conv7 = nn.Conv1d(16, 23, 5, padding=2)#self.conv(14, 20, 5)
        self.conv8 = nn.Conv1d(18, 23, 5, padding=2)#self.conv(20, 23, 5)
        self.conv9 = nn.Conv1d(20, 23, 5, padding=2)#self.conv(20, 23, 5)

        self.conv_last1 = nn.Conv1d(23, 23, 3, padding=1)
        self.conv_last2 = nn.Conv1d(23, 23, 1, padding=0)

        #self.Max_pool = torch.nn.MaxPool1d(2,return_indices=True)
        self.Avg_pool = torch.nn.AvgPool1d(2)
        self.Max_pool = torch.nn.MaxPool1d(2)

        #self.Latent_avg_pool =  nn.AdaptiveAvgPool1d(self.latent_dim)#nn.AdaptiveMaxPool1d(self.latent_dim)
        self.Latent_max_pool =  nn.AdaptiveMaxPool1d(self.latent_dim)#self.latent_dim)

        self.Up_sample_first = nn.Upsample(124, scale_factor=None, align_corners=None)
        self.Up_sample_mid = nn.Upsample(size=None, scale_factor=2, align_corners=None)
        self.Up_sample_last = nn.Upsample(size=500, scale_factor=None, align_corners=None)

    #self.UnPool = nn.MaxUnpool1d(2, stride=2)
    
    def initialize(self, input_data):
        init_x = self.embed(input_data)
        init_x = torch.transpose(init_x, 1, 2)
        #init_x = rnn.pack_padded_sequence(init_x, valid_elems, enforce_sorted=False, batch_first=True)
        return init_x

    def Encode(self,data):
        x = F.relu(self.conv0(data))
        x = self.Avg_pool(x)
        
        x = F.relu(self.conv1(x))
        x = self.Avg_pool(x)

        x = F.relu(self.conv2(x))
        x = self.Latent_max_pool(x)
        #x = self.Max_pool(x)

        #x = F.relu(self.conv3(x))
        #x = self.Latent_max_pool(x)
        #x = self.Max_pool(x)

        #x = self.conv4(x)
        #x = self.Latent_max_pool(x)
        
        #x = self.conv_mid(x)
        
        return x
  
    def Decode(self,x):
    
        x_con = self.Up_sample_first(x)
        x_con = F.relu(self.conv5(x_con))

        x_con = self.Up_sample_mid(x_con)
        x_con = F.relu(self.conv6(x_con))
        
        #x_con = self.Up_sample_mid(x_con)
        x_con = self.Up_sample_last(x_con)
        x_con = F.relu(self.conv7(x_con))
        
        #x_con = self.Up_sample_last(x_con)
        #x_con = F.relu(self.conv8(x_con))

        #x_con = F.relu(self.conv8(x_con))
        #x_con = self.Up_sample_last(x_con)

        x_con = F.relu(self.conv_last1(x_con))
        x_con = self.conv_last2(x_con)

        return x_con

    def forward(self, data):
        init_data = self.initialize(data)
        x = self.Encode(init_data)
        x_con = self.Decode(x)
        #x_con = torch.sigmoid(x_con)
        return x_con, torch.flatten(x, start_dim=1)

    def save(self, filename):
        args_dict = {
            "latent_dim": self.latent_dim,
        }
        torch.save({
            "state_dict": self.state_dict(),
            "args_dict": args_dict
        }, filename)