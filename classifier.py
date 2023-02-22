import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BinaryClassifier(nn.Module):
    def __init__(self, args, feat_dim):
        super(BinaryClassifier, self).__init__()
        
        self.linear1 = nn.Linear(feat_dim, int(feat_dim/2))
        
        self.linear2 = nn.Linear(int(feat_dim/2), 1)
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.sigmoid = nn.Sigmoid()
        

        
    def forward(self, x):
        '''
            x is a matrrix with size m*h  where m is the number of instances and h is the number of features for each instance
        '''
        x = self.linear1(x)
        
        x = self.relu(x)
        
        x = self.dropout(x)
        
        scores = self.linear2(x)

        scores = self.sigmoid(scores) 
        
        # scores = self.dropout(scores)
        
        return scores


class ClassAttentionBinaryClassifier(nn.Module):
    
    def __init__(self, args, feat_dim):
        super(ClassAttentionBinaryClassifier, self).__init__()
        
        self.linear = nn.Linear(feat_dim, 1)
        
        self.w1 = nn.Linear(feat_dim, int(feat_dim/2), bias=False)
        
        self.w2 = nn.Linear(int(feat_dim/2), 1, bias=False)
        
        self.tanh = nn.Tanh()
        
        self.softmax = nn.Softmax(dim=0)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y_sup):
        '''
            x: a tensor with size m*h  where m is the number of instances and h is the number of features for each instance
            y_sup: a list of labels of support examples presented in x. Thus, y_sup.size(0) == x.size(0) 
        '''
        labels = list(set(y_sup))
        
        y_sup = torch.Tensor(y_sup)
        
        rep_labels = []
        
        for label in labels:
            
            mask = y_sup.eq(label)

            x_label = x[mask]
            
            # self-attention (Lin et al., 2017)
            a = self.softmax(self.w2(self.tanh(self.w1(x_label))))
            
            rep_label = a * x_label 
            
            rep_labels.append(rep_label)
            
        rep_lables = torch.cat(rep_labels)
        

        scores = self.linear(rep_lables)
        
        scores = self.sigmoid(scores) 
        
        return scores

class InstanceAttentionBinaryClassifier(nn.Module):
    def __init__(self, args, feat_dim):
        super(InstanceAttentionBinaryClassifier, self).__init__()
        
        self.linear = nn.Linear(feat_dim, 1)
        
        self.num_heads = 1 # args.num_heads
        
        self.atten = nn.MultiheadAttention(feat_dim, self.num_heads)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        '''
            x is a matrrix with size m*h  where m is the number of instances and h is the number of features for each instance
        '''
        
        x = x.view(1, x.size(0), x.size(1))
        
        x, attn_weights = self.atten(query=x, key=x, value=x)
        
        scores = self.linear(x)
        
        scores = self.sigmoid(scores)
        
        scores = scores.view(scores.size(1), scores.size(2))
        
        return scores

if __name__=="__main__":
   
    interplay_matrix =  torch.randn(5, 3) # n=5, h=3
    
       
    classifier = BinaryClassifier(args=None, feat_dim=3)
    y_q_hat = classifier(interplay_matrix)
    print(y_q_hat)
    print(y_q_hat.shape)
    print("="*10)
    
    classifier = InstanceAttentionBinaryClassifier(args=None, feat_dim=3)
    y_q_hat = classifier(interplay_matrix)
    y_q_hat = classifier(interplay_matrix)
    print(y_q_hat)
    print(y_q_hat.shape)
    print("="*10)

    
    classifier = ClassAttentionBinaryClassifier(args=None, feat_dim=3)
    y_q_hat = classifier(interplay_matrix, y_sup = [0,0,1,1,1])
    print(y_q_hat)
    print(y_q_hat.shape)
    print("="*10)
    
    
    
    

