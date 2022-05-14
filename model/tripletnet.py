from logging import exception
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

#Copy from https://github.com/avilash/pytorch-siamese-triplet/blob/master/model/net.py
#
class TripletNet(nn.Module):
    def __init__(self, embeddingNet):
        super(TripletNet, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, anchorImage, positiveImage, negativeImage):
        anchor = self.embeddingNet(anchorImage)
        positive = self.embeddingNet(positiveImage)
        negative = self.embeddingNet(negativeImage)
        return anchor, positive, negative 

def get_model(dataset, device):
    embeddingNet = None
    
    if (dataset == ('casia-webface' or 'vggface2')):
        embeddingNet = InceptionResnetV1(dataset).eval()
    else:
        raise Exception('Please enter a valid dataset name : either casia-webface or vggface2')
    
    model = TripletNet(embeddingNet)
    model = model.to(device)
    
    return model
