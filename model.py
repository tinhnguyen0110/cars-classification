import torch.nn as nn
import torch
import timm 
def build_model(num_classes):
    # model = torch.hub.load('pytorch/vision:v0.13.0', 'resnet152', weights="IMAGENET1K_V2")
    
    model = timm.create_model('tf_efficientnetv2_m_in21ft1k', pretrained=False)
    
    model.classifier = nn.Sequential(
                nn.Linear(1280, num_classes))
        
    return model