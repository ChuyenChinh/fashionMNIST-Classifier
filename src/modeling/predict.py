from src.config import BASE_DIR, MODEL_DIR,DATA_DIR
from src.dataset import get_dataloaders
from src.modeling.network import fashionMLP

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

CLASSES  = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
)


def load_trained_model(model_dir,device):
    state_dict = torch.load(model_dir,device)
    model = fashionMLP()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_1_sample(model, img ,device):
    img = img.to(device)
    
    with torch.no_grad():
        out = model(img)
        prob = F.softmax(out,dim=1)
        
        confidence , index = torch.max(prob,dim=1)
    
    return confidence.item() * 100 , CLASSES[index.item()]

def plot(img):
    img = img.squeeze().cpu()
    img = img / 2 + 0.5
    npimg = img.numpy()
    npimg = np.transpose(npimg,(1,2,0))
    plt.imshow(npimg)
    plt.show()
        
def run_demo_prediction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ , testloader = get_dataloaders(DATA_DIR,1)
    model_dir = os.path.join(MODEL_DIR,'fashion_mlp.pth')
    model = load_trained_model(model_dir,device)
    
    #Random 
    sample = iter(testloader)
    img , label = next(sample)
    conf , pred = predict_1_sample(model,img,device)
    
    plot(img)
    print(f"Ground Truth: {CLASSES[label.item()]}")
    print(f"Prediction: {pred}")
    print(f"Confidence {conf:.2f}%")

def evaluate():
    correct = 0
    total = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ , testloader = get_dataloaders(DATA_DIR,1)
    model_dir = os.path.join(MODEL_DIR,'fashion_mlp.pth')
    model = load_trained_model(model_dir,device)
    with torch.no_grad():
        for test_img , test_label in testloader:
            test_img , test_label = test_img.to(device) , test_label.to(device)
            out = model(test_img)
            probs = F.softmax(out,1)
            
            conf , pred = torch.max(probs,dim=1)
            
            total += test_label.shape[0]
            correct += (pred == test_label).sum().item()
    
    return f" Accuracy: {correct / total * 100}"
        