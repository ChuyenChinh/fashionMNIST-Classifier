from src.dataset import get_dataloaders
from src.modeling.network import fashionMLP
from src.config import BASE_DIR,DATA_DIR,MODEL_DIR,BATCH_SIZE,LEARNING_RATE,EPOCHS
import torch
import torch.nn as nn
import os

def train_loop():
    train_loader, test_loader = get_dataloaders(DATA_DIR,BATCH_SIZE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fashionMLP()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)
    
    for i in range(EPOCHS):
        acc_loss = 0.0
        model.train()
        for train_X , train_y in train_loader:
            train_X , train_y = train_X.to(device) , train_y.to(device)
            pred = model(train_X)
            loss = criterion(pred,train_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc_loss += loss.item()
        
        print(f"epoch{i+1} loss = {acc_loss/len(train_loader)}")
    
    save_path = os.path.join(MODEL_DIR, 'fashion_mlp.pth')
    torch.save(model.state_dict(), save_path)
    
    

if __name__ == '__main__':
    train_loop()