import torch

def save_model(model,optimizer):
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, 'text_weights.pth.tar')

def load_model(model,optimizer,file_name='text_weights.pth.tar'):
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
