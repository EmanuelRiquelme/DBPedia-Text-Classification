from torch.utils.data import DataLoader
from dataset import dataset
from model import BERTClass
from loss import Loss
import torch
from tqdm import trange
if __name__ == '__main__':
    train_set = dataset(csv_file_name ='archive (1)/DBPEDIA_train.csv')
    batch_size= 26
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = BERTClass().to('cuda:0')
    opt = torch.optim.Adam(model.parameters(),lr= 2e-5,weight_decay = 5e-4)
    device = 'cuda:0'
    epochs = 10
    loss_fn = Loss()
    for epoch in (t := trange(epochs)):
        it = iter(train_set)
        temp_loss = []
        for _ in range(len(train_set)):
            input,target = next(it)
            input,target = input.to(device),target.to(device)
            output = model(input)
            loss = loss_fn(output,target)
            temp_loss.append(loss.item())
            loss.backward()
            opt.step()
        temp_loss = sum(temp_loss)/len(temp_loss)
        t.set_description(f'loss : {temp_loss:.2f}')
