from torch.utils.data import DataLoader
from dataset import dataset
from model import BERTClass
import argparse
from loss import Loss
import torch
from tqdm import trange
from utils import save_model,load_model

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str,default = False, 
                    help='Name of the model')

parser.add_argument('--epochs', type=int,default = 25, 
                    help='Number of epochs')

parser.add_argument('--batch_size', type=int,default = 25, 
                    help='batch_size')

parser.add_argument('--lr', type=int,default = 2e-5, 
                    help='name of the model')
parser.add_argument('--csv_file', type=str,default = 'DBPEDIA_train.csv' , 
                    help='csv_dataset')

def train(csv_file,batch_size,lr,model_name,epochs):
    train_set = dataset(csv_file)
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BERTClass().to(device)
    opt = torch.optim.Adam(model.parameters(),lr= lr,weight_decay = 5e-4)
    if model_name:
        try:
            load_model(model = model,optimizer = opt,file_name = model_name)
        except:
            print('{model_name} not found!')
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
        save_model(model,opt)
    print('Model trained successfully!')

if __name__ == '__main__':
    args = parser.parse_args()
    train(
    csv_file = args.csv_file,
    lr = args.lr,
    batch_size = args.batch_size,
    model_name = args.model_name,
    epochs = args.epochs
        )
