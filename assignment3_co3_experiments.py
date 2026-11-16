"""
Assignment 3 CO3 experiment starter
Install: pip install numpy pandas scikit-learn torch
Run: python assignment3_co3_experiments.py
This starter uses controlled synthetic sequence tasks for RNN diagnostics.
For final submission, repeat with faculty-approved PAMAP2/SST/energy datasets.
"""
import time, itertools, random
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

def marker_data(n=4000,T=50):
    X=np.random.randn(n,T,1).astype("float32")*.1
    y=np.random.randint(0,2,n).astype("int64")
    X[:,0,0]=np.where(y==1,1.,-1.)
    return torch.tensor(X),torch.tensor(y)

class RNNModel(nn.Module):
    def __init__(self,kind="rnn",h=32,depth=1):
        super().__init__()
        cls=nn.RNN if kind=="rnn" else nn.LSTM
        self.rnn=cls(1,h,num_layers=depth,batch_first=True)
        self.head=nn.Linear(h,2)
    def forward(self,x):
        out,state=self.rnn(x)
        return self.head(out[:,-1])

def train(model,X,y,epochs=6,lr=.01,clip=1.0):
    dl=DataLoader(TensorDataset(X,y),batch_size=64,shuffle=True)
    model.to(DEVICE); opt=torch.optim.SGD(model.parameters(),lr=lr)
    lossfn=nn.CrossEntropyLoss(); hist=[]; t0=time.perf_counter()
    for ep in range(epochs):
        model.train(); total=0
        for xb,yb in dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            loss=lossfn(model(xb),yb); loss.backward()
            if clip: nn.utils.clip_grad_norm_(model.parameters(),clip)
            opt.step(); total+=loss.item()*len(xb)
        hist.append(total/len(X))
    return hist,time.perf_counter()-t0

def run_memory_experiment():
    rows=[]
    for T in [10,25,50,100,200]:
        X,y=marker_data(T=T)
        for kind in ["rnn","lstm"]:
            m=RNNModel(kind=kind)
            hist,secs=train(m,X,y)
            rows.append({"model":kind,"T":T,"train_loss":hist[-1],"seconds":secs,
                         "parameters":sum(p.numel() for p in m.parameters())})
    pd.DataFrame(rows).to_csv("q1_q2_rnn_results.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))

def run_depth_experiment():
    X,y=marker_data(T=50); rows=[]
    for depth in [1,2,4,8]:
        m=RNNModel("rnn",h=32,depth=depth)
        hist,secs=train(m,X,y,epochs=5,lr=.005)
        rows.append({"depth":depth,"parameters":sum(p.numel() for p in m.parameters()),
                     "loss":hist[-1],"seconds":secs})
    pd.DataFrame(rows).to_csv("q4_depth_results.csv",index=False)

def run_gradient_time_probe():
    X,y=marker_data(n=64,T=100); m=RNNModel().to(DEVICE)
    X=X.to(DEVICE); y=y.to(DEVICE)
    out,state=m.rnn(X); out.retain_grad()
    loss=nn.CrossEntropyLoss()(m.head(out[:,-1]),y); loss.backward()
    norms=out.grad.norm(dim=2).mean(0).detach().cpu().numpy()
    pd.DataFrame({"time_step":np.arange(1,101),"gradient_norm":norms}).to_csv("q1_gradient_by_time.csv",index=False)

def run_search_demo():
    grid=list(itertools.product([1e-4,1e-3,1e-2],[32,64,128],[16,32,64]))
    sample=random.Random(SEED).sample(grid,10)
    pd.DataFrame(sample,columns=["learning_rate","hidden_size","truncation"]).to_csv("q6_random_trials.csv",index=False)
    print("Grid candidates:",len(grid),"Random trials:",len(sample))

if __name__=="__main__":
    print("Device:",DEVICE)
    run_memory_experiment()
    run_depth_experiment()
    run_gradient_time_probe()
    run_search_demo()
    print("Generated CSV evidence files. Use approved repository data for final assignment results.")
