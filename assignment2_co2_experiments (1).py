"""
Deep Learning Assignment 2 – CO2 experiment starter
Install:
  pip install numpy pandas scikit-learn matplotlib torch torchvision
Run:
  python assignment2_co2_experiments.py

HIGGS: place HIGGS.csv.gz in this folder (first column label, next 28 features).
Fashion-MNIST: torchvision downloads it automatically.
All numeric results are generated locally; no values are fabricated.
"""
import os, time, itertools, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import make_classification
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

SEED=42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

def load_higgs(max_rows=100000):
    path="HIGGS.csv.gz"
    if not os.path.exists(path):
        raise FileNotFoundError("Download HIGGS.csv.gz from the UCI HIGGS dataset and place it beside this script.")
    df=pd.read_csv(path,header=None,compression="gzip",nrows=max_rows)
    y=df.iloc[:,0].astype("int64").to_numpy()
    X=df.iloc[:,1:].astype("float32").to_numpy()
    return X,y

def make_mlp(input_dim, width, depth, activation):
    acts={"sigmoid":nn.Sigmoid,"tanh":nn.Tanh,"relu":nn.ReLU}
    layers=[]; d=input_dim
    for _ in range(depth):
        layers += [nn.Linear(d,width),acts[activation]()]
        d=width
    layers += [nn.Linear(d,2)]
    model=nn.Sequential(*layers)
    for m in model.modules():
        if isinstance(m,nn.Linear):
            if activation=="relu": nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)
            else: nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    return model

def train_torch(model, Xtr,ytr,Xv,yv,epochs=8,lr=.001,batch=128,optimizer="sgd",capture_grads=False):
    model.to(DEVICE)
    ds=TensorDataset(torch.tensor(Xtr,dtype=torch.float32),torch.tensor(ytr,dtype=torch.long))
    dl=DataLoader(ds,batch_size=batch,shuffle=True)
    opt=torch.optim.SGD(model.parameters(),lr=lr) if optimizer=="sgd" else torch.optim.Adam(model.parameters(),lr=lr)
    lossfn=nn.CrossEntropyLoss()
    history=[]; grad_history=[]
    for ep in range(epochs):
        model.train(); total=0
        for xb,yb in dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            loss=lossfn(model(xb),yb); loss.backward()
            if capture_grads:
                grad_history.append([float(m.weight.grad.norm().detach().cpu()) for m in model.modules() if isinstance(m,nn.Linear) and m.weight.grad is not None])
            opt.step(); total += loss.item()*len(xb)
        model.eval()
        with torch.no_grad():
            xv=torch.tensor(Xv,dtype=torch.float32).to(DEVICE); yvt=torch.tensor(yv,dtype=torch.long).to(DEVICE)
            vl=lossfn(model(xv),yvt).item()
        history.append((total/len(ds),vl))
    return history,grad_history

def q1_activation_depth():
    X,y=load_higgs(20000)
    Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr).astype("float32"); Xv=sc.transform(Xv).astype("float32")
    rows=[]
    for act in ["sigmoid","tanh","relu"]:
        for depth in [2,6,12]:
            model=make_mlp(Xtr.shape[1],128,depth,act)
            t=time.perf_counter(); hist,gr=train_torch(model,Xtr,ytr,Xv,yv,epochs=5,capture_grads=True); elapsed=time.perf_counter()-t
            rows.append({"activation":act,"depth":depth,"train_loss":hist[-1][0],"val_loss":hist[-1][1],"seconds":elapsed,
                         "gradient_norms":str(np.mean(np.array(gr),axis=0).tolist()) if gr else ""})
    pd.DataFrame(rows).to_csv("q1_activation_results.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))

def q2_sgd():
    X,y=load_higgs(100000)
    rows=[]
    for n in [1000,10000,100000]:
        xx,yy=X[:n],y[:n]
        Xtr,Xv,ytr,yv=train_test_split(xx,yy,test_size=.2,random_state=SEED,stratify=yy)
        sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr).astype("float32"); Xv=sc.transform(Xv).astype("float32")
        for lr in [.0001,.001,.01,.05]:
            for bs in [32,128,512]:
                model=make_mlp(Xtr.shape[1],128,3,"relu")
                t=time.perf_counter()
                hist,_=train_torch(model,Xtr,ytr,Xv,yv,epochs=8,lr=lr,batch=bs,optimizer="sgd")
                elapsed=time.perf_counter()-t
                for ep,(tl,vl) in enumerate(hist,1):
                    rows.append({"n":n,"lr":lr,"batch":bs,"epoch":ep,"train_loss":tl,"val_loss":vl,"elapsed_s":elapsed})
    pd.DataFrame(rows).to_csv("q2_sgd_results.csv",index=False)

# Manual backpropagation: one hidden layer sigmoid network for gradient-check demonstration.
def manual_forward_backward(X,y,W1,b1,W2,b2):
    sig=lambda z:1/(1+np.exp(-np.clip(z,-40,40)))
    a1=sig(X@W1+b1); logits=a1@W2+b2
    logits=logits-logits.max(axis=1,keepdims=True)
    probs=np.exp(logits); probs/=probs.sum(axis=1,keepdims=True)
    Y=np.eye(2)[y]; m=len(y)
    loss=-np.sum(Y*np.log(probs+1e-12))/m
    dz2=(probs-Y)/m; dW2=a1.T@dz2; db2=dz2.sum(axis=0)
    dz1=(dz2@W2.T)*a1*(1-a1); dW1=X.T@dz1; db1=dz1.sum(axis=0)
    return loss,(dW1,db1,dW2,db2)
def q3_gradient_check():
    rng=np.random.default_rng(SEED); X=rng.normal(size=(8,4)); y=rng.integers(0,2,8)
    W1=rng.normal(scale=.1,size=(4,5)); b1=np.zeros(5); W2=rng.normal(scale=.1,size=(5,2)); b2=np.zeros(2)
    loss,grads=manual_forward_backward(X,y,W1,b1,W2,b2)
    params=[W1,b1,W2,b2]; eps=1e-5; errors=[]
    for p,g in zip(params,grads):
        for ix in np.ndindex(p.shape):
            old=p[ix]; p[ix]=old+eps; lp=manual_forward_backward(X,y,*params)[0]
            p[ix]=old-eps; lm=manual_forward_backward(X,y,*params)[0]
            p[ix]=old; gn=(lp-lm)/(2*eps); errors.append(abs(gn-g[ix]))
    print("Manual backprop gradient-check max absolute error:",max(errors))

def fashion_data(ntrain=5000):
    tf=transforms.ToTensor()
    train=datasets.FashionMNIST("./fashion_data",train=True,download=True,transform=tf)
    test=datasets.FashionMNIST("./fashion_data",train=False,download=True,transform=tf)
    ids=np.random.default_rng(SEED).choice(len(train),ntrain,replace=False)
    X=torch.stack([train[i][0] for i in ids]); y=torch.tensor([train[i][1] for i in ids])
    Xt=torch.stack([test[i][0] for i in range(len(test))]); yt=torch.tensor([test[i][1] for i in range(len(test))])
    return X,y,Xt,yt

class BaselineCNN(nn.Module):
    def __init__(self,gap=False):
        super().__init__(); self.gap=gap
        self.features=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                    nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.head=(nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,10)) if gap
                   else nn.Sequential(nn.Flatten(),nn.Linear(64*7*7,128),nn.ReLU(),nn.Linear(128,10)))
    def forward(self,x): return self.head(self.features(x))

def train_cnn(model,X,y,Xt,yt,epochs=5,batch=128):
    ds=TensorDataset(X,y); dl=DataLoader(ds,batch_size=batch,shuffle=True)
    model.to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=.001); lossfn=nn.CrossEntropyLoss()
    rows=[]; t=time.perf_counter()
    for ep in range(epochs):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            loss=lossfn(model(xb),yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits=model(Xt.to(DEVICE)); vl=lossfn(logits,yt.to(DEVICE)).item()
            acc=(logits.argmax(1)==yt.to(DEVICE)).float().mean().item()
        rows.append({"epoch":ep+1,"val_loss":vl,"test_accuracy":acc})
    return rows,time.perf_counter()-t

def q5_q7_cnn():
    X,y,Xt,yt=fashion_data(5000)
    outputs=[]
    for gap in [False,True]:
        model=BaselineCNN(gap=gap)
        params=sum(p.numel() for p in model.parameters())
        hist,elapsed=train_cnn(model,X,y,Xt,yt)
        outputs.append({"model":"GAP" if gap else "FlattenDense","parameters":params,"seconds":elapsed,
                        "best_val_loss":min(r["val_loss"] for r in hist),
                        "best_test_accuracy":max(r["test_accuracy"] for r in hist)})
    pd.DataFrame(outputs).to_csv("q5_q7_cnn_comparison.csv",index=False)
    print(pd.DataFrame(outputs).to_string(index=False))

def q6_grid_growth():
    spaces=[
      {"depth":[2,3],"filters":[16,32],"kernel":[3,5]},
      {"depth":[2,3,4],"filters":[16,32,64],"kernel":[3,5],"dropout":[0,.25]},
      {"depth":[2,3,4,5],"filters":[16,32,64,128],"kernel":[3,5,7],"dropout":[0,.2,.4],"lr":[.001,.0001]}
    ]
    rows=[]
    for i,s in enumerate(spaces,1):
        keys=list(s); candidates=list(itertools.product(*[s[k] for k in keys]))
        rows.append({"problem_size":i,"hyperparameters":len(keys),"candidate_count":len(candidates),
                     "example_first_candidate":str(dict(zip(keys,candidates[0])))})
    # correctness check: exhaustive best feasible on a small deterministic toy objective
    toy=[{"id":i,"mem":m,"score":sc} for i,(m,sc) in enumerate([(2,.7),(5,.9),(8,.95),(3,.8)])]
    feasible=[c for c in toy if c["mem"]<=5]
    best=max(feasible,key=lambda z:z["score"])
    assert best["id"]==1
    print("Toy exhaustive-search correctness check passed; best feasible:",best)
    pd.DataFrame(rows).to_csv("q6_grid_growth.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":
    print("Device:",DEVICE)
    q3_gradient_check()
    # HIGGS-dependent experiments require HIGGS.csv.gz in this directory.
    if os.path.exists("HIGGS.csv.gz"):
        q1_activation_depth()
        q2_sgd()
    else:
        print("Skipped Q1/Q2: add HIGGS.csv.gz from UCI to this folder.")
    q5_q7_cnn()
    q6_grid_growth()
    print("Finished. See generated CSV files for measured evidence.")
