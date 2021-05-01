from torch.autograd import grad
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from section_2_models import BasicNet


def Nu_AC1D(u_pred, X ):


    u_x = grad(u_pred,X, create_graph = True, grad_outputs = torch.ones_like(u_pred))[0].view(-1,2)
    u_xx = grad(u_x, X, create_graph = True, grad_outputs = torch.ones_like(u_x))[0][:,0].view(-1,1)

    u_t = grad(u_pred,X, create_graph = True, grad_outputs = torch.ones_like(u_pred))[0][:,1].view(-1,1)

    f = u_t-0.0001*u_xx+5*u_pred**3-5*u_pred

    return f.view(-1,1)







class AC1D_Dataset(Dataset):
    def __init__(self, X):

        self._X = torch.Tensor(X).view(-1,2)


    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):

        sample = self._X[idx]

        return sample



def plot_torch(u):

    u = u.clone().detach().numpy().reshape(-1)

    fig = plt.figure()
    o = plt.plot(u)
    return fig










def fun_u0(X):

    pi = torch.Tensor([np.pi])
    u0 = X**2*torch.cos(pi*X)

    return u0.view(-1,1)



Nt = 100
Nx = 100

xa,xb = -1,1
Tf = 1
Xnp = np.linspace(xa,xb, Nx)
Tnp = np.linspace(0, Tf, Nt)

Xgridnp, Tgridnp = np.meshgrid(Xnp,Tnp)

Xto = torch.Tensor(Xnp).view(-1,1)
To = torch.Tensor(Tnp).view(-1,1)

Xgridto, Tgridto = torch.Tensor(Xgridnp).view(-1,1), torch.Tensor(Tgridnp).view(-1,1)

flat_grid = torch.cat((Xgridto, Tgridto), dim = 1)



zeros_bx = torch.zeros((To.shape[0],2))
zeros_bt = torch.zeros((Xto.shape[0],2))

Xall_left , Xall_right, Xall_0 = zeros_bx.clone(), zeros_bx.clone(), zeros_bt.clone()

Xall_left[:,0], Xall_right[:,0] = xa, xb
Xall_left[:,1], Xall_right[:,1] = To.view(-1),To.view(-1)

Xall_0[:,1] = 0
Xall_0[:,0] = Xto.view(-1)







def train_step(X0, Xleft, Xright, Xbatch, optim, model):

    optim.zero_grad()

    Xbatch = Xbatch.clone().requires_grad_(True)


    u_int = model(Xbatch)

    #u_int.to(device)

    residual = Nu_AC1D(u_int, Xbatch )

    Xleft = Xleft.clone().requires_grad_(True)
    Xright = Xright.clone().requires_grad_(True)
    Uleft = model(Xleft)
    Uright = model(Xright)
    Uxleft = grad(Uleft,Xleft, create_graph = True, grad_outputs = torch.ones_like(Uleft))[0].view(-1,2)[:,0].view(-1,1)
    Uxright = grad(Uright,Xright, create_graph = True, grad_outputs = torch.ones_like(Uright))[0].view(-1,2)[:,0].view(-1,1)


    U0pred = model(X0)
    U0 = fun_u0(X0[:,0])

    loss_int = torch.mean(torch.square(residual))
    loss_b = torch.mean(torch.square(Uleft-Uright))+torch.mean(torch.square(Uxleft-Uxright))

    loss_u0 =torch.mean(torch.square(U0pred-U0))

    loss = loss_int+loss_b+100*loss_u0

    loss.backward()

    optim.step()

    out = {"loss":loss, "loss_int":loss_int,"loss_b":loss_b, "loss_u0":loss_u0}

    return out


def validation(epoch, model, list_outs, fun_u0, save_every = 20, results_dir = "./results/ac1d_test"):

    plt.close("all")
    try:
        os.makedirs(results_dir)
    except Exception as e:
        pass

    with torch.no_grad():

        loss_int = np.mean([list_out["loss_int"].detach().numpy() for list_out in list_outs])
        loss_u0 = np.mean([list_out["loss_u0"].detach().numpy() for list_out in list_outs])
        loss_b = np.mean([list_out["loss_b"].detach().numpy() for list_out in list_outs])

        Xval = np.linspace(-1,1,100)
        Tval = np.linspace(0,1,100)
        Xgrid,Tgrid = np.meshgrid(Xval,Tval)
        Xgrid,Tgrid = torch.Tensor(Xgrid).view(-1,1), torch.Tensor(Tgrid).view(-1,1)
        flat_val = torch.cat((Xgrid,Tgrid),dim = 1)
        out = model(flat_val)
        out = out.view(100,100)


        if epoch%save_every == 0:
            fig_grid = plt.figure()

            p = plt.imshow(out.detach().numpy())
            fig_grid.savefig(os.path.join(results_dir, "ugrid_epoch_{}.png".format(epoch)))

            x0 = torch.linspace(-1,1,100)
            t0 = torch.zeros((100))
            X0 = torch.cat((x0.view(-1,1),t0.view(-1,1)), dim = 1)

            u0pred = model(X0).view(-1)
            u0 = fun_u0(x0).view(-1)

            fig_0 = plt.figure()

            p1 = plt.plot(u0.detach().numpy(), label = "u0")
            p2 = plt.plot(u0pred.detach().numpy(), label = "u0pred")

            fig_0.savefig(os.path.join(results_dir, "u0_epoch_{}.png".format(epoch)))

            ts = [0, 0.25, 0.5, 0.75, 1]

            fig_snaps = plt.figure()
            for t in ts:

                x0 = torch.linspace(-1,1,100)
                t0 = torch.ones((100))*t
                X0 = torch.cat((x0.view(-1,1),t0.view(-1,1)), dim = 1)

                u0pred = model(X0).view(-1)

                p = plt.plot(x0.detach().numpy(), u0pred.detach().numpy())

            fig_snaps.savefig(os.path.join(results_dir, "snaps_epoch_{}.png".format(epoch)))

    return loss_int, loss_u0, loss_b

def train_step_ensemble(X0, Xleft, Xright, Xbatch, optim1,optim2,optime3, model):

    optim1.zero_grad()
    optim2.zero_grad()
    optim3.zero_grad()

    Xbatch = Xbatch.clone().requires_grad_(True)


    u_int = model(Xbatch)

    #u_int.to(device)

    residual = Nu_AC1D(u_int, Xbatch )

    Xleft = Xleft.clone().requires_grad_(True)
    Xright = Xright.clone().requires_grad_(True)
    Uleft = model(Xleft)
    Uright = model(Xright)
    Uxleft = grad(Uleft,Xleft, create_graph = True, grad_outputs = torch.ones_like(Uleft))[0].view(-1,2)[:,0].view(-1,1)
    Uxright = grad(Uright,Xright, create_graph = True, grad_outputs = torch.ones_like(Uright))[0].view(-1,2)[:,0].view(-1,1)


    U0pred = model(X0)
    U0 = fun_u0(X0[:,0])

    loss_int = torch.mean(torch.square(residual))
    loss_b = torch.mean(torch.square(Uleft-Uright))+torch.mean(torch.square(Uxleft-Uxright))

    loss_u0 =torch.mean(torch.square(U0pred-U0))

    loss = loss_int+loss_b+100*loss_u0

    loss.backward()

    optim1.step()
    optim2.step()
    optim3.step()

    out = {"loss":loss, "loss_int":loss_int,"loss_b":loss_b, "loss_u0":loss_u0}

    return out
batch_size = 1000

dataset = AC1D_Dataset(flat_grid)

dataloader = DataLoader(dataset, batch_size = batch_size , shuffle = True)


model = BasicNet(2,1)
from section_2_models import Siren
model = Siren(2,1, first_omega_0 = 6, hidden_layers = 9, hidden_features = 100)

model = SirenEnsemble()
optim1 = torch.optim.Adam(model.siren1.parameters(), lr = 1e-3)
optim2 = torch.optim.Adam(model.siren2.parameters(), lr = 1e-6)
optim3 = torch.optim.Adam(model.siren3.parameters(), lr = 0.5e-4)














Nt = 100
Nx = 100

xa,xb = -1,1
Tf = 1
Xnp = np.linspace(xa,xb, Nx)
Tnp = np.linspace(0, Tf, Nt)

Xgridnp, Tgridnp = np.meshgrid(Xnp,Tnp)

Xto = torch.Tensor(Xnp).view(-1,1)
To = torch.Tensor(Tnp).view(-1,1)

Xgridto, Tgridto = torch.Tensor(Xgridnp).view(-1,1), torch.Tensor(Tgridnp).view(-1,1)

flat_grid = torch.cat((Xgridto, Tgridto), dim = 1)



zeros_bx = torch.zeros((To.shape[0],2))
zeros_bt = torch.zeros((Xto.shape[0],2))

Xall_left , Xall_right, Xall_0 = zeros_bx.clone(), zeros_bx.clone(), zeros_bt.clone()

Xall_left[:,0], Xall_right[:,0] = xa, xb
Xall_left[:,1], Xall_right[:,1] = To.view(-1),To.view(-1)

Xall_0[:,1] = 0
Xall_0[:,0] = Xto.view(-1)


batch_size = 10000

dataset = AC1D_Dataset(flat_grid)

dataloader = DataLoader(dataset, batch_size = batch_size , shuffle = False)

for param in optim1.param_groups:
    param["lr"] = 5e-3

for param in optim2.param_groups:
    param["lr"] = 2e-4

for param in optim3.param_groups:
    param["lr"] = 1e-4

model.eps2 = 1e-2

e0 =  800
epochs = 1000

results_dir = "./results/ac1d_test_ensemble2/"

lis = []
lu0s = []
lbs = []

try:
    os.makedirs(results_dir)
except Exception as e:
    print(e)

save_every = 10
for epoch in range(e0,epochs):

    dataiter = iter(dataloader)

    out_dicts = []


    for batch in tqdm(dataiter):

       # out_dict = train_step(Xall_0, Xall_left, Xall_right, batch, optim, model)
        out_dict = train_step_ensemble(Xall_0, Xall_left, Xall_right, batch, optim1,optim2,optim3, model)
        out_dicts.append(out_dict)



    loss_int, loss_u0, loss_b = validation(epoch, model, out_dicts, fun_u0, save_every = save_every,results_dir = results_dir)

    lis.append(loss_int)
    lu0s.append(loss_u0)
    lbs.append(loss_b)

    plt.close("all")
    if epoch%save_every==0:

        print("\n#### epoch {}   ##".format(epoch))
        print("loss_int", loss_int)
        print("loss_out", loss_u0)
        print("loss_b", loss_b)
        print("\n")


        fig = plt.figure()

        o = plt.plot(lis,label = "l_int")

        fig.savefig(results_dir+"l_int.png")


        fig = plt.figure()
        o = plt.plot(lu0s,label = "lu0")



        fig.savefig(results_dir+"l_u0.png")

        fig = plt.figure()
        o = plt.plot(lbs, label = "lbs")

        fig.savefig(results_dir+"losb.png")
