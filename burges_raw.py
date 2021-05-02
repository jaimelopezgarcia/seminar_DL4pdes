from torch.autograd import grad
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from section_2_models import BasicNet

class BurgersDataset(Dataset):
    def __init__(self, X):

        self._X = torch.Tensor(X).view(-1,2)


    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):

        sample = self._X[idx]

        return sample

def Nu_Burgers(u_pred, X ):

    u_x = grad(u_pred,X, create_graph = True, grad_outputs = torch.ones_like(u_pred))[0].view(-1,2)

    u_xx = grad(u_x, X, create_graph = True, grad_outputs = torch.ones_like(u_x))[0][:,0].view(-1,1)

    u_t = grad(u_pred,X, create_graph = True, grad_outputs = torch.ones_like(u_pred))[0][:,1].view(-1,1)

    pi = torch.Tensor([np.pi])

    u_x = u_x[:,0].view(-1,1)
    f = u_t+u_pred*u_x-(0.01/pi)*u_xx

    return f.view(-1,1)

def plot_torch(u):

    u = u.clone().detach().numpy().reshape(-1)

    fig = plt.figure()
    o = plt.plot(u)
    return fig







Nt = 100
Nx = 100

xa,xb = -1,1
Tf = 1
Xnp = np.linspace(xa,xb, Nx)
Tnp = np.linspace(0, Tf, Nt)

Xgridnp, Tgridnp = np.meshgrid(Xnp,Tnp)



def fun_u0(X):

    pi = torch.Tensor([np.pi])
    u0 = -torch.sin(pi*X)

    return u0.view(-1,1)


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

    residual = Nu_Burgers(u_int, Xbatch )


    Uleft = model(Xleft)
    Uright = model(Xright)
    U0pred = model(X0)
    U0 = fun_u0(X0[:,0])

    loss_int = torch.mean(torch.square(residual))
    loss_b = torch.mean(torch.square(Uleft)+torch.square(Uright))

    loss_u0 =torch.mean(torch.square(U0pred-U0))

    loss = loss_int+loss_b+loss_u0

    loss.backward()

    optim.step()

    out = {"loss":loss, "loss_int":loss_int,"loss_b":loss_b, "loss_u0":loss_u0}

    return out


def validation(epoch, model, list_outs, fun_u0, save_every = 20, results_dir = "./results/burgers_test"):

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


batch_size = 1000

dataset = BurgersDataset(flat_grid)

dataloader = DataLoader(dataset, batch_size = batch_size , shuffle = True)


model = BasicNet(2,1)
optim = torch.optim.Adam(model.parameters(), lr = 1e-3)

epochs = 1000


lis = []
lu0s = []
lbs = []

for epoch in range(epochs):

    dataiter = iter(dataloader)

    out_dicts = []


    for batch in tqdm(dataiter):

        out_dict = train_step(Xall_0, Xall_left, Xall_right, batch, optim, model)
        out_dicts.append(out_dict)



    loss_int, loss_u0, loss_b = validation(epoch, model, out_dicts, fun_u0)

    lis.append(loss_int)
    lu0s.append(loss_u0)
    lbs.append(loss_b)


    if epoch%20==0:

        print("\n#### epoch {}   ##".format(epoch))
        print("loss_int", loss_int)
        print("loss_out", loss_u0)
        print("loss_b", loss_b)
        print("\n")


        fig = plt.figure()

        o = plt.plot(lis,label = "l_int")

        fig.savefig("results/burgers_test/l_int.png")


        fig = plt.figure()
        o = plt.plot(lu0s,label = "lu0")



        fig.savefig("results/burgers_test/l_u0.png")

        fig = plt.figure()
        o = plt.plot(lbs, label = "lbs")

        fig.savefig("results/burgers_test/losb.png")
