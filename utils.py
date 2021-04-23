from scipy.interpolate import griddata
import imageio 
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import io
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D  

def fenics_fun_2_grid( fun, mesh, Nx_Ny = None):
    
    points = mesh.coordinates()
    values = fun.compute_vertex_values(mesh)
    
    x0 = mesh.coordinates()[:,0].min()
    x1 = mesh.coordinates()[:,0].max()
    y0 = mesh.coordinates()[:,1].min()
    y1 = mesh.coordinates()[:,1].max()
    
    ratio = (x1-x0)/(y1-y0)
    
    if not(Nx_Ny):
        Ny = int(np.sqrt(len(mesh.coordinates())/ratio))
        Nx = int( len(mesh.coordinates() )/ Ny )
    
    else:
        Nx = Nx_Ny[0]
        Ny = Nx_Ny[1]
    
    X,Y,Z = scatter_2_grid(points, values, (x0,x1), (y0,y1), Nx, Ny)
    
    return X,Y,Z

def fenics_fun_2_grid1D( fun, mesh, Nx = None):
    
    points = mesh.coordinates()
    values = fun.compute_vertex_values(mesh)
    
    x0 = mesh.coordinates()[:,0].min()
    x1 = mesh.coordinates()[:,0].max()

    
    
    if not(Nx):
        Nx = int( len(mesh.coordinates() ) )
    
    else:
        Nx = Nx
    
    X = np.linspace(x0, x1, Nx)
    
    Z = griddata(points, values, (X))
    
    return X,Z


def scatter_2_grid(points, values, x_range, y_range, Nx = 100, Ny = 100):
    """
    
    points array of shape [Npoints, dims]
    values array of shape [Npoints,]
    """
    
    
    x = np.linspace(x_range[0], x_range[1], Nx)
    y = np.linspace(y_range[0], y_range[1], Ny)

    X,Y = np.meshgrid(x,y)



    Z = griddata(points , values, (X,Y))
    
    return X, Y , Z


    
def make_gif(list_ims, save_name, duration = 0.05, size = (200,200)):
    
    with imageio.get_writer(save_name,mode = "I", duration = duration) as writer:
        for sol in list_ims:

            s = sol
            im = ( (s-np.min(s))*(255.0/(np.max(s)-np.min(s))) ).astype(np.uint8)
            im = Image.fromarray(im).resize(size)
            writer.append_data(np.array(im))
    writer.close()
    
def make_gif_1D_arrays(list_arrays, duration = 0.1, name = "default_name", ylim = (-1,1), xlim = (0,1)):
    
    outs = []
    for array in list_arrays:
        plt.close("all")
        fig = plt.figure()
        
        x = np.linspace(0,1,len(array))
        
        p = plt.plot(x,array)
        
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        _array = fig_to_array(fig)
        
        outs.append(_array)
        
    make_gif(outs,name, duration = duration)
    
def make_simulation_gif(eval_sim, real_sim, name, duration = 1, skip_time = ""):
    
    assert np.shape(eval_sim) == np.shape(real_sim), "shapes_not equal"
    
    
    str_time = skip_time
    
    arrays = []
    for i in range(len(eval_sim)):
        
        if skip_time:
        
            str_time = "{} x dt".format(skip_time*i)
            
        im1 = eval_sim[i]
        im2 = real_sim[i]
        plt.close("all")
        fig = plt.figure()
        plt.subplot(121)
        plt.title("Predicted {}".format(str_time),fontdict = {"fontsize":22})
        o = plt.imshow(im1)
        plt.axis('off')
        plt.subplot(122)
        plt.title("Real {}".format(str_time),fontdict = {"fontsize":22})
        o = plt.imshow(im2)
        plt.axis('off')
        fig.tight_layout()
        
        array = fig_to_array(fig)
        arrays.append(array)
        
    make_gif(arrays, name , duration = duration)
    

def eval_sim(model, test_sim):
    

        
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

        test_sim = torch.Tensor(test_sim).to(device)

        H,W = test_sim.shape[-2],test_sim.shape[-1]
        steps = test_sim.shape[0]


        _init = test_sim[0].view(1,1,H,W)

        x_eval = model(_init)

        evals = []
        evals.append(np.array(x_eval.view(H,W).cpu()))

        for i in tqdm(range(1,len(test_sim))):
            
            x_eval = model(x_eval)
            
            evals.append(np.array(x_eval.view(H,W).cpu()))
            
        
        pred = np.array(evals)[:-1]
        real = np.array(test_sim.cpu()).reshape((len(test_sim),H,W))[1:]
        
        first = np.array(test_sim[0].view(1,H,W).cpu())
        
        pred = np.concatenate((first,pred),axis = 0)
        real = np.concatenate((first,real), axis = 0)
        
    return pred, real



def fig_to_array(fig):
    
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw',quality = 95)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    
    return img_arr





def plot_phases(pred_sim, real_sim, results_dir, index = 0, epoch = ""):
    
    try:
        os.makedirs(results_dir)
    except:
        pass
    
    name = "{}_epoch_sim_{}_phases.png".format(epoch,index)
    
    fig = plt.figure()
    

    plt.subplot(121)
    means = [np.mean(np.abs(rs)) for rs in real_sim]
    plt.plot(means, label = "real")

    meansp = [np.mean(np.abs(ps)) for ps in pred_sim]
    plt.plot(meansp, label = "pred")

    plt.legend()

    plt.title("abs mean phase")


    plt.subplot(122)
    means = [np.mean(rs) for rs in real_sim]
    plt.plot(means, label = "real")

    meansp = [np.mean(ps) for ps in pred_sim]
    plt.plot(meansp, label = "pred")

    plt.legend()

    plt.title("abs  phase")
    
    fig.suptitle("epoch {}".format(epoch))
    fig.savefig(os.path.join(results_dir,name))
    
    return fig  

def process_read_logs(file_dir):
    
    data = pd.read_csv(file_dir)
    data = data.replace({np.nan:0})
    groups = data.groupby(d["epoch"])
    data = groups.aggregate("sum")
    
    return data




def plot_2D_comparison(pred, real, X_Y = None):
    
    Zu = real
    Zpred = pred
    
    if not(X_Y):
        Xl = np.linspace(0,1, int(np.sqrt(real.size)) ) 
        Yl = np.linspace(0,1, int(np.sqrt(real.size) ) )
        X,Y = np.meshgrid(Xl,Yl)

    fig = plt.figure()

    fig.set_size_inches(16,16)
    # Plot the surface.

    ax1 = fig.add_subplot(3,2,1, projection = "3d")
    surf = ax1.plot_surface(X, Y, Zu,
                           linewidth=0, )
    ax1.set_title("Analytic solution")

    ax2 = fig.add_subplot(3,2,2, projection = "3d")
    surf = ax2.plot_surface(X, Y, Zpred,
                           linewidth=0,)  

    ax2.set_title("NN solution")



    ax3 = fig.add_subplot(3,2,3)
    plt.imshow(Zu)

    ax4 = fig.add_subplot(3,2,4)
    plt.imshow(Zpred)


    fig.add_subplot(3,2,5)

    ax5 = fig.add_subplot(3,2,5)
    o = ax5.imshow(Zpred-Zu)

    ax5.set_title("Error")
    fig.colorbar(o)
    
    return fig

def _make_grid_plot_2D(Npoints, xa = 0, xb = 1, ya = 0, yb = 1):
    
    Np = Npoints
    _x = np.linspace(xa, xb,Np)
    _y = np.linspace(ya,yb,Np)
    X,Y = np.meshgrid(_x,_y)

    
    return X,Y


        
def plot_2D_comparison_analytical(model_2D, fun_validation,Npoints = 80, xa = -1, xb = 1, ya = -1, yb = 1):
    
    fun_u = fun_validation
    Np = Npoints
    
    X, Y = _make_grid_plot_2D(Np,xa = xa, xb = xb, ya = ya, yb = yb)
    
    _X, _Y = X.reshape(-1), Y.reshape(-1)
    
    Xnp = np.concatenate((_X.reshape((-1,1)), _Y.reshape((-1,1))), axis = 1)

    outu = fun_u(_X,_Y)
    Zreal  = outu.reshape((Np,Np))

    with torch.no_grad():
        Zpred = model_2D(torch.Tensor(Xnp)).detach().numpy()
        Zpred = Zpred.reshape((Np,Np))
        
    fig = plot_2D_comparison(Zpred, Zreal)
    
    return fig