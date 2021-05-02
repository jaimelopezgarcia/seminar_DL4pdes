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
import skimage.transform

plt.style.use("ggplot")


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
    
    _max = np.max(real_sim)
    _min = np.min(real_sim)
    
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
        o = plt.imshow(im1,cmap='gray', vmin=_min, vmax=_max)
        plt.axis('off')
        plt.subplot(122)
        plt.title("Real {}".format(str_time),fontdict = {"fontsize":22})
        o = plt.imshow(im2,cmap='gray', vmin=_min, vmax=_max)
        plt.axis('off')
        fig.tight_layout()
        
        array = fig_to_array(fig)
        arrays.append(array)
        
    make_gif(arrays, name , duration = duration)
 
def eval_sim_batch(model, test_sim):
    
    model.eval()

    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        test_sim = torch.Tensor(test_sim).to(device)

        H,W = test_sim.shape[-2],test_sim.shape[-1]
        steps = test_sim.shape[1]
        batch = test_sim.shape[0]


        _init = test_sim[:,0].view(batch,1,H,W)

        x_eval = model(_init)

        evals = []
        evals.append(np.array(x_eval.view(batch,H,W).cpu().detach().numpy()))

        for i in tqdm(range(1,steps)):

            x_eval = model(x_eval)

            evals.append(np.array(x_eval.view(batch,H,W).cpu().detach().numpy()) )


        pred = np.array(evals)[:-1].transpose([1,0,2,3])
        real = np.array(test_sim.cpu().detach().numpy()).reshape((batch,steps,H,W))[:,1:,:]

        first = np.array(test_sim[:,0].view(batch,1,H,W).cpu().detach().numpy())

        pred = np.concatenate((first,pred),axis = 1)
        real = np.concatenate((first,real), axis = 1)
        
    return pred, real


def eval_sim(model, test_sim):
    
    
    model.eval()

        
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

        test_sim = torch.Tensor(test_sim).to(device)

        H,W = test_sim.shape[-2],test_sim.shape[-1]
        steps = test_sim.shape[0]


        _init = test_sim[0].view(1,1,H,W)

        x_eval = model(_init)

        evals = []
        evals.append(np.array(x_eval.view(H,W).cpu().detach().numpy()))

        for i in tqdm(range(1,len(test_sim))):
            
            x_eval = model(x_eval)
            
            evals.append(np.array(x_eval.view(H,W).cpu().detach().numpy()) )
            
        
        pred = np.array(evals)[:-1]
        real = np.array(test_sim.cpu().detach().numpy()).reshape((len(test_sim),H,W))[1:]
        
        first = np.array(test_sim[0].view(1,H,W).cpu().detach().numpy())
        
        pred = np.concatenate((first,pred),axis = 0)
        real = np.concatenate((first,real), axis = 0)
        
    return pred, real


def plot_save_error_time(pred_batch,real_batch, skip_time , name = "time_error", results_dir = "./"):
    """
    batch_sim  (Nsims,steps,H,W)
    
    skip time, n times ahead used in training
    """
    
    assert len(np.shape(pred_batch)) == 4
    assert np.shape(pred_batch)==np.shape(real_batch)
    
    error_time = np.mean( np.square(pred_batch-real_batch) ,axis = (0,2,3))
    
    skip_time = int(skip_time)
    
    t = np.arange(0,len(pred_batch[0]),1)*skip_time
    
    fig = plt.figure(figsize = ( 12, 12))
    
    o = plt.plot(t,error_time)
    plt.ylabel("MSE")
    plt.xlabel("time steps")
    plt.title("Prediction error vs time")
    
    fig.savefig(os.path.join(results_dir,name)+".png")
    
    
    return fig

def plot_save_error_time_2_model_comparison(pred_batch1,pred_batch2,real_batch, skip_time , names = ["model1","model2"],name = "time_error", results_dir = "./"):
    """
    batch_sim  (Nsims,steps,H,W)
    
    skip time, n times ahead used in training
    """
    
    assert len(np.shape(pred_batch1)) == 4
    assert np.shape(pred_batch1)==np.shape(real_batch)
    
    error_time1 = np.mean( np.square(pred_batch1-real_batch) ,axis = (0,2,3))
    error_time2 = np.mean( np.square(pred_batch2-real_batch) ,axis = (0,2,3))
    
    skip_time = int(skip_time)
    
    t = np.arange(0,len(pred_batch1[0]),1)*skip_time
    
    fig = plt.figure(figsize = ( 12, 12))
    
    o = plt.plot(t,error_time1, label = names[0])
    o = plt.plot(t,error_time2, label = names[1])
    plt.ylabel("MSE")
    plt.xlabel("time steps")
    plt.title("Prediction error vs time")
    plt.legend()
    
    fig.savefig(os.path.join(results_dir,name)+".png")
    
    
    return fig


def plot_compare_2_models(model1,model2,test_arrays,skip_time,t0_tf=(20,60),results_dir = "./",
                          names = ["model1","model2"]):
    try:
        os.makedirs(results_dir)
    except:
        pass
    p1,r = eval_sim_batch(model1,test_arrays[:,t0_tf[0]:t0_tf[1]:skip_time])
    p2,r = eval_sim_batch(model2,test_arrays[:,t0_tf[0]:t0_tf[1]:skip_time])

    plot_save_error_time_2_model_comparison(p1,p2,r,skip_time,names = names, 
                                            results_dir = results_dir,name = "two_models_time_error")


def make_batch_simulation_gif(batch_sim,name, results_dir = "./", size = (200,200), duration = 0.2):
    """
    batch_sim  (Nsims,steps,H,W)
    """
    
    assert len(np.shape(batch_sim)) == 4
    
    for i in tqdm(range(len(batch_sim))):
        _name = name+"{}.gif".format(i)
        _name = os.path.join(results_dir,_name)
        gif = make_gif(batch_sim[i],_name, size = size,duration = duration)

def fig_to_array(fig):
    
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw',quality = 95)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    
    return img_arr





def plot_phases(pred_sim, real_sim, results_dir, index = 0, epoch = "", name = None):
    
    try:
        os.makedirs(results_dir)
    except:
        pass
    
    
    
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
    
    if not(name):
        _name = "epoch {}".format(epoch)
        name = "{}_epoch_sim_{}_phases.png".format(epoch,index)
    else:
        _name = name
        name_dir = name+".PNG"
    fig.suptitle(_name)
    fig.savefig(os.path.join(results_dir,name_dir))
    
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

def final_model_evaluationAC(model, test_arrays, skip_time, t0 = 20, t0_2 = 2,
                             results_dir = "./results", name = None, skip_gifs_1 = 4,
                          skip_gifs_2 = 1, skip_save_corrupted = 4):
    """
    test_arrays (batch,steps,1,H,W)
    """
    if not(name):
        name = str(model.__class__)[8:-2].replace(".","_")
        
    results_dir = os.path.join(results_dir, name)

    try:
        os.makedirs(results_dir)
    except:
        pass
        
    assert len(np.shape(test_arrays)) == 5, "shape must be (batch,steps,H,W)"
    
    pred, real = eval_sim_batch(model, test_arrays[:,t0::skip_time,...])
    
    o = plot_save_error_time(pred,real,skip_time, name = "time_error",results_dir = results_dir)
    
    make_batch_simulation_gif(pred[::skip_gifs_1],"pred", results_dir = results_dir)
    make_batch_simulation_gif(real[::skip_gifs_1],"real",results_dir = results_dir)
    
    
    for i,(p,r) in enumerate(zip(pred[::skip_gifs_2],real[::skip_gifs_2])):
        make_simulation_gif(p,r, os.path.join(results_dir,"sim_comparative_t0_{}_{}.gif".format(t0,i*skip_gifs_2)),
                            skip_time = skip_time,duration = 0.5)
        plot_phases(p,r,results_dir, name ="phases_{}".format(i*skip_gifs_2))
        
    
    
    pred, real = eval_sim_batch(model, test_arrays[:,t0_2::skip_time,...])##closer to random seed
    for i,(p,r) in enumerate(zip(pred[::skip_gifs_2],real[::skip_gifs_2])):
        make_simulation_gif(p,r, os.path.join(results_dir,"sim_comparative_t0_{}_{}.gif".format(t0_2,i*skip_gifs_2)),
                            skip_time = skip_time,duration = 1)
        
    eval_model_corrupted_input(model,test_arrays,skip_time, results_dir = results_dir, skip_save = skip_save_corrupted)
        
        
        
def eval_model_corrupted_input(model,test_arrays,skip_time,results_dir = "./",skip_save = 1):
    try:
        os.makedirs(results_dir)
    except:
        pass
    for i in tqdm(range(len(test_arrays[::skip_save]))):
        s=i*skip_save
        test_sim = test_arrays[i][20:]
        H,W = np.shape(test_sim)[-2],np.shape(test_sim)[-1]
        test_corrupted =  test_sim+(np.random.random((len(test_sim),1,H,W))-0.5)
        test_resized = skimage.transform.resize(test_sim,(len(test_sim),1,int(H/2),int(W/2)))
        ind1 = np.random.randint(0,H,(1000,1000))
        test_sampled = np.zeros((len(test_sim),1,H,W))
        test_sampled[...,ind1[:,0],ind1[:,1]] = test_sim[...,ind1[:,0],ind1[:,1]]

        p0,r0 = eval_sim(model, test_sim[::skip_time])
        p1,r1 = eval_sim(model,test_corrupted[::skip_time])
        p2,r2 = eval_sim(model,test_resized[::skip_time])
        p3,r3 = eval_sim(model,test_sampled[::skip_time])

        names = [
            "{}_vanilla.gif".format(s),
            "{}_corrupted.gif".format(s),
            "{}_downsampled.gif".format(s),
            "{}_sampled.gif".format(s)
        ]

        names = [os.path.join(results_dir,name) for name in names]
        make_simulation_gif(p0,r0,names[0])
        make_simulation_gif(p1,r1,names[1])
        make_simulation_gif(p2,r2,names[2])
        make_simulation_gif(p3,r3,names[3])