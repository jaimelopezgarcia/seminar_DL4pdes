from scipy.interpolate import griddata
import imageio 
from PIL import Image
import numpy as np

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