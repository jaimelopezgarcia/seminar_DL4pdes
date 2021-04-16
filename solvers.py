import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import scipy  
from tqdm import tqdm
from utils import fenics_fun_2_grid
import os
from utils import make_gif
from fenics import *
import imageio

def eval_gaussians(x,y, stds, mus, amps = None):
    
    fun_eval = 0.0
    for i in range(len(stds)):
        std = stds[i]
        mu = mus[i]
        
        if not(isinstance(amps,np.ndarray) or isinstance(amps,list)):
            amp = 1.0
        else:
            amp = amps[i]
        fun_eval = fun_eval + amp*16*x*(1-x)*y*(1-y)*np.exp(-0.5*(x-mu[0])**2/std[0]**2-0.5*(y-mu[1])**2/std[1]**2)
    
    return fun_eval

def eval_rectangles(x,y, origins, shapes , amps  = None):
    
    fun_eval = 0.0
    for i in range(len(origins)):
        shape = shapes[i]
        origin = origins[i]
        
        if not(isinstance(amps,np.ndarray) or isinstance(amps,list)):
            amp = 1.0
        else:
            amp = amps[i]
        
        val = 0.0
        
        if (x>origin[0] and x<(origin[0]+shape[0]) ):
            if (y>origin[1] and y<(origin[1]+shape[1])):
                val = amp
            
        fun_eval = fun_eval + val
    
    return fun_eval



def eval_image(x,y, image):
    
    image = image.T
    
    dimx, dimy = np.shape(image)
    
    coordx = int(np.floor(dimx*x))
    coordy = int(np.floor(dimy*y))
    
    if coordx == dimx:
        coordx = coordx-1
    if coordy == dimy:
        coordy = coordy-1
    
    val = image[coordx,coordy]
    
    return val








########## ALLEN CAHN ###############

class InitialConditionsAC(UserExpression):
    
    def __init__(self, mode = "random"):
        super().__init__()
        self._mode = mode
        
        self._image = imageio.read("./data/cage_meme.jpg").get_data(0)
        self._image = (self._image/np.max(self._image)-0.5)*1.8


    def eval(self, values, x):
        
        if self._mode == "random":
            values[0] =  0.02*(0.5 - np.random.random())
        elif self._mode == "image":
            values[0] =  eval_image(x[0],x[1],self._image)
            
        elif self._mode == "fourier_high_freq":
            values[0] = 0.02*np.cos(np.pi*2*7*x[0])*np.cos(np.pi*2*4*x[1])-0.02*np.cos(np.pi*2*5*x[0])*np.cos(np.pi*2*8*x[1])+0.01*np.cos(np.pi*2*2*x[0])*np.cos(np.pi*2*6*x[1])\
                        +0.02*np.cos(np.pi*2*9*x[0])*np.cos(np.pi*2*9*x[1])
            
        elif self._mode == "fourier_low_freq":
            values[0] = 0.02*np.cos(np.pi*2*2*x[0])*np.cos(np.pi*2*4*x[1])-0.02*np.cos(np.pi*2*3*x[0])*np.cos(np.pi*2*1*x[1])+0.01*np.cos(np.pi*2*5*x[0])*np.cos(np.pi*2*2*x[1])\
                        +0.02*np.cos(np.pi*2*3*x[0])*np.cos(np.pi*2*3*x[1])






class PeriodicBoundary(SubDomain):
    
    def __init__(self, L = 1.0):
        super().__init__()
        self._L = L
    
    def inside(self, x, on_boundary):
        
        L = self._L
        sides = bool(x[0] < DOLFIN_EPS and (x[0]-L) < DOLFIN_EPS and on_boundary)
        updown = bool(x[1] < DOLFIN_EPS and (x[1]-L) < DOLFIN_EPS and on_boundary)

        return sides or updown

    def map(self, x, y):

        L = self._L

        if near(x[0], L) and near(x[1], L):
            y[0] = x[0] - L
            y[1] = x[1] - L
        elif near(x[0], L):
            y[0] = x[0] - L
            y[1] = x[1]
        else:  
            y[0] = x[0]
            y[1] = x[1] - L







def solve_allen_cahn(eps = 0.01, n_elements = 60,T_dt = 200, ratio_speed = 10 ,initial_conditions = "random", dtype_out = np.float32):
    
    t0 = 100 # iterations to solve with smaller dt
    
    dt0 = eps*2*1e-3
    
    ratio_speed = ratio_speed #increasing step when evolution when interfaces are formed
    
    _dt = dt0*ratio_speed

    dt = Constant(dt0)
    
    mesh = UnitSquareMesh(n_elements, n_elements)
    V= FunctionSpace(mesh, "P", 1,constrained_domain=PeriodicBoundary())

    u = Function(V)  
    v = TestFunction(V)
    u_n = Function(V)


    # Create intial conditions and interpolate
    u_init = InitialConditionsAC(initial_conditions)
    u_n.interpolate(u_init)



    F = (u*v-u_n*v+dt*dot(grad(u),grad(v))+dt*(1/eps**2)*(u**2-1)*u*v)*dx


    sols_first = []
    
    sols = []
    for i in tqdm(range(T_dt+t0)):
        
        if i>t0:
            
            dt.assign(_dt) #increasing dt after initial decomposition

        solve(F == 0, u)



        u_n.assign(u)

        X,Y,Z = fenics_fun_2_grid(u,mesh)
        
        if i>t0:
            sols.append(Z)
        else:
            sols_first.append(Z)
            
    sols_first = np.array(sols_first).astype(dtype_out)[::ratio_speed]
    sols = np.array(sols).astype(dtype_out)
    
    sols = np.concatenate((sols_first,sols), axis = 0)
    
    return sols







def post_process_and_save_ac(sols, results_dir = "./data/AC", name_simulation = "AC_sim_default_name", save_plots = False):
    
    try:
        os.makedirs(results_dir)
    except:
        pass
    


    np.save(os.path.join(results_dir, name_simulation+"_array"), sols)
    
    if save_plots:
        
        make_gif(sols,os.path.join(results_dir, name_simulation+"gif"+".gif") ) 

        amps = []
        for sol in sols:
            amps.append(np.sum(np.abs(sol)))
        amps = np.array(amps)/np.size(sol)

        ampsn = []
        for sol in sols:
            ampsn.append(np.sum(sol))
        ampsn = np.array(ampsn)/np.size(sol)


        fig = plt.figure(figsize = (15,15))

        plt.plot(np.arange(0,len(amps),1),amps, label = "avg abs phase")
        plt.plot(np.arange(0,len(ampsn),1),np.abs(ampsn), label = "abs avg phase")

        plt.legend()
        plt.savefig(os.path.join(results_dir,name_simulation+"avg_phases_time.png"))
        
        
        
        
        
        
        
        
        
        
        
###### HEAT EQUATION ########



class InitialConditionsHE(UserExpression):
    
    def __init__(self, mode = "rectangles", n_elements_max = 5, L = 1.0):
        
        super().__init__()
        
        self._image = imageio.read("./data/cage_meme.jpg").get_data(0)
        self._image = self._image/np.max(self._image)
        
        n_elements = np.random.randint(1,n_elements_max)
        self._mode = mode
        widths = np.random.random(size =(n_elements))*0.3+0.05
        heights = np.random.random(size =(n_elements))*0.3+0.05
        self._amps = 0.5*(np.random.random(size = (n_elements))-0.5)+1
        self._shapes = [(width,height) for (width,height) in zip(widths,heights)]
        
        xs = np.random.random(size = (n_elements))*(1-0.36)
        ys = np.random.random(size = (n_elements))*(1-0.36)
        self._origins = [(x,y) for (x,y) in zip(xs,ys)]
        
        
        
        
        stdxs = np.random.random(size =(n_elements))*0.1+0.05
        stdys = np.random.random(size =(n_elements))*0.1+0.05
        self._stds = [(stdx, stdy) for (stdx,stdy) in zip(stdxs,stdys)]
        xs = np.random.random(size = (n_elements))
        ys = np.random.random(size = (n_elements))
        self._mus = [(x,y) for (x,y) in zip(xs,ys)]
        


    def eval(self, values, x):
        
        if self._mode == "rectangles":
            values[0] =  eval_rectangles(x[0],x[1], self._origins, self._shapes, self._amps)
        elif self._mode == "gaussians":
            values[0] = eval_gaussians(x[0],x[1], self._stds, self._mus, self._amps)
            
        elif self._mode == "image":
            values[0] = eval_image(x[0],x[1],self._image)
            
        elif self._mode == "fourier_high_freq":
            values[0] = 0.4*np.cos(np.pi*2*7*x[0])*np.cos(np.pi*2*4*x[1])-0.04*np.cos(np.pi*2*5*x[0])*np.cos(np.pi*2*8*x[1])+0.3*np.cos(np.pi*2*2*x[0])*np.cos(np.pi*2*6*x[1])\
                        +0.4*np.cos(np.pi*2*9*x[0])*np.cos(np.pi*2*9*x[1])
            
        elif self._mode == "fourier_low_freq":
            values[0] = 0.4*np.cos(np.pi*2*2*x[0])*np.cos(np.pi*2*4*x[1])-0.4*np.cos(np.pi*2*3*x[0])*np.cos(np.pi*2*1*x[1])+0.3*np.cos(np.pi*2*5*x[0])*np.cos(np.pi*2*2*x[1])\
                        +0.4*np.cos(np.pi*2*3*x[0])*np.cos(np.pi*2*3*x[1])


def solve_heat_equation(sigma = 1, dt = 0.1**2*1e-2, T_dt = 400, nx_ny = 60, initial_condition = "rectangles"):


    T = np.arange(0,T_dt*dt,dt)

    nx = ny = nx_ny
    mesh = mesh = UnitSquareMesh(nx,ny)
    V = FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, Constant(0), boundary)

    u_init = InitialConditionsHE(initial_condition)

    u_n = interpolate(u_init, V)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(0)

    F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
    a, L = lhs(F), rhs(F)



    u = Function(V)

    sols = []
    for n in tqdm(T):


        solve(a == L, u, bc)

        u_n.assign(u)

        X,Y,Z = fenics_fun_2_grid(u, mesh)

        sols.append(Z)
    
    sols = np.array(sols).astype(np.float32)
    
    
    return sols


def post_process_and_save_he(sols, results_dir = "./data/HE", name_simulation = "HE_sim_default_name", save_plots = False):
    
    try:
        os.makedirs(results_dir)
    except:
        pass
    


    np.save(os.path.join(results_dir, name_simulation+"_array"), sols)
    
    if save_plots:
        
        make_gif(sols,os.path.join(results_dir, name_simulation+"gif"+".gif") ) 


        ampsn = []
        for sol in sols:
            ampsn.append(np.sum(sol))
        ampsn = np.array(ampsn)/np.size(sol)
        
        amps_max = []
        for sol in sols:
            amps_max.append(np.max(sol))
        amps_max = np.array(amps_max)


        fig = plt.figure(figsize = (15,15))


        plt.plot(np.arange(0,len(ampsn),1),np.abs(ampsn), label = "avg amp")
        plt.legend()
        fig.savefig(os.path.join(results_dir,name_simulation+"amplitude_avg.png"))
        
        
        
        fig = plt.figure(figsize = (15,15))
        plt.plot(np.arange(0,len(amps_max),1),np.abs(amps_max), label = "amp max")
        plt.legend()
        fig.savefig(os.path.join(results_dir,name_simulation+"amplitude_max.png"))