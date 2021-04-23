##### POISSON########
import sympy as sym
from sympy.abc import x,y


def fun_poisson_linear_1D(u, sigma ):
    
    
    f = - sigma*sym.diff(sym.diff(u, x), x)
    f = sym.simplify(f)
    fun_f = sym.lambdify((x),f,"numpy")
    fun_u = sym.lambdify((x),u,"numpy")
    u_code = sym.printing.ccode(u)
    f_code = sym.printing.ccode(f)
    
    return fun_f, fun_u,  u_code, f_code



def fun_poisson_linear_2D(u, sigma ):
    
    
    f = - sigma*(sym.diff(sym.diff(u, x), x)+ sym.diff(sym.diff(u, y), y) )
    f = sym.simplify(f)
    fun_f = sym.lambdify((x,y),f,"numpy")
    fun_u = sym.lambdify((x,y),u,"numpy")
    u_code = sym.printing.ccode(u)
    f_code = sym.printing.ccode(f)
    
    return fun_f, fun_u,  u_code, f_code