import scipy as sp
import numdifftools as nd
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS
MU_N = 1
PI = np.pi
ROD_LENGTH = 4 #No units, use units of mks for now
G_POT_DEPTH = -5
GRAV_CONSTANT = -9.81
BOB_MASS = 1
POT_MAP = 2 #This needs to be a calculated potential map, ideally saved because they are *expensive* to generate
NUMBER_STEPS = 50

#https://stackoverflow.com/questions/30378676/calculate-curl-of-a-vector-field-in-python-and-plot-it-with-matplotlib
def h(x):
    return np.array([3*x[0]**2,4*x[1]*x[2]**3, 2*x[0]])

def curl(f,x):
    jac = nd.Jacobian(f)(x)
    return np.array([jac[2,1]-jac[1,2],jac[0,2]-jac[2,0],jac[1,0]-jac[0,1]])

def m(k,x):
    return k*norm_v(x)

def norm_v(v):
    return v/np.linalg.norm(v)

def xy_convert(x,y):
    z = -1 * np.sqrt(ROD_LENGTH**2 - (x**2 + y**2))
    
    if np.isnan(z):
        return np.array([0,0,0])
    else:
        return np.array([x,y,z])
    
    #return np.array([x,y,z])
def dist_to_magnet(x,mag):
    pass

class Magnet:
    def __init__(self,x,m):
        self.x = x
        self.m = m
        return None
    
    def gen_R(self,x):
         R = x - self.x
         norm_R = np.linalg.norm(R)
         return R/(norm_R**3 )
    
    def mag_find(self,x):
        return np.cross(self.m,self.gen_R(x))

def calc_grav_potential(x):
    return BOB_MASS * GRAV_CONSTANT * x[2] 

def generatePositionArray():
    number_steps = NUMBER_STEPS
    empty_array = np.empty((number_steps,number_steps,3))
    for q in range(number_steps):
        for r in range(number_steps):
            empty_array[q][r] = xy_convert(-0.5 * ROD_LENGTH + (q+0.5) * ROD_LENGTH/number_steps, -0.5 * ROD_LENGTH + (r+0.5) * ROD_LENGTH/number_steps)
    print("Pos Array Done")
    return empty_array

def calcPotential(mag_array,x):
    #if np.isnan(x).any():
    #    return 0
    mag_vector_potential = np.array([0.,0.,0.])
    for mag in mag_array:
        mag_vector_potential += curl(mag.mag_find,x)
    mag_potential = np.dot(m(1,x), mag_vector_potential)
    return np.array([x[0],x[1],calc_grav_potential(x)+ mag_potential])
    #return calc_grav_potential(x)+mag_potential

def calcPotentialGrid(position_array,mag_array):
    position_shape = position_array.shape
    empty_array = np.empty(position_shape[0:3])

    for q in range(position_shape[0]):
        for r in range(position_shape[1]):
            empty_array[q][r] = calcPotential(mag_array, position_array[q][r])
        print("pgrid step" + str(1000* q))
    return empty_array

class particle:
    def __init__(self, potentialMap,xst,yst,vxst,vyst):
        self.timestep = 0.01
        self.potential_map = potentialMap
        
        self.x = xst
        self.y = yst
        
        self.vx = vxst
        self.vy = vyst
        
        self.mass = 1
        return None
    
    def calculate_new_position(self):
        self.x += self.vx * self.timestep
        self.y += self.vy * self.timestep
        return None
    
    def calculate_new_velocity(self):
        box_position = self.convert_pos_to_matrix_space(self.x, self.y)
        g_x,g_y,whoasked = np.gradient(self.potential_map)
        print(box_position) 
        a_x = g_x[box_position[0],box_position[1],2]
        a_y = g_y[box_position[0],box_position[1],2]
        
        self.vx += a_x * self.timestep
        self.vy += a_y * self.timestep
        return None
        
    def convert_pos_to_matrix_space(self,x,y):
        matrix_x = round((x + 0.5 * ROD_LENGTH)*(NUMBER_STEPS/ROD_LENGTH) - 0.5) 
        matrix_y = round((y + 0.5 * ROD_LENGTH)*(NUMBER_STEPS/ROD_LENGTH) - 0.5)
        return (matrix_x,matrix_y)

    def update(self):
        self.calculate_new_position()
        self.calculate_new_velocity()
        return (self.x,self.y)

if __name__ == "__main__":

    #xpos = xy_convert(1.05,0)
    m1 = Magnet(np.array([0,0,G_POT_DEPTH]),np.array([0,0,-10]))
    m2 = Magnet(np.array([0,-1,G_POT_DEPTH]),np.array([0,0,-10]))
    m3 = Magnet(np.array([0,1,G_POT_DEPTH]),np.array([0,0,-10]))
    m4 = Magnet(np.array([-1,0,G_POT_DEPTH]),np.array([0,0,-10]))
    
    mag_array = [m1]#,m2,m3,m4]#[m1,m2]
    
    emily = generatePositionArray()
    
    alice = calcPotentialGrid(emily,mag_array)
    #print(alice)
    
    pt = particle(alice,0,-1,0.1,0)
    #foxtrot_alfa,foxtrot_bravo,foxtrot_charlie = pt.calculate_acceleration()
    #print(foxtrot_alfa[7,7,2])
    #print(foxtrot_bravo[7,7,2])
    #print(pt.convert_pos_to_matrix_space(0.01, 1.3))
    #number_steps = 11
    #q= round((x + 0.5 * ROD_LENGTH)*(number_steps/ROD_LENGTH) - 0.5) 
    data = []
    for x in range(3100):
        data.append(pt.update())
    
    xfa, yfa = zip(*data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(xfa, yfa)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.xlabel("x")
    plt.ylabel("y")

    ax.set_aspect('equal', adjustable='box')


plt.show()
    
    
    # Create array of all magnets, create array of points were magnet on rod
    # could be, find potential for all of those points, 
