import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sympy.utilities.iterables import variations
import time
from matplotlib.ticker import PercentFormatter
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")
#%matplotlib inline
#%matplotlib widget

# GLOBAL CONSTANTS
J=-1
h=1
K=2

def saw_3d(n):
    x, y, z = [n], [n], [n]
    positions = set([(n,n,n)])  #positions is a set that stores all sites visited by the walk
    #stuck = False
    for i in range(n-1):
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        directions_feasible = []  #directions_feasible stores the available directions
        for dx, dy, dz in directions:
            if (x[-1] + dx, y[-1] + dy, z[-1]+dz) not in positions:  #checks if direction leads to a site not visited before
                directions_feasible.append((dx,dy,dz))
        if directions_feasible:  #checks if there is a direction available
            dx, dy,dz = directions_feasible[np.random.randint(0,len(directions_feasible))]  #choose a direction at random among available ones
            positions.add((x[-1] + dx, y[-1] + dy,z[-1]+dz))
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
            z.append(z[-1] + dz)
        else:  #in that case the walk is stuck
            #stuck = True
            #steps = i+1
            break  #terminate the walk prematurely
        #steps = n+1
    #print(positions)
    return x, y, z #, stuck, steps

def create_initial(rows, cols,height):
    return np.random.choice((-1,1), size=(rows,cols,height))

def merge_lists(x,y,z):
    return [[x[i],y[i],z[i]] for i in range(len(x))]

def spin_on_saw(spin_array,x,y,z):
    spins_list = merge_lists(x,y,z)
    for i in range(len(spin_array)):
        for j in range(len(spin_array[i])):
            for k in range(len(spin_array[i][j])):
                if [i,j,k] not in spins_list:
                    spin_array[i][j][k] = 0
    return spin_array

def plus_min(spin_array):
    plus_spin = []
    min_spin = []
    for i in range(len(spin_array)):
        for j in range(len(spin_array)):
            for k in range(len(spin_array)):
                if spin_array[i][j][k] == 1:
                    plus_spin.append([i,j,k])
                if spin_array[i][j][k] == -1:
                    min_spin.append([i,j,k])
    return plus_spin, min_spin

def neighbors(spin_array,N,x,y,z):
    left   = (x, (y-1),z)
    right  = (x, (y+1),z)
    top    = ((x-1), y,z)
    bottom = ((x+1), y,z)
    under = (x,y,z-1)
    over = (x,y,z+1)
    '''left   = (x, (y+1+N)%N)
    right  = (x, (y-1+N)%N)
    top    = ((x+1+N)%N, y)
    bottom = ((x-1+N)%N, y)'''
    return [spin_array[left[0], left[1],left[2]],
            spin_array[right[0], right[1],right[2]],
            spin_array[top[0], top[1], top[2]],
            spin_array[bottom[0], bottom[1], bottom[2]],
            spin_array[over[0], over[1], over[2]],
            spin_array[under[0], under[1], under[2]]
           ]

def energy(spin_array, N, x_pos ,y_pos,z_pos):
    return (J*spin_array[x_pos][y_pos][z_pos]*sum(neighbors(spin_array, N, x_pos, y_pos,z_pos)) + h*spin_array[x_pos][y_pos][z_pos])

def total_energy(spin_array,N,x,y,z):
    spin_list = merge_lists(x,y,z)
    '''sum_energy = 0
    for i in range(len(spin_list)):
        print("position: ",spin_list[i][0],spin_list[i][1])
        print("neighbours: ", neighbors(spin_array, N, spin_list[i][0], spin_list[i][1]))
        print("energy:",energy(spin_array,N, spin_list[i][0], spin_list[i][1]))
    print(K*len(x))'''
    return -(sum([energy(spin_array,N, spin_list[i][0], spin_list[i][1], spin_list[i][2]) for i in range(len(spin_list))]) + K*len(x))

'''Functions for the SAW move'''


def correct_z(x, y, z, i, spin_array):
    possible = False
    positions = []
    if z[i] == z[i + 1] == z[i - 1]:
        '''
        if i == 0 or i == len(x)-1:
            positions = endpoints(x,y,i,spin_array)
            if len(positions)!=0:
                possible = True
            return possible, positions
        '''
        if y[i] + 1 == y[i - 1] and x[i] == x[i - 1] and y[i + 1] == y[i] and x[i] + 1 == x[i + 1] and \
                spin_array[x[i] + 1][y[i] + 1][z[i]] == 0:  # 1
            positions = [x[i] + 1, y[i] + 1, z[i]]
            possible = True

        if y[i] == y[i - 1] and x[i] - 1 == x[i - 1] and y[i + 1] == y[i] - 1 and x[i] == x[i + 1] and \
                spin_array[x[i] - 1][y[i] - 1][z[i]] == 0:  # 2
            positions = [x[i] - 1, y[i] - 1, z[i]]
            possible = True

        if y[i] == y[i - 1] and x[i] + 1 == x[i - 1] and y[i + 1] == y[i] + 1 and x[i] == x[i + 1] and \
                spin_array[x[i] + 1][y[i] + 1][z[i]] == 0:  # 3
            positions = [x[i] + 1, y[i] + 1, z[i]]
            possible = True

        if y[i] - 1 == y[i - 1] and x[i] == x[i - 1] and y[i + 1] == y[i] and x[i] - 1 == x[i + 1] and \
                spin_array[x[i] - 1][y[i] - 1][z[i]] == 0:  # 4
            positions = [x[i] - 1, y[i] - 1, z[i]]
            possible = True

        if y[i] == y[i - 1] and x[i] + 1 == x[i - 1] and y[i + 1] == y[i] - 1 and x[i] == x[i + 1] and \
                spin_array[x[i] + 1][y[i] - 1][z[i]] == 0:  # 5
            positions = [x[i] + 1, y[i] - 1, z[i]]
            possible = True
            return possible, positions
        if y[i] + 1 == y[i - 1] and x[i] == x[i - 1] and y[i + 1] == y[i] and x[i] - 1 == x[i + 1] and \
                spin_array[x[i] - 1][y[i] + 1][z[i]] == 0:  # 6
            positions = [x[i] - 1, y[i] + 1, z[i]]
            possible = True

        if y[i] - 1 == y[i - 1] and x[i] == x[i - 1] and y[i + 1] == y[i] and x[i] + 1 == x[i + 1] and \
                spin_array[x[i] + 1][y[i] - 1][z[i]] == 0:  # 7
            positions = [x[i] + 1, y[i] - 1, z[i]]
            possible = True

        if y[i] == y[i - 1] and x[i] - 1 == x[i - 1] and y[i + 1] == y[i] + 1 and x[i] == x[i + 1] and \
                spin_array[x[i] - 1][y[i] + 1][z[i]] == 0:  # 8
            positions = [x[i] - 1, y[i] + 1, z[i]]
            possible = True
        return possible, positions
    else:
        return possible, positions


def correct_y(x, y, z, i, spin_array):
    possible = False
    positions = []
    if y[i] == y[i + 1] == y[i - 1]:
        '''
        if i == 0 or i == len(x)-1:
            positions = endpoints(x,y,i,spin_array)
            if len(positions)!=0:
                possible = True
            return possible, positions
        '''
        if z[i] + 1 == z[i - 1] and x[i] == x[i - 1] and z[i + 1] == z[i] and x[i] + 1 == x[i + 1] and \
                spin_array[x[i] + 1][y[i]][z[i] + 1] == 0:  # 1
            positions = [x[i] + 1, y[i], z[i] + 1]
            possible = True

        if z[i] == z[i - 1] and x[i] - 1 == x[i - 1] and z[i + 1] == z[i] - 1 and x[i] == x[i + 1] and \
                spin_array[x[i] - 1][y[i]][z[i] - 1] == 0:  # 2
            positions = [x[i] - 1, y[i], z[i] - 1]
            possible = True

        if z[i] == z[i - 1] and x[i] + 1 == x[i - 1] and z[i + 1] == z[i] + 1 and x[i] == x[i + 1] and \
                spin_array[x[i] + 1][y[i]][z[i] + 1] == 0:  # 3
            positions = [x[i] + 1, y[i], z[i] + 1]
            possible = True

        if z[i] - 1 == z[i - 1] and x[i] == x[i - 1] and z[i + 1] == z[i] and x[i] - 1 == x[i + 1] and \
                spin_array[x[i] - 1][y[i]][z[i] - 1] == 0:  # 4
            positions = [x[i] - 1, y[i], z[i] - 1]
            possible = True

        if z[i] == z[i - 1] and x[i] + 1 == x[i - 1] and z[i + 1] == z[i] - 1 and x[i] == x[i + 1] and \
                spin_array[x[i] + 1][y[i]][z[i] - 1] == 0:  # 5
            positions = [x[i] + 1, y[i], z[i] - 1]
            possible = True
            return possible, positions
        if z[i] + 1 == z[i - 1] and x[i] == x[i - 1] and z[i + 1] == z[i] and x[i] - 1 == x[i + 1] and \
                spin_array[x[i] - 1][y[i]][z[i] + 1] == 0:  # 6
            positions = [x[i] - 1, y[i], z[i] + 1]
            possible = True

        if z[i] - 1 == z[i - 1] and x[i] == x[i - 1] and z[i + 1] == z[i] and x[i] + 1 == x[i + 1] and \
                spin_array[x[i] + 1][y[i]][z[i] - 1] == 0:  # 7
            positions = [x[i] + 1, y[i], z[i] - 1]
            possible = True

        if z[i] == z[i - 1] and x[i] - 1 == x[i - 1] and z[i + 1] == z[i] + 1 and x[i] == x[i + 1] and \
                spin_array[x[i] - 1][y[i]][z[i] + 1] == 0:  # 8
            positions = [x[i] - 1, y[i], z[i] + 1]
            possible = True
        return possible, positions
    else:
        return possible, positions


def correct_x(x, y, z, i, spin_array):
    possible = False
    positions = []
    if x[i] == x[i + 1] == x[i - 1]:
        '''
        if i == 0 or i == len(x)-1:
            positions = endpoints(x,y,i,spin_array)
            if len(positions)!=0:
                possible = True
            return possible, positions
        '''
        if y[i] + 1 == y[i - 1] and z[i] == z[i - 1] and y[i + 1] == y[i] and z[i] + 1 == z[i + 1] and \
                spin_array[x[i]][y[i] + 1][z[i] + 1] == 0:  # 1
            positions = [x[i], y[i] + 1, z[i] + 1]
            possible = True

        if y[i] == y[i - 1] and z[i] - 1 == z[i - 1] and y[i + 1] == y[i] - 1 and z[i] == z[i + 1] and \
                spin_array[x[i]][y[i] - 1][z[i] - 1] == 0:  # 2
            positions = [x[i], y[i] - 1, z[i] - 1]
            possible = True

        if y[i] == y[i - 1] and z[i] + 1 == z[i - 1] and y[i + 1] == y[i] + 1 and z[i] == z[i + 1] and \
                spin_array[x[i]][y[i] + 1][z[i] + 1] == 0:  # 3
            positions = [x[i], y[i] + 1, z[i] + 1]
            possible = True

        if y[i] - 1 == y[i - 1] and z[i] == z[i - 1] and y[i + 1] == y[i] and z[i] - 1 == z[i + 1] and \
                spin_array[x[i]][y[i] - 1][z[i] - 1] == 0:  # 4
            positions = [x[i], y[i] - 1, z[i] - 1]
            possible = True

        if y[i] == y[i - 1] and z[i] + 1 == z[i - 1] and y[i + 1] == y[i] - 1 and z[i] == z[i + 1] and \
                spin_array[x[i]][y[i] - 1][z[i] + 1] == 0:  # 5
            positions = [x[i], y[i] - 1, z[i] + 1]
            possible = True
            return possible, positions
        if y[i] + 1 == y[i - 1] and z[i] == z[i - 1] and y[i + 1] == y[i] and z[i] - 1 == z[i + 1] and \
                spin_array[x[i]][y[i] + 1][z[i] - 1] == 0:  # 6
            positions = [x[i], y[i] + 1, z[i] - 1]
            possible = True

        if y[i] - 1 == y[i - 1] and z[i] == z[i - 1] and y[i + 1] == y[i] and z[i] + 1 == z[i + 1] and \
                spin_array[x[i]][y[i] - 1][z[i] + 1] == 0:  # 7
            positions = [x[i], y[i] - 1, z[i] + 1]
            possible = True

        if y[i] == y[i - 1] and z[i] - 1 == z[i - 1] and y[i + 1] == y[i] + 1 and z[i] == z[i + 1] and \
                spin_array[x[i]][y[i] + 1][z[i] - 1] == 0:  # 8
            positions = [x[i], y[i] + 1, z[i] - 1]
            possible = True
        return possible, positions
    else:
        return possible, positions

def find_possible_points(x,y,z,spins):
    possible_points_x = []
    possible_points_y = []
    possible_points_z = []
    for i in range(1,len(x)-1):
        possible_x, positions_x = correct_x(x,y,z,i,spins)
        possible_y, positions_y = correct_y(x,y,z,i,spins)
        possible_z, positions_z = correct_z(x,y,z,i,spins)
        if possible_x:
            possible_points_x.append(i) #test: [x[i],y[i],i]
        if possible_y:
            possible_points_y.append(i) #test: [x[i],y[i],i]
        if possible_z:
            possible_points_z.append(i) #test: [x[i],y[i],i]
    '''possible, positions = correct_x(x,y,len(x)-1, spins)
    if posible:
        possible_points.append(len(x)-1)'''
    if len(possible_points_x) ==0:
        raise ValueError('list of possible points_x must be non-empty, the SAW cannot move in this direction.')
    if len(possible_points_y) ==0:
        raise ValueError('list of possible points_y must be non-empty, the SAW cannot move in this direction.')
    if len(possible_points_z) ==0:
        raise ValueError('list of possible points_z must be non-empty, the SAW cannot move in this direction.')
    return possible_points_x, possible_points_y, possible_points_z

def spins_1dising(spin_array,x,y,z):
    spin_1d = [0]#[0]
    for i in range(len(x)):
        spin_1d.append(spin_array[x[i]][y[i]][z[i]])
    spin_1d.append(0)
    return spin_1d

def energy_ising_1d(spin_list):
    sum_energy = 0
    for i in range(1,len(spin_list)-1):
        sum_energy += J*spin_list[i]*(spin_list[i-1]+spin_list[i+1])
        #print(sum_energy)
    sum_energy += h*sum(spin_list)
    #print(sum_energy)
    sum_energy += K*(len(spin_list)-2)
    #print(sum_energy)
    return -sum_energy

'''METROPOLIS ALGORITHM FUNCTIONS'''
def metropol_spin(config,sweeps,T):
    β = 1/(T)
    mag = np.zeros(sweeps)
    Energy = np.zeros(sweeps)
    for sweep in range(sweeps):
        #config_copy = config.copy()
        i = np.random.randint(0,len(x3))
        x_pos = x3[i] #np.random.randint(0,99)#x[i]
        y_pos = y3[i] #np.random.randint(0,99)#y[i]
        z_pos = z3[i] #np.random.randint(0,99)#z[i]
        E_i = energy(config,n,x_pos,y_pos,z_pos)
        config[x_pos][y_pos][z_pos] = - config[x_pos][y_pos][z_pos]
        #E_i = energy(config,n,x_pos,y_pos,z_pos)
        E_f = energy(config,n,x_pos,y_pos,z_pos)
        ΔE = E_f-E_i
        r = np.random.uniform()
        if not (ΔE <=0 or r<= np.exp(-β*ΔE)):
            config[x_pos][y_pos][z_pos] = - config[x_pos][y_pos][z_pos]
        #Magnetization
        mag[sweep] = sum(sum(sum(config)))/(len(x3)) #[sweep]
        #Energy
        Energy[sweep] = total_energy(config,n,x3,y3,z3) #[sweep]
        #print(mag[sweep])
    return [config, mag, Energy]

def metropol_saw(config,sweeps,T):
    β = 1/(T)
    mag = np.zeros(sweeps)
    Energy = np.zeros(sweeps)
    for sweep in range(sweeps):
        config_copy = config.copy()
        i = np.random.randint(0,len(x3))
        x_pos = x3[i]
        y_pos = y3[i]
        z_pos = z3[i]
        possible_x, positions_x = correct_x(x3,y3,z3,i,spins)
        possible_y, positions_y = correct_y(x3,y3,z3,i,spins)
        possible_z, positions_z = correct_z(x3,y3,z3,i,spins)
        if possible_x or possible_y or possible_z:
            config_copy = metropolis_saw_move(i)
            E_i = energy(config,n,x_pos,y_pos,z_pos)
            E_f = energy(config_copy,n,x_pos,y_pos,z_pos)
            ΔE = E_f-E_i
            r = np.random.uniform()
            if ΔE <=0 or r<= np.exp(-β*ΔE):
                config = metropolis_saw_move(i)
        #Magnetization
        mag[sweep] = sum(sum(sum(config)))/(len(x3))#[sweep]
        #Energy
        Energy[sweep]  = total_energy(config,n,x3,y3,z3)#[sweep]
        #print(mag[sweep])
    return [config, mag, Energy]

def metropolis_saw_move(i):
    #counter = 0
    #while counter <1:
        #i = np.random.randint(0,len(x)-1)
    #i = np.random.choice(find_possible_points(x3,y3,z3,spins))
    possible_x, positions_x = correct_x(x3,y3,z3,i,spins)
    possible_y, positions_y = correct_y(x3,y3,z3,i,spins)
    possible_z, positions_z = correct_z(x3,y3,z3,i,spins)
    if possible_x:
        spins[positions_x[0]][positions_x[1]][positions_x[2]] = spins[x3[i]][y3[i]][z3[i]]
        spins[x3[i]][y3[i]][z3[i]] = 0
        x3[i] = positions_x[0]
        y3[i] = positions_x[1]
        z3[i] = positions_x[2]
    if possible_y:
        spins[positions_y[0]][positions_y[1]][positions_y[2]] = spins[x3[i]][y3[i]][z3[i]]
        spins[x3[i]][y3[i]][z3[i]] = 0
        x3[i] = positions_y[0]
        y3[i] = positions_y[1]
        z3[i] = positions_y[2]
    if possible_z:
        spins[positions_z[0]][positions_z[1]][positions_z[2]] = spins[x3[i]][y3[i]][z3[i]]
        spins[x3[i]][y3[i]][z3[i]] = 0
        x3[i] = positions_z[0]
        y3[i] = positions_z[1]
        z3[i] = positions_z[2]
        #counter += 1
    plus_spin, min_spin = plus_min(spins)
    #min_spin
    #plus_spin_x, plus_spin_y = zip(*plus_spin)
    #min_spin_x, min_spin_y = zip(*min_spin)
    return spins

def metropolis_combo(config,sweeps,T):
    # random choice of either a spin or saw move
    magnetization = np.zeros(sweeps)
    energy_saw = np.zeros(sweeps)
    spins_copy = np.zeros(sweeps)
    for sweep in range(sweeps):
        if np.random.randint(0,2) == 0: #spin move
            temp_var = metropol_spin(spins, 1, T)
            #spins_copy = temp_var[0]
            magnetization[sweep] = temp_var[1]
            energy_saw[sweep] = temp_var[2]
        else: # path change
            temp_var = metropol_saw(spins, 1, T)
            #spins_copy = temp_var[0]
            magnetization[sweep] = temp_var[1]
            energy_saw[sweep] = temp_var[2]
    return magnetization, energy_saw

n = 50
x3, y3, z3 = saw_3d(n)

x_original = x3.copy()
y_original = y3.copy()
z_original = z3.copy()

path = merge_lists(x3, y3, z3)

N = 3 * n
config_initial = create_initial(N, N, N)  # np.ones((N,N))
spins = spin_on_saw(config_initial, x3, y3, z3)
spins_original = spins.copy()
# spins_1d = spins_1dising(spins,x,y)
# spins = create_high_initial(spin_array,x,y)
# print(spins)
# print(config_initial)
# print(dummy)
plus_spin, min_spin = plus_min(spins)
#min_spin
plus_spin_x, plus_spin_y, plus_spin_z = zip(*plus_spin)
min_spin_x, min_spin_y, min_spin_z = zip(*min_spin)
fig = plt.figure(figsize = (8, 8))
ax = plt.axes(projection='3d')
ax.grid()
ax.plot(x3, y3, z3, color='black', linewidth = 1)
#ax.scatter(x3,y3,z3, 'bo')
ax.scatter(plus_spin_x,plus_spin_y,plus_spin_z,s=100, color='green', label='1')
ax.scatter(min_spin_x,min_spin_y, min_spin_z,s=100,color = 'red', label='-1')
ax.plot(x3[0], y3[0],z3[0], 'ks', ms = 5, label='start')
ax.plot(x3[-1], y3[-1],z3[-1],'kp', ms = 5, label='end')
plt.legend()
plt.title('3D SAW of length ' + str(len(x3)), fontsize=14, fontweight='bold', y = 1.05)
#plt.savefig('/home/sander/Documents/School/univ/Fysica/master/fase_3/thesis/notes/notes/Pictures/SAW_plot_samespin.png', bbox_inches='tight')
plt.show()

'''steps and temperature'''
T=2
steps = 10**4

''' COMBINATION OF SPIN AND SAW MOVES '''
t_b = time.time()
mag_after, energy_after = metropolis_combo(spins,steps,T)
t_e = time.time()
print(t_e-t_b, "s")

# PLOT OF THE saw AFTER THE SIMULATION

plus_spin, min_spin = plus_min(spins)
plus_spin_x, plus_spin_y, plus_spin_z = zip(*plus_spin)
min_spin_x, min_spin_y, min_spin_z = zip(*min_spin)
fig = plt.figure(figsize = (8, 8))
ax = plt.axes(projection='3d')
ax.grid()
ax.plot(x3, y3, z3, color='black', linewidth = 1)
#ax.scatter(x3,y3,z3, 'bo')
ax.scatter(plus_spin_x,plus_spin_y,plus_spin_z,s=100, color='green', label='1')
ax.scatter(min_spin_x,min_spin_y, min_spin_z,s=100,color = 'red', label='-1')
ax.plot(x3[0], y3[0],z3[0], 'ks', ms = 5, label='start')
ax.plot(x3[-1], y3[-1],z3[-1],'kx', ms = 5, label='end')
plt.legend()
plt.title('3D SAW of length ' + str(len(x3)), fontsize=14, fontweight='bold', y = 1.05)
#plt.savefig('/home/sander/Documents/School/univ/Fysica/master/fase_3/thesis/notes/notes/Pictures/SAW_plot_samespin.png', bbox_inches='tight')
plt.show()
