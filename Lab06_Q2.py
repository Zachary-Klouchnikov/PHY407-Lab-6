__authors__ = "Zachary Klouchnikov and Hannah Semple"

# This file provides our answers to Q2 of Lab06 for PHY407. We modelled the motion of the floors of a building using
# Verlet's method as well as by finding the normal modes.

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

"""
FUNCTIONS
"""
def matrix(N):
    """
    Returns the matrix corresponding to the equations of motion for N many floors
    
    INPUT:
    N [int] is the number of floors in the modelled building
    
    OUTPUT:
    M [array] is the matrix
    """
    M = np.diag(np.full(N,-2)) + np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1)
    return M


def eom(xs, coeffs):
    """
    Returns the acceleration according to the equations of motion for moving building floors
    
    INPUT:
    xs [array] are the x positions in m
    coeffs [array] are coefficients taken from the equation of motion matrix
    
    OUTPUT:
    y [float] is the result from the equation of motion
    """
    a = 0
    for i in range(len(xs)):
        a += xs[i] * coeffs[i]
    return a
    

def verlet(N, dt, cycles, x0, v0):
    """
    Performs Verlet method for solving ODEs, modifed for the motion of building floors
    
    INPUT:
    N [int] number of stories
    dt [float] time step in s
    cycles [int] number of simulation cycles
    x0 [array] initial positions in m
    v0 [array] initial velocities in m/s
    
    OUTPUT:
    t [array] time array in s
    x [array] position array in m
    v [array] velocity array in m/s
    """
    
    # initialize arrays
    t = np.zeros(cycles)
    x = np.zeros((cycles, N))
    v = np.zeros((cycles, N))
    A = k_m * matrix(N)
    
    # initial conditions
    t[0] = 0
    x[0] = x0.copy()
    v[0] = v0.copy()
    
    # first step
    a0 = np.zeros(N)  #initialise acceleration
    for n in range(N):
        a0[n] = eom(x[0], A[n])  #calculate acceleration
    v_half = v[0] + 0.5 * a0 * dt  #v(t + h/2)
    x[1] = x[0] + dt * v_half  #updated velocity
    
    # Verlet loop until the final values
    for i in range(1, cycles-1):
        t[i] = i * dt  #update time array
        a_current = np.zeros(N)  #initialise acceleration
        for n in range(N):
            a_current[n] = eom(x[i], A[n])  #calculate acceleration
        
        x[i+1] = 2 * x[i] - x[i-1] + a_current * dt**2  #calculate position
        v[i] = (x[i+1] - x[i-1]) / (2 * dt)  #calculate velocity
    
    # final time and velocity
    t[-1] = (cycles-1) * dt
    v[-1] = (x[-1] - x[-2]) / dt
    
    return t, x, v

"""
PART A
"""
#constants
k_m = 400  #rad^2/s^2
dt = 1e-3  #s
cycles = 1000  #number of simulation cycles
x0 = np.array([0.1, 0., 0.])  #initial position values in m
v0 = np.array([0., 0., 0.])  #initial velocity vaalues in m/s

#floor motion of a building with 3 floors
N = 3  #number of floors
t, x_3, v_3 = verlet(N, dt, cycles, x0, v0)  #time, position, velocity values

#floor motion of a building with 10 floors
N = 10  #number of floors
x0 = np.array([0.1, 0., 0., 0,0,0,0,0,0,0])  #initial position values in m
v0 = np.array([0., 0., 0., 0,0,0,0,0,0,0])  #initial velocity vaalues in m/s
t, x_10, v_10 = verlet(N, dt, cycles, x0, v0)  #time, position, velocity values

#plotting
plt.figure(figsize=(8, 4))
plt.plot(t,np.transpose(x_3)[0], label='Floor 0', color = 'purple')
plt.plot(t,np.transpose(x_3)[1], label='Floor 1', color = 'teal')
plt.plot(t,np.transpose(x_3)[2], label='Floor 2', color = 'coral')
plt.title('Simulated Building Floor Motion (3 Floors)')
plt.xlabel('Time [s]')
plt.ylabel('Floor Motion [m]')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, alpha=0.3)
# plt.savefig('sim_3.pdf')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(t,np.transpose(x_10)[0], label='Floor 0', color = 'firebrick')
plt.plot(t,np.transpose(x_10)[1], label='Floor 1', color = 'orangered')
plt.plot(t,np.transpose(x_10)[2], label='Floor 2', color = 'orange')
plt.plot(t,np.transpose(x_10)[3], label='Floor 3', color = 'gold')
plt.plot(t,np.transpose(x_10)[4], label='Floor 4', color = 'limegreen')
plt.plot(t,np.transpose(x_10)[5], label='Floor 5', color = 'green')
plt.plot(t,np.transpose(x_10)[6], label='Floor 6', color = 'dodgerblue')
plt.plot(t,np.transpose(x_10)[7], label='Floor 7', color = 'navy')
plt.plot(t,np.transpose(x_10)[8], label='Floor 8', color = 'darkorchid')
plt.plot(t,np.transpose(x_10)[9], label='Floor 9', color = 'hotpink')
plt.grid(True, alpha=0.3)
plt.xlabel('Time [s]')
plt.ylabel('Floor Motion [m]')
plt.title('Simulated Building Floor Motion (10 Floors)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('sim_10.pdf')
plt.show()


"""
PART B
"""
###PART B
N = 3
A = k_m * matrix(N)

eigenvalues, eigenvectors = eigh(A)
frequencies = np.sqrt(-eigenvalues)

x0, v0 = np.array(([np.array([0.1, 0., 0.]), np.array([0., 0., 0.])]))

phases = np.zeros(N)  #initialising arrays
amplitudes = np.zeros(N)

#finding the phases and amplitudes for the normal modes of the system
for i in range(N):
    mode_shape = eigenvectors[:, i]

    #project initial position and velocity onto mode
    a = np.dot(mode_shape, x0)
    b = np.dot(-mode_shape, v0 / np.sqrt(-eigenvalues[i]))
  
    #calculate amplitude and phase
    amplitude = np.sqrt(a**2 + b**2)
    phase = np.arctan2(-b, a)  #using arctan2 to get quadrant

    amplitudes[i] = amplitude  #updating arrays
    phases[i] = phase


#plotting
plt.figure(figsize=(8, 4))
for floor in range(N):  # For each floor
    #calculating postion
    mode0 = amplitudes[0] * eigenvectors[floor, 0] * np.cos(frequencies[0] * t + phases[0])
    mode1 = amplitudes[1] * eigenvectors[floor, 1] * np.cos(frequencies[1] * t + phases[1])
    mode2 = amplitudes[2] * eigenvectors[floor, 2] * np.cos(frequencies[2] * t + phases[2])
    plt.plot(t, (mode0 + mode1 + mode2), label = 'Floor {} Normal Modes'.format(floor), color = '{}'.format(['purple', 'teal', 'coral'][floor]))
plt.ylabel('Floor Motion [m]')
plt.xlabel('Time [s]')
plt.title('Modal Building Floor Motion (3 Floors)')
plt.grid(True, alpha=0.3)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('mode.pdf')
plt.show()
