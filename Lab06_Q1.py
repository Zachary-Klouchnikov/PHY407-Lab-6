__authors__ = "Zachary Klouchnikov and Hannah Semple"

# This code solves and analyzes a stick-slip system under varying frictional
# forces using the Runge-Kutta 4th order method.

"""
IMPORTS
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
"""
FUNCTIONS
"""
def rk4(f, r, t, h):
    """Runge-Kutta 4th Order Method for solving ODEs.
    
    Arguments:
    f -- function representing the ODE (dr/dt = f(r, t))
    r -- array for r values
    t -- array for t values
    h -- step size
    """
    x_points = np.array([], dtype = float)
    xdot_points = np.array([], dtype = float)

    for t in t:
        x_points = np.append(x_points, r[0])
        xdot_points = np.append(xdot_points, r[1])
        k_1 = h * f(r, t)
        k_2 = h * f(r + k_1 / 2, t + h / 2)
        k_3 = h * f(r + k_2 / 2, t + h / 2)
        k_4 = h * f(r + k_3, t + h)
        r += (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

    return x_points, xdot_points

def shm(r, t):
    """Simple Harmonic Motion ODE Function.
    
    Arguments:
    r -- array for r values
    t -- time value
    """
    xdot = r[1]
    xddot = -OMEGA ** 2 * (r[0] - v_p * t)
    return np.array([xdot, xddot], dtype = float)

def ode(r, t):
    """Stick-Slip System ODE Function.
    
    Arguments:
    r -- array for r values
    t -- time value
    """
    xdot = r[1]
    xddot = -OMEGA ** 2 * (r[0] - v_p * t) - (r[0] / TAU) - GAMMA * np.exp(
        -np.abs(r[1]) / v_f)
    return np.array([xdot, xddot], dtype = float)

"""
PART A)
"""
"Constants"
ALPHA = 1.0
BETA = 1.0

"Calculating Frictional Forces"
v_f = 1.0
xdot = np.linspace(1, 100, 1000) # Velocity array

# Dynamic Frictional Force
f_f = lambda x, da: -(ALPHA + da) * x 
# Static Frictional Force
f_s = lambda x, db, dv: -(BETA + db) * np.exp(-x / (v_f + dv)) 

"Plotting Total Absolute Frictional Force"
plt.figure()

# Plotting total absolute frictional force
plt.plot(xdot, np.abs(f_f(xdot, 0.0) + f_s(xdot, 0.0, 0.0)), ls = '-',
         color = 'Teal', label = r"$\alpha/\beta = 1$")
plt.plot(xdot, np.abs(f_f(xdot, -0.5) + f_s(xdot, 0.5, 0.0)), ls = '-',
         color = 'Purple', label = r"$\alpha/\beta < 1$")
plt.plot(xdot, np.abs(f_f(xdot, 0.5) + f_s(xdot, -0.5, 0.0)), ls = '-',
         color = 'Coral', label = r"$\alpha/\beta > 1$")

# Labels
plt.title("Total Absolute Frictional Force", fontsize = 12)
plt.xlabel("Velocity of the Mass (m/s)", fontsize = 12)
plt.ylabel("Force (N)", fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# Limits
plt.xlim(0, 100)

# plt.savefig('Figures\\Total Absolute Frictional Force.pdf')
plt.show()

"""
PART B)
"""
"Constants"
OMEGA = 1.0
TAU = 1.0
GAMMA = 0.5
V_C = 1.0

"Calculating Stick-Slip System With No Friction"
v_p = 0.0   

# Defining time array and step size
h = 0.001
t = np.arange(0.0, 50.0, h, dtype = float)

# Setting initial conditions and solving ODE
r = np.array([1.0, 0.0], dtype = float)
r = rk4(shm, r, t, h)

# Defining Energy Function
energy = lambda x, xdot: (OMEGA ** 2) * ((x - v_p * t) ** 2) + (xdot ** 2)

"Plotting Energy of the System"
plt.figure()

# Plotting energy of the system
plt.plot(t, energy(r[0], r[1]), ls = '-', color = 'Teal')

# Labels
plt.title("Energy of the System", fontsize = 12)
plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel("Energy (J)", fontsize = 12)

plt.grid()

# Limits
plt.xlim(0, 50)

# plt.savefig('Figures\\Energy of the System.pdf')
plt.show()

"Plotting Stick-Slip System With No Friction"
plt.figure()

# Plotting Stick-slip system with no friction
plt.plot(t, r[0], ls = '-', color = 'Teal')

# Labels
plt.title("Stick-Slip System With No Friction", fontsize = 12)
plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel("Position (m)", fontsize = 12)

plt.grid()

# Limits
plt.xlim(0, 50)

# plt.savefig('Figures\\Stick-Slip System With No Friction.pdf')
plt.show()

# Defining constants
v_f = 0.1

# Initial position for constant velocity
x_0 = (-GAMMA * np.exp(-V_C / v_f)) / ((OMEGA ** 2) + (1 / TAU))

# Setting initial conditions and solving ODE
r = np.array([x_0, V_C], dtype = float)
r = rk4(ode, r, t, h)

"Plotting Stick-Slip System With No $v_p$"
plt.figure()

# Plotting Stick-slip system with no v_p
plt.plot(t, r[0], ls = '-', color = 'Teal', label = 'Position (m)')
plt.plot(t, r[1], ls = '-', color = 'Purple', label = 'Velocity (m/s)')

# Labels
plt.title("Stick-Slip System With No $v_p$", fontsize = 12)
plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel("Amplitude", fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# Limits
plt.xlim(0, 50)

# plt.savefig('Figures\\Stick-Slip System With No v_p.pdf')
plt.show()

"""
PART C)
"""
# Defining varying pulling velocities
v_p = np.linspace(0.1 * v_f * np.log(GAMMA * TAU / v_f), 1.5 * v_f * np.log(
    GAMMA * TAU / v_f), 3, dtype = float)

r_0 = []
r_1 = []
r_2 = []
for v_p in v_p:
    # Setting initial conditions and solving ODE
    r = np.array([0.0, 0.0], dtype = float)
    r = rk4(ode, r, t, h)
    r_0.append(r[0])
    r_1.append(r[1])
    r_2.append(-OMEGA ** 2 * (r[0] - v_p * t) - (r[0] / TAU) - GAMMA * np.exp(
        -np.abs(r[1]) / v_f))

"Plotting Stick-Slip System Position With Varying $v_p$"
plt.figure()

# Plotting Stick-slip system position with varying v_p
plt.plot(t, r_0[0], ls = '-', color = 'Teal', label = "$v_p = 0.016$")
plt.plot(t, r_0[1], ls = '-', color = 'Purple', label = "$v_p = 0.129$")
plt.plot(t, r_0[2], ls = '-', color = 'Coral', label = "$v_p = 0.241$")

# Labels
plt.title("Stick-Slip System Position With Varying $v_p$", fontsize = 12)
plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel("Position (m)", fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# Limits
plt.xlim(0, 50)

# plt.savefig('Figures\\Stick-Slip System Position With Varying v_p.pdf')
plt.show()

"Plotting Stick-Slip System Velocity With Varying $v_p$"
plt.figure()

# Plotting Stick-slip system velocity with varying v_p
plt.plot(t, r_1[0], ls = '-', color = 'Teal', label = "$v_p = 0.016$")
plt.plot(t, r_1[1], ls = '-', color = 'Purple', label = "$v_p = 0.129$")
plt.plot(t, r_1[2], ls = '-', color = 'Coral', label = "$v_p = 0.241$")

# Labels
plt.title("Stick-Slip System Velocity With Varying $v_p$", fontsize = 12)
plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel("Velocity (m/s)", fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# Limits
plt.xlim(0, 50)

# plt.savefig('Figures\\Stick-Slip System Velocity With Varying v_p.pdf')
plt.show()

"Plotting Stick-Slip System Acceleration With Varying $v_p$"
plt.figure()

# Plotting Stick-slip system position with varying v_p
plt.plot(t, r_2[0], ls = '-', color = 'Teal', label = "$v_p = 0.016$")
plt.plot(t, r_2[1], ls = '-', color = 'Purple', label = "$v_p = 0.129$")
plt.plot(t, r_2[2], ls = '-', color = 'Coral', label = "$v_p = 0.241$")

# Labels
plt.title("Stick-Slip System Acceleration With Varying $v_p$", fontsize = 12)
plt.xlabel("Time (s)", fontsize = 12)
plt.ylabel("Acceleration (m/s²)", fontsize = 12)

plt.legend(fontsize = 12)
plt.grid()

# Limits
plt.xlim(0, 50)

# plt.savefig('Figures\\Stick-Slip System Acceleration With Varying v_p.pdf')
plt.show()
