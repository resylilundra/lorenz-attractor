import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 7.0
rho = 99.0
beta = 10.0/3.0

#System of ODEs for the Lorenz systems
def lorenz(t, state):
    x, y, z = state
    dx_dt = sigma * (y-x)
    dy_dt = x* (rho-z) -y
    dz_dt = x*y - beta*z
    return np.array([dx_dt,dy_dt,dz_dt])


#Initial conditions and time span (creating a time grid with NumPy)
initial_state = np.array([1.001,1.0,1.0])
t_span = (0,50)  # the start and end time for simulation. SciPy uses this
#format because it chooses the times step sizes adaptively based on the dynamics of the system.
t_points = np.linspace(t_span[0], t_span[1], 10000) # This creates the actual time grid you use for Euler, RK2 or manual simulation.
#It tells the loop excatly where to evaluate the state of the system at each time step.


#Euler"s method

def euler_method(lorenz, initial_state,t_points):
    state = np.zeros((len(t_points),3)) # Preallocate an array to store the state at each time point
    state[0] = initial_state

    for i in range(0, len(t_points)-1):
        state[i+1] = state[i] + lorenz(t_points[i],state[i]) * (t_points[i+1]-t_points[i])
    return state
state_euler = euler_method(lorenz, initial_state, t_points) # This is the computed trajectory of the Lorenz system using Euler's method, 
#stored in state_euler.


#RK4 method

def rk4_method(lorenz, initial_state, t_points):
    state = np.zeros((len(t_points),3))
    state[0] = initial_state

    for i in range(0,len(t_points)-1):
        h = t_points[i+1] - t_points[i]
        K1 = lorenz(t_points[i],state[i]) 
        K2 = lorenz(t_points[i]+h/2,state[i] + K1*h/2) 
        K3 = lorenz(t_points[i]+h/2,state[i] + K2*h/2)
        K4 = lorenz(t_points[i]+h, state[i] + K3*h)
        state[i+1] = state [i] + h/6 * (K1 + 2*K2 + 2*K3 + K4)
    return state
state_rk4 = rk4_method(lorenz, initial_state, t_points)

# Storing results using Pandas DataFrame

df_euler = pd.DataFrame({"t": t_points, "x": state_euler[:,0], "y": state_euler[:,1], "z": state_euler[:,2]})
df_rk4 = pd.DataFrame({"t": t_points, "x": state_rk4[:,0],"y": state_rk4[:,1], "z": state_rk4[:,2]})
print(df_euler)
print(df_rk4)

print("Basic Euler final x:", state_euler[-1, 0])
print("Basic RK4 final x:", state_rk4[-1, 0])


# plotting the results

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(projection = '3d')
ax.plot( df_euler['x'],df_euler['y'],df_euler['z'], lw=0.5, label = "Euler method")
ax.plot(df_rk4['x'],df_rk4['y'],df_rk4['z'], lw=0.5, label = "RK4 method")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()
plt.savefig("lorenz_attractor.png", dpi=150)
print("Plot saved to lorenz_attractpr.png")
plt.show()
