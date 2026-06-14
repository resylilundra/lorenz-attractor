import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LorenzModel:

    def __init__(self, sigma=7.0, rho=99.0, beta=10.0/3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    def __call__(self, t, state):  # the method call turns objects of that class into callable objects. So the object behaves like a function.
        return self.rhs(t, state)
    def rhs(self, t, state):
        x, y, z = state
        dx_dt = self.sigma * (y-x)
        dy_dt = x*(self.rho-z) -y
        dz_dt =x*y -self.beta*z
        return np.array([dx_dt,dy_dt,dz_dt])

class BasedFixedStepIntegrator:
    '''Base class for fixed step integration methods.
    It defines the common interface and shared functionality for all fixed step integrators.'''

    def __init__(self, dt=1e-8):
        self.dt = dt
        self.name = self.__class__.__name__  # This sets the name attribute to the name of the specific integrator class (e.g., "EulerIntegrator" or "RK4Integrator").

    def integrate(self, f, tspan, initial_state, t_eval=None):
        """Integrate the system of ODEs defined by f over the time span tspan with the given initial state.
        If t_eval is not provided, a time grid is created using the integrator's dt.
        Returns self with attributes .t and .state set to the evaluation times and solution array.
        """
        t0, tf = tspan
        if t_eval is None:
            t_eval = np.arange(t0, tf, self.dt)

        if len(t_eval) == 0:
            raise ValueError("t_eval must contain at least one time point")

        # Pre-allocate state array and set initial condition
        state = np.zeros((len(t_eval), len(initial_state)))
        state[0] = initial_state

        # Integrate using subclass update step
        for i in range(1, len(t_eval)):
            t = t_eval[i - 1]
            state[i] = self.update(f, t, state[i - 1])

        self.t, self.state = np.array(t_eval), np.array(state)
        return self
    def update(self, f, t, state):
        """This method should be implemented by subclasses to perform a single integration step using the specific method."""
        raise NotImplementedError("Subclasses must implement the update method")

class EulerIntegrator(BasedFixedStepIntegrator):
    """ Note: super() calls the constructor of the BaseIntegrator class
    kwargs() collects any keyword arguments and passes them to the base class constructor."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def update(self,f,t,state):
        return state + self.dt * f(t,state)

# Now we will define the Runge-kutta second order wich is a modified version of Euler method

class ModifiedEulerIntegrator(BasedFixedStepIntegrator):
    ''' This class implements the modified Euler method which is second-order method so it is more accurate
    than the basic Euler method.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def update(self,f,t,state):
        k1 = f(t,state)
        k2 = f(t + self.dt/2, state + self.dt * k1/2)
        return state + self.dt * k2
    
if __name__ == "__main__":
    model = LorenzModel()
    solver = EulerIntegrator(dt = 5e-3)

    result = solver.integrate( model, tspan = (0,50),  initial_state = np.array ([1.001,1.0,1.0]  ) )
    df_result = pd.DataFrame({"t": result.t, "x": result.state[:,0], "y": result.state [:,1], "z": result.state[:,2] })

    print("Integration finished.  Ten final states : " , df_result.tail(10) )  # This prints the last 10 rows of the DataFrame, showing the final states of the system at the end of the integration.
    print("Advanced Euler final x:", result.state[-1, 0])

    solver_rk2 = ModifiedEulerIntegrator(dt = 5e-3)
    result_rk2 = solver_rk2.integrate(model, tspan = (0,50), initial_state = np.array([1.001,1.0,1.0]))
    df_result_rk2 = pd.DataFrame({"t": result_rk2.t, "x": result_rk2.state[:,0], "y": result_rk2.state [:,1], "z": result_rk2.state[:,2] })


    print("Integration finished.  Ten final states : " , df_result_rk2.tail(10) )  # This prints the last 10 rows of the DataFrame, showing the final states of the system at the end of the integration.
    print("Advanced RK2 final x:", result_rk2.state[-1, 0])

    # Plotting the results

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(projection = '3d')
    ax.plot(df_result['x'], df_result['y'], df_result['z'], lw = 1, color = "b" ,label= "Euler method")
    ax.plot(df_result_rk2['x'], df_result_rk2['y'], df_result_rk2['z'], lw = 1, color = "k" ,label= "RK2 method")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z=axis")
    ax.legend()
    plt.title("Lorenz system :Euler method  and Modified Euler method (Rk2)")
    plt.savefig(f"Lorenz_{solver.name}_{solver.dt}.png", dpi = 150)
    print("Plot saved to lorenz_euler.png")    

	

     
