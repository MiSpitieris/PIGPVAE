import torch
import torch.nn as nn
from torchdiffeq import odeint

# Newton's law of cooling (or heating)
class NewtonsLaw(nn.Module):
    def __init__(self):
        super(NewtonsLaw, self).__init__()

    def forward(self, T0, t, Ts, k):
        """
        Parameters:
            t (torch.Tensor): The time tensor.
            Ts (float): The surrounding temperature.

        Returns:
            torch.Tensor: Temperatures at time t.
        """
        return (T0 - Ts) * torch.exp(-k * t) + Ts

class SimplePendulumSolver(nn.Module):
    def __init__(self, solver=odeint):
        """
        Initializes the Simple Pendulum solver.

        Parameters:
            solver: A function like `odeint` from `torchdiffeq` for solving the ODE.
        """
        super(SimplePendulumSolver, self).__init__()
        self.solver = solver

    def pendulum_ode(self, t, y, omega):
        """
        Defines the ODE system for the simple pendulum.

        Parameters:
            t (torch.Tensor): The time tensor (not used in autonomous system).
            y (torch.Tensor): The state vector [theta, theta_dot].
            omega (float or torch.Tensor): Frequency parameter for the pendulum.

        Returns:
            torch.Tensor: Time derivatives of theta and theta_dot.
        """
        theta, theta_dot = y
        return torch.stack([
            theta_dot,
            -omega * torch.sin(theta)
        ])

    def forward(self, y0, t_span, omega):
        """
        Solves the pendulum ODE.

        Parameters:
            y0 (torch.Tensor): Initial condition [theta0, theta_dot0].
            t_span (torch.Tensor): Times at which to evaluate the solution.
            omega (float or torch.Tensor): Frequency parameter.

        Returns:
            torch.Tensor: Solution as (len(t_span), 2), where columns are [theta, theta_dot].
        """
        sol = self.solver(lambda t, y: self.pendulum_ode(t, y, omega), y0, t_span)
        return sol[:, 0]  # Return only theta values:
    def __init__(self, solver=odeint):
        """
        Initializes the Simple Pendulum solver.

        Parameters:
            solver: A function like `odeint` from `torchdiffeq` for solving the ODE.
        """
        super(SimplePendulumSolver, self).__init__()
        self.solver = solver

    def pendulum_ode(self, t, y, omega):
        """
        Defines the ODE system for the simple pendulum.

        Parameters:
            t (torch.Tensor): The time tensor (not used in autonomous system).
            y (torch.Tensor): The state vector [theta, theta_dot].
            omega (float or torch.Tensor): Frequency parameter for the pendulum.

        Returns:
            torch.Tensor: Time derivatives of theta and theta_dot.
        """
        theta, theta_dot = y
        return torch.stack([
            theta_dot,
            -omega * torch.sin(theta)
        ])

    def forward(self, y0, t_span, omega):
        """
        Solves the pendulum ODE.

        Parameters:
            y0 (torch.Tensor): Initial condition [theta0, theta_dot0].
            t_span (torch.Tensor): Times at which to evaluate the solution.
            omega (float or torch.Tensor): Frequency parameter.

        Returns:
            torch.Tensor: Solution as (len(t_span), 2), where columns are [theta, theta_dot].
        """
        sol = self.solver(lambda t, y: self.pendulum_ode(t, y, omega), y0, t_span)
        return sol[:, 0]  # Return only theta values