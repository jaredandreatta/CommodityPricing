import numpy as np
import scipy 
from scipy import stats
import scipy.optimize as opt
from scipy.interpolate import CubicSpline
import statsmodels


class CompetitiveStorageModel:
    def __init__(self, a, b, delta, mu, sigma, rho, r = 0.0) -> None:
        """
        Initialize a competitive storage model object.

        Parameters:
        a (float): Intercept of the inverse demand function
        b (float): Slope of the inverse demand function. Note that b is nonnegative.
        delta (float): Depreciation rate of inventories
        mu (float): Mean of harvest shocks
        sigma (float): Standard deviation of harvest shocks
        rho (float): Autoregressive parameter for the harvest process
        r (float): Real interest rate. r is assumed to be 0 unless otherwise specified
        """
        self.a = a
        self.b = b
        self.delta = delta
        self.mu = mu
        self.sigma = sigma
        self.rho = rho
        self.r = r 

    def __repr__(self) -> str:
        return (f"CompetitiveStorageModel(a={self.a}, b={self.b}, r={self.r}, delta={self.delta}, "
                f"mu={self.mu}, sigma={self.sigma}, rho={self.rho})")

    def inverse_demand(self, z):
        """
        Generate a linear inverse demand curve.

        Parameters:
        z (np.ndarray): Array of harvest quantities

        Returns:
        np.ndarray: Array of estimated price values based on coinciding shock values.
        """
        return self.a + self.b * z

    def generate_shocks(self, n) -> np.ndarray:
        """
        Generate a series of harvest shocks based on the AR(1) process. We use the 
        parameters mu (mean) and sigma (std dev) to define the initial shock value
        and estimate the succeeding shock values as a linear autoregressive process.

        z_{t+1} - μ = ρ(z_{t}-μ) + σϵ_{t+1}

        Parameters:
        n (int): Number of time periods

        Returns:
        np.ndarray: Array of generated harvest shocks
        """
        shocks = np.zeros(n)
        shocks[0] = np.random.normal(self.mu, self.sigma)
        for t in range(1, n):
            shocks[t] = self.rho * (shocks[t-1] - self.mu) + np.random.normal(0, self.sigma)
        return shocks
    
    def initial_state(self, inventory, z) -> float:
        """
        Calculates initial state variable for the price simulation function.

        Parameters:
        inventory (int): Initial inventory quantity
        z (np.ndarray): Array of harvest quantities

        Returns:
        float: Value of initial state variable (amount on hand)
        """
        return inventory+z[0]
    
    def eq_price_fn(self):
        pass
    

