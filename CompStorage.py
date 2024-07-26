import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import norm

class CompetitiveStorageModel:
    def __init__(self, a, b, delta, mu, sigma, rho, r=0.0) -> None:
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

    def P(self, z):
        """
        Generate a linear inverse demand curve.

        Parameters:
        z (np.ndarray): Array of harvest quantities

        Returns:
        np.ndarray: Array of estimated price values based on coinciding shock values.
        """
        return self.a + self.b * z

    def P_inv(self, p):
        """
        Generate the inverse of the linear inverse demand curve.

        Parameters:
        p (np.ndarray): Array of price values

        Returns:
        np.ndarray: Array of estimated harvest quantities based on price values.
        """
        return (p - self.a) / self.b

    def generate_shocks(self, n) -> np.ndarray:
        """
        Generate a series of harvest shocks based on the AR(1) process. We use the 
        parameters mu (mean) and sigma (std dev) to define the initial shock value
        and evolve them according to the autoregressive parameter rho.

        Parameters:
        n (int): Number of shocks to generate

        Returns:
        np.ndarray: Array of generated shocks
        """
        shocks = np.zeros(n)
        shocks[0] = np.random.normal(self.mu, self.sigma)
        for t in range(1, n):
            shocks[t] = self.mu + self.rho * (shocks[t-1] - self.mu) + np.random.normal(0, self.sigma)
        return shocks

    def initial_state(self, initial_inventory, z_0):
        """
        Calculate the initial state variable given initial inventory and first harvest quantity.

        Parameters:
        initial_inventory (float): Initial inventory quantity
        z (float): First harvest quantity

        Returns:
        float: Initial state variable
        """
        return z_0 + (1 - self.delta) * initial_inventory

    def state_variables(self, initial_inventory, z, f):
        """
        Calculate state variables for the given initial inventory and harvest quantities.

        Parameters:
        initial_inventory (float): Initial inventory quantity
        z (np.ndarray): Array of harvest quantities
        f (callable): Equilibrium price function

        Returns:
        np.ndarray: Array of state variables
        """
        n = len(z)
        x = np.zeros(n)
        x[0] = self.initial_state(initial_inventory, z[0])
        for t in range(1, n):
            x[t] = (1 - self.delta) * (x[t-1] - self.P_inv(f(x[t-1]))) + z[t]
        return x

    def eq_price_fn(self, z, iter=1000):
        """
        Calculate the equilibrium price function based on equation (12) in the paper.

        Parameters:
        z (np.ndarray): Array of harvest quantities
        iter (int): Number of iterations for convergence

        Returns:
        callable: Function that computes the equilibrium price for given state variables
        """
        n = len(z)
        x = np.linspace(min(z), max(z), n)  # Using a range of state variables for interpolation
        f = self.P(x)  # Initial price guess

        for _ in range(iter):
            f_new = np.zeros_like(f)
            for t in range(n):
                integrand = lambda z_val: np.interp(
                    z_val + (1 - self.delta) * (x[t] - self.P_inv(f[t])), x, f)
                expectation, _ = quad(integrand, self.mu - 5*self.sigma, self.mu + 5*self.sigma)
                expectation /= (2 * self.sigma)
                f_new[t] = max(self.r * expectation, self.P(x[t]))
            if np.allclose(f, f_new, atol=1e-6):
                break
            f = f_new

        # Interpolating the final function
        f_func = interp1d(x, f, kind='cubic', fill_value="extrapolate")
        return f_func





