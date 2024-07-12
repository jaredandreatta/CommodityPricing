import numpy as np

class CompetitiveStorageModel:
    def __init__(self, a: float, b: float, delta: float, mu: float, sigma: float, rho: float, r: float = 0.0) -> None:
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

        # Placeholder variables that will be modified later
        self.P_t = None  # Current Price
        self.z_t = None  # Current Shock
        self.I_t = None  # Current Inventory
        self.x_t = None  # Current amount on-hand

    def __repr__(self) -> str:
        """
        Return a string representation of the model parameters.
        """
        return (f"CompetitiveStorageModel(a={self.a}, b={self.b}, r={self.r}, delta={self.delta}, "
                f"mu={self.mu}, sigma={self.sigma}, rho={self.rho})")

    def inverse_demand(self, z):
        """
        Generate an inverse demand curve.

        Parameters:
        z (np.ndarray): Array of randomly generated shocks
        """
        return self.a + self.b * z

    def generate_shocks(self, n: int) -> np.ndarray:
        """
        Generate a series of harvest shocks based on the AR(1) process. We use the 
        parameters mu (mean) and sigma (std dev) in this process defined by the model
        to randomy generate these shocks.

        Parameters:
        n (int): Number of shocks to generate

        Returns:
        np.ndarray: Array of generated harvest shocks
        """
        shocks = np.zeros(n)
        shocks[0] = np.random.normal(self.mu, self.sigma)
        for t in range(1, n):
            shocks[t] = self.rho * (shocks[t-1] - self.mu) + np.random.normal(0, self.sigma)
        return shocks

    

