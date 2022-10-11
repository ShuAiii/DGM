import pdb
from functools import lru_cache

from scipy import interpolate
import scipy as sy
import numpy as np

U_BAR = 40
E1 = -3
E2 = 1


class SpreadOption:

    def __init__(self,
                 strike,
                 r,
                 sigma1,
                 sigma2,
                 rho,
                 T,
                 t,
                 grid_size):

        self.x1_int = np.log(100 / strike)
        self.x2_int = np.log(100 / strike)
        self.strike = strike
        self.r = r

        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.tau = T - t
        self.grid_size = grid_size

        # Fast-Fourier grid spacing
        grid_spacing = np.linspace(0, grid_size - 1, grid_size)

        # u1 and u2 forms the mesh in frequency space
        u_bar1 = self.find_bound(self.x1_int)
        u_bar2 = self.find_bound(self.x2_int)
        eta1 = 2 * u_bar1 / self.grid_size
        eta2 = 2 * u_bar2 / self.grid_size
        u1 = -u_bar1 + eta1 * grid_spacing
        u2 = -u_bar2 + eta2 * grid_spacing

        # x1 and x2 forms the mesh in log-stock space
        eta_star1 = np.pi / u_bar1
        eta_star2 = np.pi / u_bar2
        x1 = -0.5 * self.grid_size * eta_star1 + eta_star1 * grid_spacing
        x2 = -0.5 * self.grid_size * eta_star2 + eta_star2 * grid_spacing
        self.x1 = x1
        self.x2 = x2

        # Create mesh on a complex plane from the frequency space mesh
        i_vec = u1 + 1j * E1
        j_vec = u2 + 1j * E2
        mat = np.stack(np.meshgrid(i_vec, j_vec), axis=2).transpose([1, 0, 2])

        # Vectorized characteristic function
        phi_mat = self.calculate_characteristic_func(mat)

        # Vectorized complex gamma function
        p_hat_mat = self.calculate_gamma_func(mat)

        # Create a sign matrix for H and C matrix calculation
        x = np.arange(self.grid_size)
        sign_mat = (-1) ** np.sum(np.meshgrid(x, x), axis=0)

        # Calculate the H matrix
        self.h_mat = sign_mat * phi_mat * p_hat_mat

        # Calculate the C matrix
        x1_mat, x2_mat = np.meshgrid(x1, x2)
        self.c_mat = sign_mat * eta1 * eta2 * (self.grid_size / (2 * np.pi)) ** 2 * \
                     np.exp(-E1 * x1_mat - E2 * x2_mat).transpose()

        # Not needed anymore after interpolation
        self.p1 = np.abs(self.x1_int - x1).argmin()
        self.p2 = np.abs(self.x2_int - x2).argmin()

        # Create the pricer
        self.pricer = self.create_pricer()

    def find_bound(self,
                   x0: float):

        for i in range(self.grid_size):
            u_test = np.pi * (i - self.grid_size / 2) / x0
            if u_test > U_BAR:
                return u_test

        return U_BAR

    def calculate_gamma_func(self,
                             u_matrix: np.ndarray):

        p_hat_mat = sy.special.gamma(1j * u_matrix.sum(axis=2) - 1) * sy.special.gamma(-1j * u_matrix[..., 1]) / \
                    sy.special.gamma(1j * u_matrix[..., 0] + 1)

        return p_hat_mat

    def calculate_characteristic_func(self,
                                      u_matrix: np.ndarray):

        r_vec = self.r * np.ones([2, 1])
        sigma_vec = np.array([self.sigma1 ** 2, self.sigma2 ** 2]).reshape([2, 1])
        sigma_mat = np.array(
            [[self.sigma1 ** 2, self.rho * self.sigma1 * self.sigma2],
             [self.rho * self.sigma1 * self.sigma2, self.sigma2 ** 2]]
        )

        phi_mat = np.exp(
            1j * np.matmul(u_matrix, (r_vec - 0.5 * sigma_vec)).sum(axis=2) * self.tau - \
            (np.matmul(u_matrix, sigma_mat) * u_matrix).sum(axis=2) * self.tau * 0.5
        )

        return phi_mat

    def create_pricer(self):

        v_mat = self.strike * np.exp(-self.r * self.tau) * np.real(self.c_mat * np.fft.ifft2(self.h_mat))

        return interpolate.interp2d(self.x1, self.x2, v_mat, kind='cubic')

    def valuation(self,
                  s1: float,
                  s2: float):

        x1 = np.log(s1 / self.strike)
        x2 = np.log(s2 / self.strike)

        return self.pricer(x2, x1)

    def __call__(self,
                 s1: float,
                 s2: float):

        return self.valuation(s1, s2)


spread = SpreadOption(1, 0.05, 0.4, 0.2, 0.5, 5, 1, 2048)
print(spread(100, 100))
pdb.set_trace()
