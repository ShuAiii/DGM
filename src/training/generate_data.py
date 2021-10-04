import numpy as np
from scipy.stats import norm


def bsm_data(batch_size: int,
             variables: dict,
             is_validation: bool = False) -> np.array:

    t = np.random.uniform(variables["t"][0], variables["t"][1], batch_size)
    s = np.random.uniform(variables["S"][0], variables["S"][1], batch_size)
    sigma = np.random.uniform(variables["sigma"][0],
                              variables["sigma"][1], batch_size)
    r = np.random.uniform(variables["r"][0], variables["r"][1], batch_size)
    maturity = variables["T"]

    def bsm_call(term, stock, rates, vol):
        d1 = (np.log(stock) + (rates + vol ** 2 / 2) * (maturity - term)) \
             / (vol * np.sqrt(maturity - term))
        d2 = d1 - vol * np.sqrt(maturity - term)
        return stock * norm.cdf(d1) - np.exp(-rates * term) * norm.cdf(d2)

    params = [t, s, r, sigma]

    if is_validation:
        params.append(bsm_call(t, s, r, sigma))

    return np.stack(params, axis=1).astype(np.float32)


PDE_DATA = {
    "bsm_call": bsm_data
}
