import numpy as np

def gaussian_func(y, mu, sigma):
    #y has the shape of [samples, 1]
    #only for 1d gaussian
    samples = y.shape[0]
    num_tree, leaf_num, _, _ = mu.shape
    #res = torch.zeros(samples, num_tree, leaf_num)
    #print(y.shape)
    y = np.reshape(y, [samples, 1, 1])
    y = np.repeat(y, num_tree, 1)
    y = np.repeat(y, leaf_num, 2)   

    mu = np.reshape(mu, [1, num_tree, leaf_num])
    mu = mu.repeat(samples, 0)

    sigma = np.reshape(sigma, [1, num_tree, leaf_num])
    sigma = sigma.repeat(samples, 0)  

    res = 1.0 / np.sqrt(2 * 3.14 * (sigma + 1e-9)) * \
         (np.exp(- (y - mu) ** 2 / (2 * (sigma + 1e-9))) + 1e-9)

    return res

if __name__ == "__main__":
    y = np.ones([3, 1])
    mu = np.ones([3, 4, 1, 1])
    sigma = np.ones([3, 4, 1, 1])
    res = gaussian_func(y, mu, sigma)
    print(res.shape)
