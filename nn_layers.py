import numpy as np
from skimage.util.shape import view_as_windows
from skimage.measure import block_reduce


class nn_convolutional_layer:
    """
        convolutional Layer
    """
    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        out = np.zeros((x.shape[0], self.W.shape[0],
                        x.shape[2] - self.W.shape[2] + 1, x.shape[3] - self.W.shape[3] + 1))
        for i in range(x.shape[0]):
            for j in range(self.W.shape[0]):
                out[i][j] = self.conv(x[i], self.W[j], out[i][j].shape) + self.b[0][j]
        return out

    def backprop(self, x, dLdy):
        dLdx = np.zeros(x.shape)
        dLdW = np.zeros(self.W.shape)
        dLdb = np.zeros(self.b.shape)

        for i in range(dLdy.shape[0]):
            for j in range(dLdW.shape[0]):
                dLdb[0][j] += np.sum(dLdy[i][j])
                for k in range(dLdW.shape[1]):
                    dLdW[j][k] += self.conv(x[i][k], dLdy[i][j], dLdW[j][k].shape)

                    dLdy_0 = np.pad(dLdy[i][j], ((dLdW.shape[2]-1, dLdW.shape[3]-1), (dLdW.shape[2]-1, dLdW.shape[3]-1)),
                                    'constant', constant_values=0)
                    dLdW_reverse = np.flip(self.W[j][k])
                    dLdx[i][k] += self.conv(dLdy_0, dLdW_reverse, dLdx[i][k].shape)

        return dLdx, dLdW, dLdb

    def conv(self, a, b, res_shape):
        res = view_as_windows(a, b.shape)
        res = res.reshape((res_shape + (-1,)))
        res = res.dot(b.reshape(-1, 1))
        res = np.squeeze(res, axis=2)
        return res

class nn_max_pooling_layer:
    """
        max pooling Layer
    """
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        out = block_reduce(x, (1, 1, 2, 2), np.max)
        return out

    def backprop(self, x, dLdy):
        out = block_reduce(x, (1, 1, 2, 2), np.max)
        mask = np.equal(x, out.repeat(2, axis=2).repeat(2,axis=3)).astype(int)
        dLdx = np.multiply(mask, dLdy.repeat(2, axis=2).repeat(2, axis=3))
        return dLdx



class nn_fc_layer:
    """
        fully connected linear Layer
    """
    def __init__(self, input_size, output_size, std=1):
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size))

    def forward(self,x):
        x = x.reshape((x.shape[0], -1))
        return x@self.W.T + self.b.T

    def backprop(self,x,dLdy):
        shape = x.shape
        x = x.reshape((x.shape[0], -1))

        dLdW = np.zeros(self.W.shape)
        dLdb = np.zeros(self.b.shape).T
        dLdx = np.zeros(x.shape)
        for n in range(x.shape[0]):
            dLdW += np.outer(dLdy[n], x[n])
            dLdb += dLdy[n]
            dLdx[n] = dLdy[n]@self.W
        dLdx = dLdx.reshape(shape)
        return dLdx, dLdW, dLdb

    def update_weights(self,dLdW,dLdb):
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

class nn_activation_layer:
    """
        use ReLU as activation function
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        x[x<0] = 0
        return x

    def backprop(self, x, dLdy):
        x[x <= 0] = 0
        x[x > 0] = 1
        dLdx = np.multiply(x, dLdy)
        return dLdx


class nn_softmax_layer:
    """
        softmax Layer
    """
    def __init__(self):
        pass

    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def backprop(self, x, dLdy):
        dLdx = np.zeros(x.shape)
        for n in range(x.shape[0]):
            y = np.exp(x[n]) / np.sum(np.exp(x[n]))
            y = y.reshape(-1, 1)
            dydx = np.diagflat(y) - y@y.T
            dLdx[n] = dLdy[n]@dydx
        return dLdx


class nn_cross_entropy_layer:
    """
        cross entropy Layer
    """
    def __init__(self):
        pass

    def forward(self, x, y):
        p = x[:, 0].reshape(y.shape[0],1)
        loss = -(np.multiply(1-y, np.log(p)) + np.multiply(y, np.log(1-p)))
        return np.mean(loss)

    def backprop(self, x, y):
        dLdy = 1/y.shape[0]
        dydx = np.zeros(x.shape)
        for n in range(y.shape[0]):
            dydx[n][y[n]] = -1/x[n][y[n]]
        dLdx = dLdy * dydx
        return dLdx
