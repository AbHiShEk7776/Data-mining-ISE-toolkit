import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(x.dtype)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    z = x - np.max(x, axis=1, keepdims=True)
    exp_z = np.exp(z)   
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

ACT = {
    "sigmoid": (sigmoid,dsigmoid),
    "relu": (relu,drelu),
    "tanh": (tanh,dtanh),
    "softmax": (softmax,None),
}

class NNClassifier:
    
    def __init__(self, layer_sizes,activations = None, lr = 0.01, epochs = 1000):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.lr = lr
        self.epochs = epochs
        
        if activations is None:
            hidden = ["relu"] * (self.L - 1)
            output = ["softmax" if layer_sizes[-1] > 1 else "sigmoid"]
            self.activations = hidden + output
        else:
            assert len(activations) == self.L, "activations must match number of layers"
            self.activations = activations
            
            
        self.weights = []
        self.biases = []
        self._init_params()
        
    def _init_params(self):
        self.weights.clear()
        self.biases.clear()
        
        for l in range(self.L):
            fan_in = self.layer_sizes[l]
            fan_out = self.layer_sizes[l+1]
            act = self.activations[l]
            
            # He init for ReLU, Xavier for others
            if act == "relu":
                scale = np.sqrt(2. / fan_in)
            else:
                scale = np.sqrt(1. / fan_in)
            
            weight = np.random.randn(fan_in, fan_out) * scale
            bias = np.zeros((1, fan_out))
            self.weights.append(weight)
            self.biases.append(bias)

                
    def _forward(self, X):
        
        a = [X]
        z = []
        
        for l in range(self.L):
            zl = a[-1] @ self.weights[l] + self.biases[l]
            act = self.activations[l]
            f, _ = ACT[act]
            al = f(zl) if act != "softmax" else softmax(zl)
            
            z.append(zl)
            a.append(al)    
        return a, z
    
    def _loss(self, al, y):
        
        N = y.shape[0]
        C = al.shape[1]
        
        if C == 1:
            y = y.reshape(-1, 1)
            eps = 1e-9
            loss = -np.mean(y * np.log(al + eps) + (1 - y) * np.log(1 - al + eps))
            dZL = (al - y) / N
            return loss, dZL
        else:
            y_onehot = np.zeros((N, C))
            y_onehot[np.arange(N), y] = 1
            eps = 1e-9
            loss = -np.mean(np.sum(y_onehot * np.log(al + eps), axis=1))
            dZL = (al - y_onehot) / N
            return loss, dZL       
        
    def _backward(self, a, z, dZL):
        m = a[0].shape[0]
        dW = [None] * self.L
        db = [None] * self.L
        dZ = dZL
        
        for l in reversed(range(self.L)):
            
            a_prev = a[l]
            dW[l] = (a_prev.T @ dZ)
            db[l] = np.sum(dZ, axis=0, keepdims=True)
            
            if l > 0:
                da_prev = dZ @ self.weights[l].T
                act = self.activations[l-1]
                if act == "softmax":
                    raise ValueError("Softmax cannot be used in hidden layers")
                
                _, dact = ACT[act]
                dZ = da_prev * dact(z[l-1])
                
        return dW, db
    
    def _step(self, dW, db):
        for l in range(self.L):
            self.weights[l] -= self.lr * dW[l]
            self.biases[l] -= self.lr * db[l]

    def fit(self, X, y, verbose=False):
            for epoch in range(self.epochs):
                a, z = self._forward(X)
                loss, dZL = self._loss(a[-1], y)
                dW, db = self._backward(a, z, dZL)
                self._step(dW, db)
                
                if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                    print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
            a, _ = self._forward(X)
            al = a[-1]
            if al.shape[1] == 1:
                return (al > 0.5).astype(int).flatten()
            else:
                return np.argmax(al, axis=1)
    def score(self, X, y):
            y_pred = self.predict(X)
            return np.mean(y_pred == y)
            