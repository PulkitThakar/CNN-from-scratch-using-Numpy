import tensorflow as tf
import numpy as np

fac = 5
Mnist = tf.keras.datasets.mnist

class Linear_Layer:

    def __init__(self, in_dim, out_dim, alpha = 0.01, Theta = None, bias = None):
        self.alpha = alpha
        if Theta == None:
            self.Theta = np.random.randn(in_dim, out_dim)/fac

        else:
            self.Theta = Theta

        if bias == None:
            self.bias = np.random.randn(out_dim)/fac

        else:
            self.bias = bias
        

    def forward_pass(self, X):
        self.X = X
        self.z = np.matmul(X, self.Theta) + self.bias
        return self.z

    
    def backprop(self, grad_previous):
        t= self.X.shape[0]
        self.grad = np.matmul((self.X.transpose()), grad_previous)/t
        self.grad_bias = (grad_previous.sum(axis=0))/t
        self.grad_a = np.matmul(grad_previous, self.Theta.transpose())
        return self.grad_a



    def applying_sgd(self):
            self.Theta = self.Theta - (self.alpha*self.grad)
            self.bias = self.bias - (self.alpha*self.grad_bias)

class softmax:

    def __init__(self):
        pass
    
    def expansion(self, t):
        (a,) = t.shape
        Y = np.zeros((a,10))
        for i in range(0,a):
            Y[i,t[i]] = 1
        return Y
    
    def forward_pass(self, z):
        self.z =  z
        (p,t) = self.z.shape
        self.a = np.zeros((p,t))
        for i in range(0,p):
            for ii in range(0,t):
                self.a[i,ii] = (np.exp(self.z[i,ii]))/(np.sum(np.exp(self.z[i,:])))
        return self.a

    def backprop(self, Y):
        y = self.expansion(Y)
        self.grad = (self.a - y)
        return self.grad

    def applying_sgd(self):
        pass

class relu:
    def __init__(self):
        pass

    def forward_pass(self, z):
        
        if (len(z.shape) == 3):

            z_temp = z.reshape((z.shape[0], z.shape[1]*z.shape[2]))
            z_temp_1 = self.forward_pass(z_temp)
            self.a_1 = z_temp_1.reshape((z.shape[0], z.shape[1], z.shape[2]))
            return (self.a_1)

        else:
            (p,t) = z.shape
            self.a = np.zeros((p,t))
            for i in range(0,p):
                for ii in range(0,t):
                        self.a[i,ii] = max([0,z[i,ii]])
            return self.a

    def derivative(self, a):
        if a>0:
            return 1
        else:
            return 0
    
    def backprop(self, grad_previous):
        
        if (len(grad_previous.shape)==3):

            (d, p, t) = grad_previous.shape
            self.grad = np.zeros((d, p, t))
            
            for i in range(d):
                for ii in range(p):
                    for iii in range(t):
                        self.grad[i, ii, iii] = (grad_previous[i, ii, iii] * self.derivative(self.a_1[i, ii, iii]))
            
            return (self.grad)

        else:
            (p,t) = grad_previous.shape
            self.grad = np.zeros((p,t))
            for i in range(p):
                for ii in range(t):
                    self.grad[i,ii] = grad_previous[i,ii] * self.derivative(self.a[i,ii])
            return (self.grad)

    
    def applying_sgd(self):
        pass

class padding():
    
    def __init__(self, pad = 1):
        self.pad = pad

    def forward_pass(self, data):
        X = np.pad(data , ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),'constant', constant_values=0)
        return X

    def backprop(self, y):
        return (y[:, 1:(y.shape[1]-1),1:(y.shape[2]-1)])

    def applying_sgd(self):
        pass

class Convolutional_Layer:
    def __init__(self, filter_dim = 3, stride = 1, pad = 1, alpha=0.01):
        self.filter_dim = filter_dim
        self.stride = stride
        self.filter = np.random.randn(self.filter_dim, self.filter_dim)
        self.filter = self.filter/self.filter.sum()
        self.bias = np.random.rand()/10
        self.pad = pad
        self.alpha = alpha

    def convolving(self, X, fil, dimen_x, dimen_y):
        z = np.zeros((dimen_x, dimen_y))
        for i in range(dimen_x):
            for ii in range(dimen_y):
                temp = np.multiply(X[i : i+fil.shape[0], ii : ii+fil.shape[1]], fil)
                z[i,ii] = temp.sum()
        return z
        
        
    def forward_pass(self, X):
        self.X = X
        (d, p, t) = self.X.shape
        dimen_x = int(((p - self.filter_dim)/self.stride) + 1)
        dimen_y = int(((t - self.filter_dim)/self.stride) + 1)
        self.z = np.zeros((d, dimen_x, dimen_y))
        for i in range(d):
            self.z[i] = (self.convolving(self.X[i], self.filter, dimen_x, dimen_y) + self.bias)

        return self.z

    def backprop(self, grad_z):
        (d, p, t) = grad_z.shape
        filter_1 = np.flip((np.flip(self.filter, axis = 0)), axis = 1)
        self.grads = np.zeros((d, p, t))
        for i in range(d):
            self.grads[i] = self.convolving(np.pad(grad_z[i], ((1,1), (1,1)), 'constant', constant_values = 0), filter_1, p, t)

        self.grads = np.pad(self.grads, ((0,0),(1,1),(1,1)), 'constant', constant_values = 0)

        self.grad_filter = np.zeros((self.filter_dim, self.filter_dim))

        for i in range(self.filter_dim):
            for ii in range(self.filter_dim):
                self.grad_filter[i, ii] = (np.multiply(grad_z, self.X[:, i:p+i, ii:t+ii])).sum()
        self.grad_filter = self.grad_filter/(d)

        self.grad_bias = (grad_z.sum())/(d)
        return self.grads

    def applying_sgd(self):
        self.filter = self.filter - (self.alpha*self.grad_filter)
        self.bias = self.bias - (self.alpha*self.grad_bias)


class pooling:

    def __init__(self, pool_dim = 2, stride = 2):
        self.pool_dim = pool_dim
        self.stride = stride

    def forward_pass(self, data):
        (q, p, t) = data.shape
        z_x = int((p - self.pool_dim) / self.stride) + 1
        z_y = int((t - self.pool_dim) / self.stride) + 1
        after_pool = np.zeros((q, z_x, z_y))
        for ii in range(0, q):
            liss = []
            for i in range(0,p,self.stride):
                for j in range(0,t,self.stride):
                    if (i+self.pool_dim <= p) and (j+self.pool_dim <= t):
                        temp = data[ii, i:(i+(self.pool_dim)), j:(j+(self.pool_dim))]
                        temp_1 = np.max(temp)
                        liss.append(temp_1)
            liss = np.asarray(liss)
            liss = liss.reshape((z_x, z_y))
            after_pool[ii] = liss
            del liss
        return after_pool

    def backprop(self, pooled):
        (a,b,c) = pooled.shape   
        cheated = np.zeros((a,2*b,2*c))
        for k in range(0, a):
            pooled_transpose_re = pooled[k].reshape((b*c))
            count = 0
            for i in range(0, 2*b, self.stride):
                for j in range(0, 2*c, self.stride):
                    cheated[k, i:(i+(self.stride)),j:(j+(self.stride))] = pooled_transpose_re[count]
                    count = count+1
        return cheated

    def applying_sgd(self):
        pass


class Neural_Network:

    def __init__(self, Network):
        self.Network = Network

    def forward_pass(self, X):
        n = X
        for i in self.Network:
            n = i.forward_pass(n)
            
            
        return n
    
    def backprop(self, Y):
        m = Y
        count = 1
        for i in (reversed(self.Network)):
            m = i.backprop(m)

    def applying_sgd(self):
        for i in self.Network:
            i.applying_sgd()


class reshaping:
    
    def __init__(self):
        pass

    def forward_pass(self, a):
        self.shape_a = a.shape
        
        self.final_a = a.reshape(self.shape_a[0], self.shape_a[1]*self.shape_a[2])
        return self.final_a
    
    def backprop(self, q):
        return (q.reshape(self.shape_a[0], self.shape_a[1], self.shape_a[2]))

    def applying_sgd(self):
        pass


class cross_entropy:

    def __init__(self):
        pass
    
    def expansion(self, t):
        (a,) = t.shape
        Y = np.zeros((a,10))
        for i in range(0,a):
            Y[i,t[i]] = 1
        return Y

    def loss(self, A, Y):
        exp_Y = self.expansion(Y)
        (u,i) = A.shape
        loss_matrix = np.zeros((u,i))
        for j in range(u):
            for jj in range(i):
                if exp_Y[j,jj] == 0:
                    loss_matrix[j,jj] = np.log(1 - A[j,jj])
                else:
                    loss_matrix[j,jj] = np.log(A[j,jj])
        

        return ((-(loss_matrix.sum()))/u)

class accuracy:
    def __init__(self):
        pass

    def value(self, out, Y):
        self.out = np.argmax(out, axis=1)
        p = self.out.shape[0]
        total = 0
        for i in range(p):
            if Y[i]==self.out[i]:
                total += 1
        return total/p



(Xtr, Ytr), (Xte, Yte) = Mnist.load_data()
X_testing = Xtr[:,:,:]
Y_testing = Ytr[:]
#X_testing = X_testing.reshape((60000, 28*28))
X_testing = X_testing/255
al = 0.3
stopper = 85.0

complete_NN = Neural_Network([
                                
                                padding(),
                                Convolutional_Layer(),
                                pooling(),
                                relu(),
                                padding(),
                                Convolutional_Layer(),
                                pooling(),
                                relu(),
                                reshaping(),
                                Linear_Layer(7*7, 24, alpha = al),
                                relu(),
                                Linear_Layer(24, 10, alpha = al),
                                softmax()

                                ])
CE = cross_entropy()

acc = accuracy()
epochs = 100
broke = 0
batches = 6000
for i in range(epochs):
    k = 0
    for ii in range(batches, 60001, batches):
        
        out = complete_NN.forward_pass(X_testing[k:ii])
        print("epoch:{} \t batch: {} \t loss: \t {}".format(i+1, int(ii/batches), CE.loss(out, Y_testing[k:ii])), end="\t")
        accur = acc.value(out, Y_testing[k:ii])*100
        print("accuracy: {}".format(accur))
        
        if accur >= stopper:
            broke = 1
            break
        complete_NN.backprop(Y_testing[k:ii])
        complete_NN.applying_sgd()
        k = ii
        
    if broke == 1:
        break
    

out = complete_NN.forward_pass(X_testing)
print("The final loss is {}".format(CE.loss(out, Y_testing)))
print("The final accuracy on train set is {}".format(acc.value(out, Y_testing)*100))
Xtest = Xte/255
#Xtest = Xte.reshape((10000,28*28))/255
out_1 = complete_NN.forward_pass(Xtest)
print("The accuracy on test set is {}".format(acc.value(out_1, Yte)*100))


