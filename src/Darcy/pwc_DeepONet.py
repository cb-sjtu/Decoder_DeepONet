import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, io
from sklearn.preprocessing import StandardScaler

import deepxde as dde
from deepxde.backend import tf
import tensorflow_addons as tfa
import random

def get_data(filename, ndata):
    # 5->85x85, 6->71x71, 7->61x61, 10->43x43, 12->36x36, 14->31x31, 15->29x29

    r = 7
    s = int(((421 - 1) / r) + 1)

    # # Data is of the shape (number of samples = 1024, grid size = 421x421)
    data = io.loadmat(filename)
    x_branch = data["coeff"][:ndata, ::r, ::r].astype(np.float32) * 0.1 - 0.75
    y = data["sol"][:ndata].astype(np.float32) * 100
    # The dataset has a mistake that the BC is not 0.
    y[:, 0, :] = 0
    y[:, -1, :] = 0
    y[:, :, 0] = 0
    y[:, :, -1] = 0
    
    y_unaligned=np.zeros((ndata,s,s))
    
    
    if ndata==1000:
        random.seed(12)
        # x_branchs=np.load('x_branch_una_train.npy')
        # gridss=np.load('gridss_train.npy')
        # y_unaligned=np.load('y_unaligned_train.npy')
    elif ndata==20:
        random.seed(34)
        # x_branchs=np.load('x_branch_una_test.npy')[0:s*s*20]
        # gridss=np.load('gridss_test.npy')[0:s*s*20]
        # y_unaligned=np.load('y_unaligned_test.npy')[0:s*s*20]
    for i in range(ndata):
        
        sub_y_1 =random.sample(range(1,420),s)
        sub_y_2 =random.sample(range(1,420),s)
        sub_y_1=sorted(sub_y_1)
        sub_y_2=sorted(sub_y_2)
        a= y[i, sub_y_1].astype(np.float32)
        y_unaligned[i]=a[ : ,sub_y_2]
        

    

    x_branch = x_branch.reshape(ndata, s * s)
    
    y_unaligned = y_unaligned.reshape(ndata, s * s)
    x_branchs=np.zeros((ndata*s*s,s*s))
   
    
    for i in range(ndata):
        for j in range (s):
            x_branchs[s*i+j]=x_branch[i]
            
    # np.save('x_branch_una_train_r7',x_branchs)
    # np.save('gridss_test_r7',gridss)
    # np.save('y_unaligned_test_r7',y_unaligned)
    
    


    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    
    

    x_branch = x_branch.reshape(ndata, s * s)
    x = (x_branch, grid)
    # x = (x_branch, grid)
   
    return x, y_unaligned


def dirichlet(inputs, output):
    x_trunk = inputs[1]
    x, y = x_trunk[:, 0], x_trunk[:, 1]
    return 20 * x * (1 - x) * y * (1 - y) * (output + 1)


def main():
    x_train, y_train = get_data("piececonst_r421_N1024_smooth1.mat", 1000)
    x_test, y_test = get_data("piececonst_r421_N1024_smooth2.mat", 200)
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
    s=61
    m = s ** 2
    activation = "relu"
    branch = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m,)),
            tf.keras.layers.Reshape((s, s, 1)),
            tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Conv2D(128, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=activation),
            tf.keras.layers.Dense(256),
        ]
    )
    branch.summary()
    net = dde.maps.DeepONetCartesianProd(
        [m, branch], [2,256,512,512,256], activation, "Glorot normal"
    )

    scaler = StandardScaler().fit(y_train)
    std = np.sqrt(scaler.var_.astype(np.float32))

    def output_transform(inputs, outputs):
        return outputs * std + scaler.mean_.astype(np.float32)

    net.apply_output_transform(output_transform)
    # net.apply_output_transform(dirichlet)

    model = dde.Model(data, net)
    model.compile(
        "adam", lr=0.00003,
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
    )
    losshistory, train_state = model.train(epochs=1000, batch_size=None,display_every=100,model_save_path="deeponet_model_2/")
    dde.saveplot(losshistory, train_state,issave=True,isplot=True)
    
    y_pred = model.predict(data.test_x)
    y_pred=np.transpose(y_pred)
    test_y=np.transpose(data.test_y)
    grid=x_test[1]
    N_points=841
    results=[]
    for k in range(200):
        spacename="./deeponet_model_2/test"+str(k)+".dat"
        test=np.reshape(test_y[:,k],(N_points,1))
        pre=np.reshape(y_pred[:,k],(N_points,1))
        error=np.abs(pre-test)
        result=np.hstack((grid,test,pre,error))
        np.savetxt(spacename,result)
        with open(spacename, 'r+', encoding='utf-8') as f:
            content = f.read()
            f.seek(0, 0)
            f.write('VARIABLES="x","y","U","U_p","E"'+'\n'+'Zone i=43 j=43'+'\n'+content)
        results=np.append(results,result)
    results=np.reshape(results,(200,N_points,5))
   
    print(
        "# Parameters:",
        np.sum(
            [
                np.prod(v.get_shape().as_list())
                for v in tf.trainable_variables()
            ]
        ),
    )


if __name__ == "__main__":
    main()
