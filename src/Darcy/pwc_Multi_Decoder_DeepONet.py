from deepxde.nn.tensorflow_compat_v1.mionet import MIONetCartesianProd,MIONet_CNN
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, io
from sklearn.preprocessing import StandardScaler

import deepxde as dde
from deepxde.backend import tf
import tensorflow_addons as tfa
import random
def dirichlet(inputs, output):
    x_trunk = inputs[1]
    x, y = x_trunk[:, 0], x_trunk[:, 1]
    return 20 * x * (1 - x) * y * (1 - y) * (output + 1)


def get_data(filename, ndata):
    # 5->85x85, 6->71x71, 7->61x61, 10->43x43, 12->36x36, 14->31x31, 15->29x29
    r = 7
    s = int(((421 - 1) / r) + 1)

    # # Data is of the shape (number of samples = 1024, grid size = 421x421)
    data = io.loadmat(filename)
    x_branch = data["coeff"][:ndata, ::r, ::r].astype(np.float32) * 0.1 - 0.75
    # y = data["sol"][:ndata].astype(np.float32) * 100
    # # The dataset has a mistake that the BC is not 0.
    # y[:, 0, :] = 0
    # y[:, -1, :] = 0
    # y[:, :, 0] = 0
    # y[:, :, -1] = 0
    
    # y_unaligned=np.zeros((ndata,s,s))
    # grids=[]
    # grids.append(np.linspace(0, 1, 421, dtype=np.float32))
    # grids.append(np.linspace(0, 1, 421, dtype=np.float32))
    # gridss = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    # gridss=gridss.reshape(421,421,2)
    # grid=np.zeros((ndata,s,s,2))
    
    if ndata==1000:
        random.seed(12)
        # x_branchs=np.load('x_branch_una_train.npy')
        gridss=np.load('gridss_train_r7.npy').reshape(ndata, s * s,2)
        
        y_unaligned=np.load('y_unaligned_train_r7.npy').reshape(ndata, s * s)
        label_average=np.zeros((s*s))
        label_averages=np.zeros((ndata,s*s))
        for i in range(ndata):
            label_average=y_unaligned[i]+label_average
        label_average=label_average/ndata
        for i in range(ndata):
            label_averages[i]=label_average
    elif ndata==200:
        random.seed(34)
        # x_branchs=np.load('x_branch_una_test.npy')[0:s*s*20]
        gridss=np.load('gridss_test_r7.npy')[0:s*s*200].reshape(ndata, s * s,2)
        # print(gridss[1])
        
        y_unaligned=np.load('y_unaligned_test_r7.npy')[0:s*s*200].reshape(ndata, s * s)
        label_averages=0
    # for i in range(ndata):
        
    #     sub_y_1 =random.sample(range(1,420),s)
    #     sub_y_2 =random.sample(range(1,420),s)
    #     sub_y_1=sorted(sub_y_1)
    #     sub_y_2=sorted(sub_y_2)
    #     a= y[i, sub_y_1].astype(np.float32)
    #     y_unaligned[i]=a[ : ,sub_y_2]
    #     grida = gridss[sub_y_1]
    #     grid[i]=grida[: ,sub_y_2]

    

    # x_branch = x_branch.reshape(ndata, s * s)
    
    # y_unaligned = y_unaligned.reshape(ndata* s * s,1)
    # x_branchs=np.zeros((ndata*s*s,s*s))
    # gridss=grid.reshape(ndata*s*s,2)
    
    # for i in range(ndata):
    #     for j in range (s):
    #         x_branchs[s*i+j]=x_branch[i]
            
    # np.save('x_branch_una_train_r10',x_branchs)
    # np.save('gridss_train_r10',gridss)
    # np.save('y_unaligned_train_r10',y_unaligned)
    x_branch = x_branch.reshape(ndata, s * s)
    
    
    
    x = [x_branch,gridss]
    
    return x, y_unaligned,label_averages

def main():
    
    x_train, y_train,label_averages = get_data("piececonst_r421_N1024_smooth1.mat", 1000)
    x_test, y_test,label_averages_0 = get_data("piececonst_r421_N1024_smooth2.mat", 200)
    x_train=(x_train[0],label_averages,x_train[1])
    x_test=(x_test[0],label_averages[0:200],x_test[1])
    data = dde.data.Quadruple(x_train, y_train, x_test, y_test)
    s=61
    m = s ** 2
    activation = (
        ["relu", "relu", "relu"]
    )
    branch_1 = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m,)),
            tf.keras.layers.Reshape((s, s, 1)),
            tf.keras.layers.Conv2D(16, (5, 5), strides=2, activation="relu"),
            tf.keras.layers.Conv2D(8, (5, 5), strides=2, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(20),
        ]
    )
    branch_1.summary()
    branch_1=[m,branch_1]
  
    branch_2 = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m,)),
            tf.keras.layers.Reshape((s, s, 1)),
            tf.keras.layers.Conv2D(16, (5, 5), strides=2, activation="relu"),
            tf.keras.layers.Conv2D(8, (5, 5), strides=2, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(20),
        ]
    )
    branch_2.summary()
    branch_2=[m,branch_2]

    trunk =  tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(m,2,)),
                    tf.keras.layers.Reshape((s, s,2)),
                    tf.keras.layers.Conv2D(16, (5,5), strides=2, activation="relu"),
                    tf.keras.layers.Conv2D(8, (5, 5), strides=2, activation="relu"),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation="relu"),
                    tf.keras.layers.Dense(20),
                ]
            )
    trunk.summary()
    trunk=[2,trunk]

    
    dot= tf.keras.Sequential(
        [   tf.keras.layers.InputLayer(input_shape=(20,3,)),
            # tf.keras.layers.Reshape((3,128, 1)),
            # tf.keras.layers.Dense(1000, activation="relu"),
            # tf.keras.layers.Dense(581248, activation="relu"),
            # tf.keras.layers.Reshape((38,239,64)),
            # tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation="relu"),
            # tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=[2,2], strides=[5,1], activation="relu",padding="valid",data_format='channels_last'),
            # tf.keras.layers.Conv2D(filters=1, kernel_size=[1,1], strides=[1,1], activation="relu",padding="valid",data_format='channels_last'),
            tf.keras.layers.Flatten(),   
            tf.keras.layers.Dense(500, activation="relu"),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(m),

        ]
      )
    dot.summary()
    dot=[dot]


    net = MIONet_CNN(
        branch_1,
        branch_2,
        trunk,
        dot,
        {"branch1": activation[0], "branch2": activation[1], "trunk": activation[2]},
        kernel_initializer="Glorot normal",
        regularization=None,
    )

    scaler = StandardScaler().fit(y_train)
    std = np.sqrt(scaler.var_.astype(np.float32))

    def output_transform(inputs, outputs):
        return outputs * std + scaler.mean_.astype(np.float32)

    net.apply_output_transform(output_transform)
    net.apply_output_transform(dirichlet)

    model = dde.Model(data, net)
    model.compile(
        "adam", lr=3e-5,
        decay=("inverse time", 1, 1e-4),

        metrics=["l2 relative error"],
    )
    losshistory, train_state = model.train(epochs=25000, batch_size=500,display_every=2000,model_save_path="deeponet_AE_model_unaligned/")
    print("\nTraining done ...\n")
    dde.saveplot(losshistory, train_state,issave=True,isplot=True)
    
    # model.restore("./deeponet_AE_model_unaligned/-25000.ckpt")
    y_pred = model.predict(data.test_x)
    y_pred=np.transpose(y_pred)
    test_y=np.transpose(data.test_y)
    grid=x_test[2]
    N_points=s*s
    results=[]
    for k in range(200):
        spacename="./deeponet_AE_model_unaligned/test_r10_"+str(k)+".dat"
        test=np.reshape(test_y[:,k],(N_points,1))
        pre=np.reshape(y_pred[:,k],(N_points,1))
        error=np.abs(pre-test)
        result=np.hstack((grid[k],test,pre,error))
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