import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from sklearn.preprocessing import StandardScaler
import deepxde as dde
from deepxde.backend import tf
import tensorflow_addons as tfa

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def get_data():
    N_points=241*100
    data=np.zeros((200,N_points,7))
    xy_branch=np.load('xys-256.npy')
    # xy_branch=xy_branch[:,:,0]
    xy_branch=np.reshape(xy_branch,(200,512))
    label=np.load('cps.npy')
    label=np.reshape(label,(200,29161))
    label=label[:,0:N_points]
    xy=np.load("1_2_3_4_12s.npy")
    xy=xy[:,0:N_points,2:4]
    # print(label[0,240])
    X=np.load('t_k_20220905.npy')
    X=X[:,0:N_points,:]
    # X=np.reshape(X,(200,241))
    # num_grid=121
    order = np.reshape(np.arange (200),(200,))
    # k=np.tile(k[:, :], [200,1])
    # k=np.reshape(k,())
    # xxs=[]
    # for i in range(len(X)):
    #     for j in range(len(X[0])):
    #         for kk in range(len(k)):
    #           xx=[X[i,j],k[kk]]
    #           xxs=np.append(xxs,xx)
    
    data[:,0,6]=order
    data[:,0:512,0]=xy_branch
    data[:,:,1]  =label
    data[:,:,[2,3]]  =X
    data[:,:,[4,5]]  =xy
    
    np.random.seed(1234) 
    # per = np.random.permutation(data.shape[0])
    np.random.shuffle(data)
    
    order_random=data[:,0,6]
    np.save('order_random.npy',order_random)
    # data=tuple(data)
    # data=set(tuple(data))
    data_train=data[0:180,:,:].astype(np.float32)
    data_test=data[180:200,:,:].astype(np.float32)

    xy_branch_train=data_train[:,0:512,0]
    label_train=data_train[:,:,1]
    Xx_train=data_train[:,:,[2,3]]
    xy_branch_test=data_test[:,0:512,0]
    label_test=data_test[:,:,1]
    Xx_test=data_test[:,:,[2,3]]
    grid = np.reshape(np.linspace(0, 1, 241*100),(241*100,1))
    X_train = (xy_branch_train, grid)
    X_test = (xy_branch_test, grid)

    
    return X_train, label_train, X_test, label_test, data_test


def pod(y):
    n = len(y)
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    # C = 1 / (n - 1) * y.T @ y
    # w, v = np.linalg.eigh(C)
    # np.save('w_3.npy',w)
    # np.save('v_3.npy',v)
    w=np.load('w_100.npy')
    v=np.load('v_100.npy')
    w = np.flip(w)
    v = np.fliplr(v)
    v *= len(y_mean) ** 0.5
    w_cumsum = np.cumsum(w)
    print(w_cumsum[:16] / w_cumsum[-1])
    w_cumsum_1=w_cumsum/ w_cumsum[-1]
    # plt.figure()
    # plt.plot(w_cumsum_1[:20])
    # plt.savefig('w')
    # plt.figure()
    # for i in range(8):
    #     plt.subplot(2, 4, i + 1)
    #     plt.plot(v[:, i])
    # plt.show()
    # plt.savefig('1')
    return y_mean, v


  
def main():
    N_points=24100
    x_train, y_train, x_test, y_test ,datatest= get_data()
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    y_mean, v = pod(y_train)
    # np.load('w_100.npy')
    modes =64
    m = 512
    activation="relu"
    branch = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m,)),
            tf.keras.layers.Reshape((32, 16, 1)),
            tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Conv2D(128, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=activation),
            tf.keras.layers.Dense(64),
        ]
    )
    
    branch.summary()
    net = dde.maps.PODDeepONet(v[:, :modes], [m, branch],'relu',"Glorot normal")

    def output_transform(inputs, outputs):
        # return outputs + y_mean
        # return outputs / modes ** 0.5 + y_mean
        return outputs / modes + y_mean

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=3e-3,
        metrics=["mean l2 relative error"],    
        decay=("inverse time", 1, 1e-4)
    )
    
    # losshistory, train_state = model.train(epochs=25000, batch_size=None, display_every=100,model_save_path="./result-pod-100-m64/",)
    # dde.saveplot(losshistory, train_state,issave=True,isplot=True)
    model.restore("./result-pod-100-m64/-25000.ckpt")

#输出预测数据
    # try_grid= np.reshape(np.linspace(0, 1, 241*50),(241*50,1))
    # try_branch=x_test[0]
    # tryy=(try_branch,try_grid)
    label_pre = model.predict(x_test)
    label_pre=np.transpose(label_pre)
    # grid = np.reshape(x_test[1],(N_points,2))
    
    label_test=np.reshape(y_test,(20,N_points))
    label_test=np.transpose(label_test)
    xy=np.reshape(datatest[:,:,[4,5]],(20,N_points,2))
    grid=np.reshape(datatest[:,:,[2,3]],(20,N_points,2))
    results=[]
    
    for k in range(len(x_test[0])):
        
        test=np.reshape(label_test[:,k],(N_points,1))
        pre=np.reshape(label_pre[:,k],(N_points,1))
        error=np.abs(pre-test)
        result=np.hstack((grid[k],xy[k],test,pre,error))
        xy_test=x_test[0]
        shape=np.reshape(xy_test[k,:],(256,2))
        np.savetxt("./result-pod-100-m64/shape"+str(k)+".dat",shape)
        np.savetxt("./result-pod-100-m64/test"+str(k)+".dat",result)
        results=np.append(results,result)
    results=np.reshape(results,(20,N_points,7))
    np.save('_results_0005_5535.npy',results)

if __name__ == "__main__":
    main()
