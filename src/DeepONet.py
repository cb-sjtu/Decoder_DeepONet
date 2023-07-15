from cProfile import label
from matplotlib.pyplot import grid
import tensorflow
from random import random, sample
import deepxde as dde
import numpy as np
from deepxde.backend import tf
from scipy import io
from sklearn.preprocessing import StandardScaler
import random
import tensorflow_addons as tfa
import keras
from keras.backend import set_session

#周期性边界条件
# def periodic(x):
#     x *= 2 * np.pi
#     return tf.concat(
#         [tf.math.cos(x), tf.math.sin(x), tf.math.cos(2 * x), tf.math.sin(2 * x)], 1
#     )

#设置数据集
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)
set_session(sess)
keras.backend.clear_session()


def get_data():
    
    N_points=24100
    N_test=10
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
    data[:,0:N_points,1]  =label
    data[:,0:N_points,[2,3]]  =X
    data[:,0:N_points,[4,5]]  =xy
    
    np.random.seed(1234) 
    # per = np.random.permutation(data.shape[0])
    np.random.shuffle(data)
    
    order_random=data[:,0,6]
    np.save('order_random.npy',order_random)
    # data=tuple(data)
    # data=set(tuple(data))
    data_train=data[0:190,:,:].astype(np.float32)
    data_test=data[190:200,:,:].astype(np.float32)

    xy_branch_train=data_train[:,0:512,0]
    label_train=data_train[:,0:N_points,1]
    Xx_train=data_train[:,0:N_points,[2,3]]
    Xx_train=np.reshape(Xx_train,(190*N_points,2))
    xy_branch_test=data_test[:,0:512,0]
    label_test=data_test[:,0:N_points,1]
    Xx_test=data_test[:,0:N_points,[2,3]]
    Xx_test=np.reshape(Xx_test,(10*N_points,2))
    grid = np.reshape(np.linspace(0, 1, 241*1),(241*1,1))

    # xy_branch_train_dg=np.zeros((190*N_points,512))
    # for i in range(180):
    #     for j in range(N_points):
    #         k=i*N_points+j
    #         xy_branch_train_dg[k]=xy_branch_train[i]
    xy_branch_test_dg=np.zeros((N_test*N_points,512))
    for i in range(1):
        for j in range(N_points):
            k=i*N_points+j 
            xy_branch_test_dg[k]=xy_branch_test[i]

    # print(xy_branch_train_dg[24099:24102])
    # np.save('xy_branch_train_dg.npy',xy_branch_train_dg)
    # np.save('xy_branch_test_dg.npy',xy_branch_test_dg)
    xy_branch_train_dg=np.load('xy_branch_train_dg.npy')
    # xy_branch_test_dg=np.load('xy_branch_test_dg.npy')
    # np.random.shuffle(xy_branch_train_dg)
    # np.random.shuffle(Xx_train)
    # np.random.shuffle(xy_branch_test_dg)
    # np.random.shuffle(Xx_test)
    X_train = (xy_branch_train_dg, Xx_train)
    X_test = (xy_branch_test_dg, Xx_test[0:N_test*N_points])
    label_train=np.reshape(label_train,(190*N_points,1))
   
    label_test=np.reshape(label_test,(10*N_points,1))

    
    return X_train, label_train, X_test, label_test[0:N_test*N_points], data_test
    
    

    
   

def get_data_pre(N_Bs,N_grid):
    
    grid = np.reshape(np.linspace(0, 1, N_grid),(N_grid,1))

    
    xy_pre=np.load('predict_branch.npy')
    xy_pre=np.reshape(xy_pre,(N_Bs,512))
    
   
    x_predict = (xy_pre, grid)
   

    return  x_predict

#设置训练优化方法
def train(model, lr, epochs):
    decay = ("inverse time", 1, 0.0001)
    model.compile("adam", lr=lr,  loss='MSE',metrics=["l2 relative error"],decay=decay)
    losshistory, train_state = model.train(epochs=epochs, batch_size=10000,display_every=500,model_save_path='./deeponet_dg_24100_model_2/')
    # dde.postprocessing.save_loss_history(losshistory, "./deeponet_dg_24100_model_2/loss.dat")
    print("\nTraining done ...\n")
    dde.saveplot(losshistory, train_state,issave=True,isplot=True)
    # model.restore("./deeponet_dg_24100_model_2/-150000.ckpt")
#搭建网络结构
def main():
    x_train, y_train, x_test, y_test,datatest = get_data()
    
    # for k in range(len(x_test[0])):
        
       
    #     xy_test=x_test[0]
    #     shape=np.reshape(xy_test[k,:],(256,2))
    #     np.savetxt("./test_shape"+str(k)+".dat",shape)
    m = 512  
    branch = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m,)),
            tf.keras.layers.Reshape((32, 16, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation="relu"),
            tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(200),
        ]
    )
    
    branch.summary()
    
    net = dde.maps.DeepONet(
        [m, branch], [2,512,1024,512,200], "tanh", "Glorot normal"
    )
    # net.apply_feature_transform(periodic)

    scaler = StandardScaler().fit(y_train)
    std = np.sqrt(scaler.var_.astype(np.float32))

    def output_transform(inputs, outputs):
        return outputs * std + scaler.mean_.astype(np.float32)

    net.apply_output_transform(output_transform)

    data = dde.data.Triple(x_train, y_train, x_test, y_test)
    model = dde.Model(data, net) 

    lr = 0.0003
    epochs = 25000
    
    
    train(model, lr, epochs)
    # x_predict = get_data_pre(1,241)
    label_pre = model.predict(x_test)
    label_pre=np.reshape(label_pre,(10,24100))
    label_pre=np.transpose(label_pre)
    # grid = np.reshape(x_test[1],(29161,2))
    
    label_test=np.reshape(y_test,(10,24100))
    label_test=np.transpose(label_test)
    xy=np.reshape(datatest[:,:,[4,5]],(10,24100,2))
    grid=np.reshape(datatest[:,:,[2,3]],(10,24100,2))
    results=[]
    
    for k in range(10):
        
        test=np.reshape(label_test[:,k],(24100,1))
        pre=np.reshape(label_pre[:,k],(24100,1))
        error=np.abs(pre-test)
        result=np.hstack((grid[k],xy[k],test,pre,error))
        xy_test=x_test[0]
        shape=np.reshape(xy_test[k,:],(256,2))
        np.savetxt("./deeponet_dg_24100_model_2/shape"+str(k)+".dat",shape)
        np.savetxt("./deeponet_dg_24100_model_2/"+"test"+str(k)+".dat",result)
        results=np.append(results,result)
    results=np.reshape(results,(10,24100,7))
    np.save('_results_0005_5535.npy',results)
    print(
        "# Parameters:",
        np.sum(
            [
                np.prod(v.get_shape().as_list())
                for v in tf.compat.v1.trainable_variables()
            ]
        ),
    )
   

if __name__ == "__main__":
    main()
    