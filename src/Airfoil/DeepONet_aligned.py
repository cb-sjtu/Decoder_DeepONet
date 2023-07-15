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

#周期性边界条件
# def periodic(x):
#     x *= 2 * np.pi
#     return tf.concat(
#         [tf.math.cos(x), tf.math.sin(x), tf.math.cos(2 * x), tf.math.sin(2 * x)], 1
#     )

#设置数据集


def get_data():
    
    grid = np.reshape(np.linspace(0, 1, 241),(241,1))
    
    data=np.zeros((200,512,2))
    xy_branch=np.load('xys-256.npy')
    xy_branch=np.reshape(xy_branch,(200,512))
    label=np.load('labels_0005_5535.npy')
    label=np.reshape(label,(200,241))
    data[:,:,0]=xy_branch
    data[:,0:241,1]  =label
    np.random.seed(1234) 
    np.random.shuffle(data)
    # data=tuple(data)
    # data=set(tuple(data))
    data_train=data[0:180,:,:]
    data_test=data[180:200,:,:]

    xy_branch_train=data_train[:,:,0]
    label_train=data_train[:,0:241,1]
    xy_branch_test=data_test[:,:,0]
    label_test=data_test[:,0:241,1]
    x_train = (xy_branch_train, grid)
    x_test = (xy_branch_test, grid)

    
    return x_train, label_train, x_test, label_test, 

def get_data_pre(N_Bs,N_grid):
    
    grid = np.reshape(np.linspace(0, 1, N_grid),(N_grid,1))

    
    xy_pre=np.load('predict_branch.npy')
    xy_pre=np.reshape(xy_pre,(N_Bs,512))
    
   
    x_predict = (xy_pre, grid)
   

    return  x_predict

#设置训练优化方法
def train(model, lr, epochs):
    decay = ("inverse time", epochs // 5, 0.5)
    model.compile("adam", lr=lr,  loss='MSE',metrics=["l2 relative error"], decay=decay)
    losshistory, train_state = model.train(epochs=epochs, batch_size=None,display_every=500,model_save_path='./result-2')
    # dde.postprocessing.save_loss_history(losshistory, "loss.dat")
    print("\nTraining done ...\n")
    dde.saveplot(losshistory, train_state,issave=True,isplot=True)
#搭建网络结构
def main():
    x_train, y_train, x_test, y_test = get_data()
    
    # for k in range(len(x_test[0])):
        
       
    #     xy_test=x_test[0]
    #     shape=np.reshape(xy_test[k,:],(256,2))
    #     np.savetxt("./test_shape"+str(k)+".dat",shape)
        

    m = 512
    net = dde.maps.DeepONetCartesianProd(
        [m, 256, 256, 256, 256], [1, 256, 256], "tanh", "Glorot normal"
    )
    # net.apply_feature_transform(periodic)

    scaler = StandardScaler().fit(y_train)
    std = np.sqrt(scaler.var_.astype(np.float32))

    def output_transform(inputs, outputs):
        return outputs * std + scaler.mean_.astype(np.float32)

    net.apply_output_transform(output_transform)

    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
    model = dde.Model(data, net) 

    lr = 0.00001
    epochs = 20000
    
    
    train(model, lr, epochs)
    # x_predict = get_data_pre(1,241)
    label_pre = model.predict(x_test)
    label_pre=np.transpose(label_pre)
    grid = np.reshape(np.linspace(0, 1, 241),(241,1))
    
    label_test=np.reshape(y_test,(20,241))
    label_test=np.transpose(label_test)
    
    results=[]
    
    for k in range(len(x_test[0])):
        
        test=np.reshape(label_test[:,k],(241,1))
        pre=np.reshape(label_pre[:,k],(241,1))
        result=np.hstack((grid,test,pre))
        xy_test=x_test[0]
        shape=np.reshape(xy_test[k,:],(256,2))
        np.savetxt("./result-2/shape"+str(k)+".dat",shape)
        np.savetxt("./result-2/"+str(epochs)+"test"+str(k)+".dat",result)
        results=np.append(results,result)
    results=np.reshape(results,(20,241,3))
    np.save(str(epochs)+'_results_0005_5535.npy',results)
   

if __name__ == "__main__":
    main()
    