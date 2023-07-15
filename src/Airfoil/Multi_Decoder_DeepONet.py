import deepxde as dde
import numpy as np
from deepxde.backend import tf

from deepxde.nn.tensorflow_compat_v1.mionet import MIONetCartesianProd,MIONet_CNN
from deepxde.data.quadruple import QuadrupleCartesianProd,Quadruple
from sklearn.preprocessing import StandardScaler

def network(problem, m,N_points):
    if problem == "ODE":
        branch_1 = [m, 200, 200]
        branch_2 = [N_points, 200, 200]
        trunk = [1, 200, 200]
    elif problem == "DR":
        branch_1 = [m, 200, 200]
        branch_2 = [m, 200, 200]
        trunk = [2, 200, 200]
    elif problem == "ADVD":
        branch_1 = [m, 200, 200]
        branch_2 = [m, 200, 200]
        trunk = [2, 300, 300, 300]
    elif problem == "flow":
        branch_1 = tf.keras.Sequential(
            [
                
                tf.keras.layers.InputLayer(input_shape=(m,)),
                tf.keras.layers.Reshape((32, 16, 1)),
                tf.keras.layers.Conv2D(32, (5, 5), strides=2, activation="relu"),
                tf.keras.layers.Conv2D(16, (5, 5), strides=2, activation="relu"),
                tf.keras.layers.Flatten(),
                # tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(200),
                # tf.keras.layers.Dense(modes),
            ]
        )
        branch_1.summary()

        branch_1=[m,branch_1]
        branch_2= tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(N_points,)),
                tf.keras.layers.Reshape((int(N_points/241), 241, 1)),
                tf.keras.layers.Conv2D(32, (5, 5), strides=2, activation="relu"),
                tf.keras.layers.Conv2D(16, (5, 5), strides=2, activation="relu"),
                tf.keras.layers.Flatten(),
                # tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(200),
                # tf.keras.layers.Dense(500),
            ]
        )
        branch_2.summary()
        branch_2=[N_points,branch_2]


        trunk =  tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(N_points,2,)),
                tf.keras.layers.Reshape((int(N_points/241), 241, 2)),
                tf.keras.layers.Conv2D(32, (5, 5), strides=2, activation="relu"),
                tf.keras.layers.Conv2D(16, (5, 5), strides=2, activation="relu"),
                tf.keras.layers.Flatten(),
                # tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(200),
                # tf.keras.layers.Dense(500),
            ]
        )
        trunk.summary()
        trunk=[2,trunk]





        dot= tf.keras.Sequential(
        [   tf.keras.layers.InputLayer(input_shape=(200,3,)),
            tf.keras.layers.Reshape((3,200, 1)),
            # tf.keras.layers.Dense(1000, activation="relu"),
            # tf.keras.layers.Dense(581248, activation="relu"),
            # tf.keras.layers.Reshape((38,239,64)),
            # tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation="relu"),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=[2,2], strides=[5,5], activation="relu",padding="valid",data_format='channels_last'),
            tf.keras.layers.Conv2D(filters=1, kernel_size=[1,1], strides=[1,1], activation="relu",padding="valid",data_format='channels_last'),
            tf.keras.layers.Flatten(),   
            # tf.keras.layers.Dense(1000, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(N_points),

        ]
      )
    dot.summary()
    dot=[dot]
    return branch_1,branch_2, trunk,dot

def get_data(N_points):
    
    if N_points<512 :
        data=np.zeros((200,512,7))
    else:
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
    tk=np.load('t_k_20220905.npy')
    tk=tk[:,0:N_points,:]
    
    order = np.reshape(np.arange (200),(200,))
    data[:,0,6]=order
    data[:,0:512,0]=xy_branch
    data[:,:N_points,1]  =label
    data[:,:N_points,[2,3]]  =tk
    data[:,:N_points,[4,5]]  =xy
    
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
    label_train=data_train[:,:N_points,1]
    tk_train=data_train[:,:,[2,3]]
    xy_branch_test=data_test[:,0:512,0]
    label_test=data_test[:,:N_points,1]
    tk_test=data_test[:,:,[2,3]]
    grid = np.reshape(np.linspace(-5, 5, N_points),(N_points,1))
    tk_same=np.zeros((200,N_points,2))
    for i in range(len(tk_same)):
        if i<180:
          tk_same[i]=tk_train[0]
        else:
          tk_same[i]=tk_test[0]

    xy_train=data_train[:,:,[4,5]]
    xy_test=data_test[:,:,[4,5]]
    
    
    # label_train=np.reshape(label_train,(180*24100,1))
    # label_test=np.reshape(label_test,(20*24100,1))
    label_average=np.zeros((N_points))
    label_averages=np.zeros((180,N_points))
    for i in range(180):
        label_average=label_train[i]+label_average
    label_average=label_average/180
    for i in range(180):
        label_averages[i]=label_average

    X_train = (xy_branch_train,label_averages,xy_train)
    X_test = (xy_branch_test, label_averages[0:20],xy_test)
    return X_train, label_train, X_test, label_test,data_test,xy_branch_test, label_test

def run(problem, lr, epochs, m, N_points,activation, initializer):
    # training_data = np.load("../data/" + problem + "_train.npz", allow_pickle=True)
    # testing_data = np.load("../data/" + problem + "_test.npz", allow_pickle=True)
    # X_train = training_data["X_train"]
    # y_train = training_data["y_train"]
    # X_test = testing_data["X_test"]
    # y_test = testing_data["y_test"]

    X_train, y_train, X_test, y_test,datatest,xy_branch_test, label_test=get_data(N_points)

    branch_net_1,branch_net_2, trunk_net,dot = network(problem,m,N_points)

    data = Quadruple(X_train, y_train, X_test, y_test)
    net = MIONet_CNN(
        branch_net_1,
        branch_net_2,
        trunk_net,
        dot,
        {"branch1": activation[0], "branch2": activation[1], "trunk": activation[2]},
        kernel_initializer=initializer,
        regularization=None,
    )

    scaler = StandardScaler().fit(y_train)
    std = np.sqrt(scaler.var_.astype(np.float32))

    def output_transform(inputs, outputs):
        return outputs * std + scaler.mean_.astype(np.float32)

    net.apply_output_transform(output_transform)


    model = dde.Model(data, net)
    model.compile("adam", lr=lr,
        metrics=["mean l2 relative error"], 
        decay=("inverse time", 1, 1e-4))
    checker = dde.callbacks.ModelCheckpoint(
        "CNN_deeponet_average/CNN_deeponet_average_model.ckpt", save_better_only=True, period=10000
    )
    
    losshistory, train_state = model.train(epochs=epochs ,batch_size=None,display_every=100, callbacks=[checker],model_save_path="CNN_deeponet_average/")
    dde.saveplot(losshistory, train_state,issave=True,isplot=True)

    # model.restore("./CNN_deeponet_average/-50000.ckpt")
    # grid_0=np.zeros((20,N_points,2))
    # X_test0=(xy_branch_test, label_test, grid_0)


    label_pre = model.predict(X_test)
    label_pre=np.transpose(label_pre)
    # grid = np.reshape(X_test[1],(N_points,2))
    
    label_test=np.reshape(y_test,(20,N_points))
    label_test=np.transpose(label_test)
    xy=np.reshape(datatest[:,:N_points,[4,5]],(20,N_points,2))
    grid=np.reshape(datatest[:,:N_points,[2,3]],(20,N_points,2))
    results=[]
    
    for k in range(len(X_test[0])):
        
        test=np.reshape(label_test[:,k],(N_points,1))
        pre=np.reshape(label_pre[:,k],(N_points,1))
        error=np.abs(pre-test)
        result=np.hstack((grid[k],xy[k],test,pre,error))
        xy_test=X_test[0]
        shape=np.reshape(xy_test[k,:],(256,2))
        np.savetxt("./CNN_deeponet_average/shape"+str(k)+".dat",shape)
        np.savetxt("./CNN_deeponet_average/test_"+str(k)+".dat",result)
        results=np.append(results,result)
    results=np.reshape(results,(20,N_points,7))
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


def main():
    # Problems:
    # - "ODE": Antiderivative, Nonlinear ODE
    # - "DR": Diffusion-reaction
    # - "ADVD": Advection-diffusion
    problem = "flow"
    T = 1
    N_points=241*100
    m = 512
    lr = 0.003
    epochs =50000
    activation = (
        ["relu", None, "relu"] if problem in ["ADVD"] else ["relu", "relu", "relu"]
    )
    initializer = "Glorot normal"

    run(problem, lr, epochs, m,N_points, activation, initializer)

    


if __name__ == "__main__":
    main()
  