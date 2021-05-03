import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers 
from tensorflow.keras import regularizers
from keras.callbacks import CSVLogger
from time import time
import pickle

 
import golois
 
planes = 21
moves = 361
N = 10000
epochs = 500
batch = 128
filters = 95
 
 
input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype ('float32')
 
policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical (policy)
 
value = np.random.randint(2, size=(N,))
value = value.astype ('float32')
 
end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype ('float32')
 
groups = np.zeros((N, 19, 19, 1))
groups = groups.astype ('float32')
 
print ("getValidation", flush = True)
golois.getValidation (input_data, policy, value, end)

 
#---------------------- Model -------------------
 
input = keras.Input(shape=(19, 19, planes), name='board')
# print("input shape", input.shape)

x = layers.Conv2D(filters, 1, activation='relu', padding='same')(input)
# print("x shape ",x.shape)

for i in range (6):
    print("----------------block N° ",i,"----------------\n")
    x1 = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    # print("after 1st conv x1 shape", x1.shape)
    x1 = layers.Conv2D(filters, 3, padding='same')(x1)
    # print("after 12nd conv x1 shape", x1.shape)
    x = layers.add([x1,x])
    # print("after add x shape", x.shape)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    
 #------------------- Uncomment if willing to try a verion with Bottleneck blocks ----------------------------
 
 #(less performant than simple ResNet blocks with fine tuning)  
# filters = 280
 # for i in range (10):
    # print("--------------- Bottleneck Block N° ",i,"--------------\n")
    # x1 = layers.Conv2D(filters/4,1, activation='relu', padding='same')(x)
    # x1 = layers.BatchNormalization()(x1)
    # x1 = layers.Conv2D(filters/4,3,activation='relu', padding='same')(x1)
    # x1 = layers.BatchNormalization()(x1)
    # x1 = layers.Conv2D(filters,1, padding='same')(x1)
    # x = layers.add([x1,x])
    # x = layers.ReLU()(x)
    # x = layers.BatchNormalization()(x)
    
#------------------------------------------------------------------------------------------------------

policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias = False, kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)
value_head = layers.GlobalAveragePooling2D()(x)
value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)
 
model = keras.Model(inputs=input, outputs=[policy_head, value_head])
model.summary()


#------------------ Training ----------------

model.compile(optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.9),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy' : 1.0, 'value' : 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

train_logger = CSVLogger('training.log', separator=',', append=False)
val_logger = open('val_logger.txt', 'wb')

min_improv_pol= 1e-3
min_improv_val= 1e-3
patience=15

lst_pol_acc= 0
lst_val_mse= 0
lst_improv_pol=1
lst_improv_val=1
lst_decay=50
learning_decay=0.1

start=time()

for i in range (1, epochs + 1):
    print ('epoch ' + str (i))
    golois.getBatch (input_data, policy, value, end, groups)
    history = model.fit(input_data,
                        {'policy': policy, 'value': value},   
                        epochs=1, batch_size=batch, callbacks=[train_logger])

    if (i % 10 == 0):
        golois.getValidation (input_data, policy, value, end)
        val = model.evaluate (input_data,
                              [policy, value], verbose = 0, batch_size=batch)
        print ("val =", val)
        
        pickle.dump(val, val_logger)
    
        if (val[3]>lst_pol_acc or val[4]> lst_val_mse) and (i% 200 == 0):    #done when trained on Collab, where training over all epochs could be unfinished and thus final save unreached
            model.save ('models/intermediate_model'+str(i)+'.h5')        
            print("saving intermediate model")

        #check for early stopping
        if val[3]- lst_pol_acc < min_improv_pol and i-lst_improv_pol>patience:
            print("Stopping training after {} epochs; policy accuracy stagnating at {}".format(i,val[3]) )
            break
        if val[4]-lst_val_mse < min_improv_val and i-lst_improv_val>patience:
            print("Stopping training after {} epochs; value MSE stagnating at {}".format(i,val[4]))
            break


        lst_improv_pol=i
        lst_improv_val=i
        
        lst_pol_acc= val[3]
        lst_val_mse= val[4]
    

    
    if (i-lst_decay == 50 and i >= 100): 

        old_lr = keras.backend.get_value(model.optimizer.learning_rate)                 
        new_lr = max(old_lr * learning_decay, 0.00001)                    
        keras.backend.set_value(model.optimizer.learning_rate, new_lr)
        print("learning rate decaying from {} to {} at epoch {} ".format(old_lr,new_lr,i))
        lst_decay=i  

print("Total time", time()-start)
model.save ('/models/final_model.h5')
val_logger.close()