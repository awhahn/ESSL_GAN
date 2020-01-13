"""
Created on Mon Mar  4 15:13:56 2019

@author: m1226
"""
import tensorflow as tf
import time
from GAN_PU_MODEL import GanModel
import GAN_UTIL_PU as util
from keras.utils.generic_utils import Progbar
import numpy as np
tf.enable_eager_execution()


def main(batch_size=100,epochs=10,real_weight=1,syn_weight=1,use_class_weights=True):
    
    EPOCHS = epochs
    Real_Weight = real_weight
    Syn_Weight = syn_weight
    Loss_Weight = [Real_Weight,Syn_Weight]
    Remove_Null = True
    target_length = 103 
    
    if Remove_Null==True:
        nb_class = 10
    else: 
        nb_class = 11
   # Retrain_EPOCHS = 1
    '''Make DS'''
    train_ds,test_ds,class_weights=util.make_dataset(data='scaled',batch=batch_size,
                                       remove_null=Remove_Null,
                                       random_state=np.random.randint(1000),
                                       categories=9)             
    test_ds.repeat()
    
    if use_class_weights==True:
        class_weights=class_weights
    else: 
        class_weights=1
        
    '''Instantiate a GAN Model'''
    GAN = GanModel(batch_size=batch_size,loss_weight=Loss_Weight,
                   nb_class=nb_class,target_length=target_length,
                   class_weights=use_class_weights)
    
    '''Run logging utility, returns dictionary with test directory path
    loss log path, accuracy path, and classifier path'''
    path = util.make_logger(GAN)
    
    '''Use Tensorflows native global step creator'''
    global_step = tf.train.get_or_create_global_step()
    
    '''Begin Training Loop'''
    for epoch in range(EPOCHS):
        start = time.time()
        
        test_iter = test_ds.make_one_shot_iterator()
        next_test = test_iter.get_next()
        
        print('Epoch %d/%d'%(epoch,EPOCHS))
        if Remove_Null:
            progbar=Progbar(target=29943)
        else:
            progbar=Progbar(target=42776)
            
        for i,(images,labels) in enumerate(train_ds):
            gen_loss,disc_loss,r_loss,f_loss = GAN.train_step(images,labels,
                                                              global_step,class_weights)
            data = [epoch+1, global_step.numpy(), disc_loss.numpy(),
                    gen_loss.numpy(),r_loss.numpy(),f_loss.numpy()]
            
            util.log(data,path['log_path'])
            acc = GAN.test(next_test)
            
            
            if (global_step.numpy() % 20) == 0:
                acc = GAN.test(next_test)
                data = [epoch+1,global_step.numpy(),acc['D_true_acc'],acc['G_Precision'],
                       acc['D_true_syn'],acc['D_false_syn'],acc['D_false_real'],
                       acc['G_confusion']]
                util.log(data,path['acc_path'])
            
            progbar.update(i*batch_size)
            
        GAN.generate_and_save_images(epoch,path['path'])
        GAN.discriminator.save_weights(path['path']+'/d_weight',save_format='h5')
        GAN.generator.save_weights(path['path']+'/g_weight',save_format='h5')
        
        print ('Time taken for epoch {} is {} sec'.format(epoch + 1,time.time()-start))

    util.loss_plots(path['log_path']) 
    util.acc_plots(path['acc_path'])
    


if __name__ == '__main__':    
    tf.enable_eager_execution()
    batches = [50]#,100]
    syn_weights = [0.1]#,0.25] #,0.5,0.75,1]
    print('Entering Training')
    for bat in batches:
        for i in range(1):
            main(batch_size=bat,syn_weight=0.1)
    print('Finish Execution')    

    