import os
import pickle
import time
from functools import partial
import numpy as np
import datetime


import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.metrics import Mean

from gan_thesis.models.wgan.utils import *
from gan_thesis.models.wgan.data import *



class WGAN:

    def __init__(self):
        """Main WGAN Model

        Args: Dictionary with
            output_dim:
                Integer dimension of the output variables including
                the one-hot encoding of the categorical variables
            embedding_dim:
                Integer dimension of random noise sampled for the generator
            gen_dim:
                Tuple with the hidden layer dimension for the generator
            crit_dim:
                Tuple with hidden layer dimension for the critic
            mode:
                'wgan' or 'wgan-gp', deciding which loss function to use
            gp_const:
                Gradient penalty constant. Only needed if mode == 'wgan-gp'
            n_critic:
                Number of critic learning iterations per generator iteration
            log_directory:
                Directory of tensorboard logs
            


        Checkpoints: yet to be added...
        """
        
        self.epoch_trained = 0
        self.initialized = False
        
    def initialize(self, df, cont_cols, cat_cols, input_params):
        
        params = {
            'embedding_dim' : 128, #dimenstion of random input samples
            'mode' : 'wgan-gp', #'wgan' or 'wgan-gp'
            'n_critic' : 5, #number of iterations of critic between generator iterations
            'gp_const' : 10, #weight on gradient penalty
            'gen_dim' : (256,256), # tuple of hidden dimensions of generator
            'crit_dim' : (256, 256), #tupe of hidden dimensions of critic
            'beta1' : 0.5, # Adam
            'beta2': 0.9, # Adam
            'lr' : 10**-4, # Adam
            'hard' : False, # Straight through gumbel-softmax 
            'temperature' : 0.2, # Gumbel softmax temperature
            'temp_anneal' : False, # temperature annealing
            'input_time' : False, #Tensorboard logging purposes
            'log_directory' : False, #tensorboard logs
            'n_pac' : 1 
        }
        
        
        for key in input_params:
            params[key] = input_params[key]
        
        
        self.latent_dim = params['embedding_dim']
        self.mode = params['mode']
        self.lr = params['lr']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.n_critic = params['n_critic']
        self.temperature = params['temperature']
        self.temp_anneal = params['temp_anneal']
        self.hard = params['hard']
        if self.mode == 'wgan-gp':
            self.gp_const = params['gp_const']
        self.input_time = params['input_time']
        self.log_dir = params['log_directory']
        self.n_pac = params['n_pac']
        
        
        self.cat_dims = tuple([len(df[cat]) for cat in cat_cols])
        self.orignal_order_cols = list(df.columns)
        

        gen_dim = params['gen_dim']
        crit_dim = params['crit_dim']
        self.generator = self.make_generator(gen_dim)
        self.critic = self.make_critic(self.n_pac*crit_dim)

        self.gen_opt, self.crit_opt = self.get_opts()
        
        

    def make_generator(self, gen_dim):
        inputs = keras.Input(shape=(self.latent_dim,))

        if type(gen_dim) == int:
            temp_layer = layers.Dense(gen_dim,
                                      kernel_initializer='normal')(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.ReLU()(temp_layer)

        else:
            temp_layer = layers.Dense(gen_dim[0],
                                      kernel_initializer='normal')(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.ReLU()(temp_layer)

            for shape in gen_dim[1:]:
                temp_layer = layers.Dense(shape,
                                          kernel_initializer='normal')(temp_layer)
                temp_layer = layers.BatchNormalization()(temp_layer)
                temp_layer = layers.ReLU()(temp_layer)

        outputs = layers.Dense(self.n_cont+self.n_cat_oht)(temp_layer)
        # cont_output = layers.Dense(self.n_cont, activation='tanh')(temp_layer)
        # cat_output = layers.Dense(self.n_cat_oht)(temp_layer)
        # outputs = layers.Concatenate(axis=1)([cont_output, cat_output])
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        return model

    def make_critic(self, crit_dim):

        inputs = keras.Input(shape=((self.n_cont+self.n_cat_oht), ))
        if self.mode == 'wgan':
            constraint = ClipConstraint(0.01)
        else:
            constraint = None

        if type(crit_dim) == int:
            temp_layer = layers.Dense(crit_dim,
                                      kernel_constraint=constraint)(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.LeakyReLU()(temp_layer)

        else:
            temp_layer = layers.Dense(crit_dim[0],
                                      kernel_constraint=constraint)(inputs)
            temp_layer = layers.BatchNormalization()(temp_layer)
            temp_layer = layers.LeakyReLU()(temp_layer)

            for shape in crit_dim[1:]:
                temp_layer = layers.Dense(shape,
                                          kernel_constraint=constraint)(temp_layer)
                temp_layer = layers.BatchNormalization()(temp_layer)
                temp_layer = layers.LeakyReLU()(temp_layer)

        outputs = layers.Dense(1)(temp_layer)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name = 'Critic')
        return model

    def get_opts(self):
        if self.mode == 'wgan':
            gen_opt = keras.optimizers.RMSprop(self.lr)
            crit_opt = keras.optimizers.RMSprop(self.lr)
        elif self.mode == 'wgan-gp':
            gen_opt = keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2)
            crit_opt = keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2)
        return gen_opt, crit_opt

    def sample_df(self, n, temperature=0.2, hard=True, scaled=False):
        array_sample = self.sample(n, temperature, hard).numpy()
        df_sample = pd.DataFrame(array_sample, columns=self.oht_shuff_cols)

        # if not scaled:
        #     df_sample = self.scaler.inverse_transfrom(df_sample)
        #     df_sample = df_sample[self.orignal_order_cols]

        return df_sample

    def sample(self, n, temperature=0.2, hard=True):
        noise = tf.random.normal((n, self.latent_dim))
        sample = self.generator(noise, training=False)
        sample = self.apply_activate(sample)

        return sample

    def scale_data(self, df, cont_cols, cat_cols, fit):
        df = df.copy()
        
        if self.initialized == False:
            
            self.orignal_order_cols = list(df.columns)
            self.scaler = dataScaler()
        
        df = data_reorder(df, cat_cols)
        df = self.scaler.transform(df, cont_cols, cat_cols, fit)
        
        if self.initialized == False:
            self.n_cont, self.n_cat_oht = len(cont_cols) , (len(df.columns)-len(cont_cols))
        df = df.astype('float32')
        self.oht_shuff_cols = list(df.columns)
        
        return df
    
    def apply_activate(self, data_batch):
        #numerical activation
        ret_data = data_batch
        # ret_data = ret_data[:,0:self.n_cont]
        # ret_data = tf.nn.tanh(ret_data)
        # ret_data = tf.concat([ret_data, data_batch[:, self.n_cont:]], axis = 1)
        
        #categorical activation
        if self.cat_dims != ():
                ret_data = sample_gumbel(ret_data, self.temperature, self.cat_dims, self.hard)
        return ret_data
    
    def train(self, dataframe, epochs, batch_size = 500, params = {}, cont_cols = [], cat_cols = [], shuffle = True,  new_data = False):
        
        df = dataframe.copy()
        self.batch_size = batch_size
        
        if self.initialized == False:
            
            df = self.scale_data(df, cont_cols, cat_cols, True)
            
            self.n_cont, self.n_cat_oht = len(cont_cols) , (len(df.columns)-len(cont_cols))
            self.initialize(df, cont_cols, cat_cols, params)
            self.initialized = True
        else:
            df = self.scale_data(df, cont_cols, cat_cols, new_data)


        dataset = df_to_dataset(df, shuffle, self.batch_size)
        loss_li = self.train_ds(dataset, epochs, len(df), self.batch_size, self.cat_dims, self.hard, self.temp_anneal, self.input_time)
        self.epoch_trained += epochs
        return loss_li
        

    def train_ds(self, dataset, epochs, n_data, batch_size=500, cat_dims=(), hard=False, temp_anneal = False, input_time = False):

        
        self.cat_dims = cat_dims
        temp_increment = self.temperature/epochs # for temperature annealing
        
        self.g_loss = Mean('generator_loss', dtype = tf.float64)
        self.c_loss = Mean('critic_loss', dtype = tf.float64)

        
        if self.log_dir:
            current_time = input_time if input_time else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            generator_log_dir = self.log_dir+'\\logs\\'+current_time+'\\gradient_tape\\generator'
            critic_log_dir = self.log_dir+ '\\logs\\'+current_time+'\\gradient_tape\\critic'
            generator_summary_writer = tf.summary.create_file_writer(generator_log_dir)
            critic_summary_writer = tf.summary.create_file_writer(critic_log_dir)
        
        
        for epoch in range(self.epoch_trained,self.epoch_trained+epochs):

            start = time.time()
            g_loss = 0
            c_loss = 0
            counter = 0
            loss_li = [[], []]
            #trace = True # Tensorboard tracing, currently not working
            for data_batch in dataset:

                # if trace: 
                #     tf.summary.trace_on(graph = True, profiler = True)
                

                c_loss = self.train_step_c(data_batch, hard)
                # if trace:
                #     with critic_summary_writer.as_default():
                #         tf.summary.trace_export(
                #             name = 'critic_trace', step = 0, profiler_outdir = critic_log_dir
                #         )             
                
                if counter % self.n_critic == 0:
                    # if trace:
                    #     tf.summary.trace_on(graph = True, profiler = True)
                
                    g_loss = self.train_step_g(batch_size, hard)
                    # if trace:
                    #     with generator_summary_writer.as_default():
                    #         tf.summary.trace_export(
                    #             'generator_trace', step = 0, profiler_outdir = generator_log_dir
                    #             )
                    #     start = False    
                        
                counter += 1
            
            if self.log_dir:
                with critic_summary_writer.as_default():
                        tf.summary.scalar('loss', c_loss, step = epoch)
                        
                with generator_summary_writer.as_default():
                            tf.summary.scalar('loss', g_loss, step = epoch)
            loss_li[0].append(c_loss.numpy())
            loss_li[1].append(g_loss.numpy())
            
            if (epoch + 1) % 5 == 0:
                # Checkpooint functionality here 
                
                print('Epoch: {}, Time Elapsed:{} sec \n Critic Loss: {} Generator Loss: {}'.format(epoch + 1,
                                                                                                          np.round(time.time() - start, 4), 
                                                                                                            my_tf_round(c_loss, 4), my_tf_round(g_loss,4)))
            #if (temp_anneal):
            #    self.set_temperature(self.temperature-temp_increment)
            dataset = dataset.shuffle(buffer_size=10000)
        return loss_li
            
    @tf.function
    def train_step_c(self, data_batch, hard):
        tot_dim = data_batch.shape[1]
        start_cat_dim = tot_dim - sum(self.cat_dims)
        noise = tf.random.normal((len(data_batch), self.latent_dim))

        with tf.GradientTape() as crit_tape:

            fake_data = self.generator(noise, training=True)
            fake_data = self.apply_activate(fake_data)
            
            real_output = self.critic(data_batch, training=True)
            fake_output = self.critic(fake_data, training=True)

            crit_loss = critic_loss(real_output, fake_output)
            if self.mode == 'wgan-gp':
                gp_loss = self.gp_const * gradient_penalty(partial(self.critic), data_batch, fake_data)
                crit_loss += gp_loss
                
            critic_gradients = crit_tape.gradient(crit_loss , self.critic.trainable_variables)
            self.crit_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
            
            self.c_loss(crit_loss)
        return crit_loss

    @tf.function
    def train_step_g(self, batch_size, hard):
        noise = tf.random.normal((batch_size, self.latent_dim))
        with tf.GradientTape() as gen_tape:
            fake_data = self.generator(noise, training=True)
            gen_tape.watch(fake_data)
            
            fake_data = self.apply_activate(fake_data)

            
            fake_output = self.critic(fake_data, training=True)
            gen_loss = generator_loss(fake_output)
            generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.gen_opt.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
            self.g_loss(gen_loss)
            print(generator_gradients)
        return gen_loss

    def set_temperature(self, temperature):
        self.temperature = temperature

    def save(self, path, force=False):
        """Save the fitted model at the given path."""
        if os.path.exists(path) and not force:
            print('The indicated path already exists. Use `force=True` to overwrite.')
            return

        base_path = os.path.dirname(path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        print('Model saved successfully.')

