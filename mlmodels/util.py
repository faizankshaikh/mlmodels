# -*- coding: utf-8 -*-
import os
import re
import sys

import toml


####################################################################################################
class to_namespace(object):
  def __init__(self, adict):
    self.__dict__.update(adict)
  
  def get(self, key):
    return self.__dict__.get(key)


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)



####################################################################################################
"""
def os_package_root_path(add_path="",n=0):
  from pathlib import Path
  add_path = os.path.join(Path(__file__).parent.absolute(), add_path)
  # print("os_package_root_path,check", add_path)
  return add_path
"""



def tf_deprecation() :
  try :
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    print("Deprecaton set to False")
  except : 
    pass



def params_json_load(path, config_mode="test"):
  import json
  pars = json.load(open(path, mode="rb"))
  pars = pars[config_mode]

  ### HyperParam, model_pars, data_pars,
  list_pars = [] 
  for t in [ "hypermodel_pars", "model_pars", "data_pars", "compute_pars", "out_pars" ] :
    pdict = pars.get(t)
    if pdict :
       list_pars.append(pdict)
    else :
       log("error in json, cannot load ",t)

  return tuple(list_pars)




def path_local_setup(current_file=None, out_folder="", sublevel=1, data_path="dataset/"):
    """
      mlmodels/dataset/
      mlmodels/ztest/  :  in gitgnore !!

    """
    root  = os_package_root_path(__file__, sublevel=0, path_add="")

    out_folder = path_norm(out_folder)
    data_path = path_norm(data_path)

    model_path = f"{out_path}/model/"
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    log(data_path, out_path, model_path)
    return data_path, out_path, model_path



def path_norm(path="") :

  if  len(path) == 0  :
    path = os_package_root_path(__file__, 0) 


  elif  path.startswith("model_") or path.startswith("/model_")   :
    path = os.path.join(os_package_root_path(__file__, 0) , path )   


  elif  path.startswith("dataset") or path.startswith("/dataset")   :
    path = os.path.join(os_package_root_path(__file__, 0) , path )   


  elif  path.startswith("template") or path.startswith("/template")   :
    path = os.path.join( os_package_root_path(__file__, 0) , path )    


  elif  path.startswith("ztest") or path.startswith("/ztest")   :
    path = os.path.join( os_package_root_path(__file__, 0) , path )    


  return path




def os_module_path():
  import  inspect
  current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  parent_dir = os.path.dirname(current_dir)
  # sys.path.insert(0, parent_dir)
  return parent_dir



def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    path = Path(os.path.realpath(filepath)).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path



def os_file_current_path():
  import inspect
  val = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  # return current_dir + "/"
  # Path of current file
  # from pathlib import Path

  # val = Path().absolute()
  val = str(os.path.join(val, ""))
  # print(val)
  return val



def log(*s, n=0,m=1):
  sspace = "#" * n
  sjump =  "\n" * m
  print(sjump, sspace, s, sspace, flush=True)




####################################################################################################
def test_module(model_uri="model_tf/1_lstm.py", data_path="dataset/", pars_choice="json", reset=True):
    ###loading the command line arguments
    #model_uri = "model_xxxx/yyyy.py"

    log("#### Module init   #################################################")
    from mlmodels.models import module_load
    module = module_load(model_uri)
    log(module)


    log("#### Loading params   ##############################################")
    param_pars = { "choice": pars_choice,  "data_path": data_path}
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)


    log("#### Run module test   ##############################################")
    from mlmodels.models import test_module as test_module_global
    test_module_global(model_uri, model_pars, data_pars, compute_pars, out_pars)









####################################################################################################
def load_config(args, config_file, config_mode, verbose=0):
    ##### Load file dict_pars as dict namespace #############################
    import json
    print(config_file) if verbose else None

    try:
       pars = json.load(open(config_file, mode="r") )
       # print(arg.param_file, model_pars)


       pars = pars[config_mode]  # test / prod
       print(config_file, pars) if verbose else None

       ### Overwrite dict_pars from CLI input and merge with toml file
       for key, x in vars(args).items():
          if x is not None:  # only values NOT set by CLI
             pars[key] = x

       # print(model_pars)
       pars = to_namespace(pars)  #  like object/namespace model_pars.instance
       return pars
       
    except Exception as e:
        print(e)
        return args



def val(x,xdefault) :
  try :
   return x if x is not None  else xdefault
  except :
      return xdefault



def get_recursive_files2(folderPath, ext):
    results = os.listdir(folderPath)
    outFiles = []
    for file in results:
      if os.path.isdir(os.path.join(folderPath, file)):
        outFiles += get_recursive_files(os.path.join(folderPath, file), ext)
      elif re.match(ext, file):
        outFiles.append(file)
  
    return outFiles


def get_recursive_files(folderPath, ext='/*model*/*.py'):
  import glob
  files = glob.glob( folderPath + ext, recursive=True) 
  return files


####################################################################################################
####################################################################################################
def env_conda_build(env_pars=None) :
   if env_pars is None :
     env_pars = { 'name' : "test" , 'python_version': '3.6.5'  }

   p = env_pars
   cmd = f"conda create -n {p['name']}  python={p['python_version']}  -y"
   print(cmd)
   os.system(cmd)


def env_pip_requirement(env_pars=None) :
   if env_pars is None :
     env_pars = { 'name' : "test" , 'file_requirement': 'requirements.txt'  }

   p = env_pars
   cmd = f"source activate {p['name']} "
   cmd = cmd + f"  && pip install -r  {p['file_requirement']}"

   print(cmd)
   os.system(cmd)



def env_build(model_uri, env_pars):
  from time import sleep

  model_uri2 = model_uri.replace("/", ".") 
  root       = os_package_root_path() 
  model_path = os.path.join(root, env_pars[ "model_path" ] )


  env_pars['name']             = model_uri2
  env_pars['python_version']   = "3.6.5"
  env_pars['file_requirement'] = model_path + "/requirements.txt"


  env_conda_build(env_pars=env_pars) 
  sleep(60)

  env_pip_requirement(env_pars=env_pars) 





####################################################################################################
####################################################################################################
def model_get_list(folder=None, block_list=[]):
  # Get all the model.py into folder  
  folder = os_package_root_path() if folder is None else folder
  # print(folder)
  module_names = get_recursive_files(folder, r'/*model*//*model*/*.py' )                       


  NO_LIST = [  "__init__.py", "util", "preprocess" ]
  NO_LIST = NO_LIST + block_list


  for t in module_names :
      t = t.replace(folder, "").replace("\\", ".")

      flag = False     
      for x in NO_LIST :
        if x in t: FLAG = True

      if not flag  :
       list_select.append( t )
  







####################################################################################################
########## TF specific #############################################################################
def load_tf(path, filename):
  """
  https://www.mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#

 """
  import mlflow
  import tensorflow as tf
  
  model_uri = path + "/" + filename
  tf_graph = tf.Graph()
  tf_sess = tf.Session(graph=tf_graph)
  with tf_graph.as_default():
    signature_def = mlflow.tensorflow.load_model(model_uri=model_uri,
                                                 tf_sess=tf_sess)
    input_tensors = [tf_graph.get_tensor_by_name(input_signature.name)
                     for _, input_signature in signature_def.inputs.items()]
    output_tensors = [tf_graph.get_tensor_by_name(output_signature.name)
                      for _, output_signature in signature_def.outputs.items()]
  return input_tensors, output_tensors


def save_tf(sess, file_path):
  import tensorflow as tf
  saver = tf.compat.v1.train.Saver()
  return saver.save(sess, file_path)





####################################################################################################
########## pyTorch specific ########################################################################
def load_tch(path, filename="model"):
  return 1


def save_tch(model, path, filename="model"):
  return 1



def load_pkl(path, filename="model"):
  return 1


def save_pkl(model, path, filename="model"):
  return 1




def load_gluon(path, filename="model"):
  return 1



def save_gluon(model, path, filename="model"):
  return 1




def load_keras(path, filename="model"):
  return 1


def save_keras(model, path, filename="model"):
  return 1






####################################################################################################
########## Other model specific ####################################################################









"""


import glob

path = "."
files = glob.glob(path + '/*model*/*.py', recursive=True)
files




from datetime import datetime, timedelt


a

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

sns.set()


class ModelFactory():
    def model_create(self, model_name, datasampler, hyper_parameters={'epoch':5}):
        if model_name == 'lstm':
            datasampler = DataSampler() # the reader object for the input data
            model =  LSTM(datasampler)
            model.set_pars(hyper_parameters)
            return model

# example usage
# data = DataSampler(file_name='x.csv', timestamp=5)
# this datasampler is sent to the model
class DataSampler():
    def __init__(self, file_name = '../dataset/GOOG-year.csv', timestamp =5):
        self.data_pars = pd.read_csv()
        self.date_ori = pd.to_datetime(data_pars.iloc[:, 0]).tolist()
        self.minmax = MinMaxScaler()
        self.timestamp = timestamp
        self.df_log = self.preprocess_df()
    
    def preprocess_df(self):
        self.minmax.fit(self.data_pars.iloc[:, 1:].astype('float32'))
        df_log = minmax.transform(data_pars.iloc[:, 1:].astype('float32'))
        df_log = pd.DataFrame(df_log)
        return df_log

    def batch_sampler(self, start, end):
        # sampler for training set
        for k in range(0, start, end):
            index   = min(k + timestamp, df_log.shape[0] -1)
            batch_x = np.expand_dims( df_log.iloc[k : index, :].values, axis = 0)
            batch_y = df_log.iloc[k + 1 : index + 1, :].values
            yield (batch_x, batch_y)
    
    def train_batch_sampler(self):
        return batch_sampler(self.df_log.shape[0] - 1, self.timestamp)
    
    def test_batch_sampler(self):
        return batch_sampler((self.df_log.shape[0] // self.timestamp) * self.timestamp, self.timestamp)
    
    def get_n_samples_per_batch(self):
        return self.df_log.shape[0] // self.timestamp
        


class BaseModelDl(object):
    
    #Base Model class used for models under Dl class 
    #acting as parent class

    def __init__(self):
        self.datasampler = None
        self.name = ''
        self.epoch = 5
        self.learning_rate = 0.01
        self.sess = None

    def get_pars(self):
        # epoch and learning rate exists in all models
        return {
            'epoch': self.epoch,
            'learning_rate': self.learning_rate
        }

    def set_pars(self, **parameters):
        # this function is common for all children classes
        for parameter, value in parameters.items():
            if hasattr(self,parameter):
               setattr(self, parameter, value)
            else:
                raise AttributeError('Class {} does not have parameter {}'.format(
                 self.name, parameter  
                ))
        return self

    def build_model(self):
        # used to create placeholders and optimizer based on parameters set
        # called after setting the parameters
        pass

    def fit(self):
        pass


    def predict(self,X):
        pass

    def load(self, model_path):
        # model_path=parent_folder/model.ckpt
        saver = tf.train.Saver()
        saver.load(self.sess, model_path)

    def save(self, model_path):
        # model_path=parent_folder/model.ckpt
        saver = tf.train.Saver()
        saver.save(self.sess, model_path)


class LSTM(BaseModelDl):
    def __init__(self, datasampler):
        super(LSTM, self).__init__()
        self.datasampler = datasampler
        self.name = 'lstm'
        self.num_layers = 1
        self.size_layer = 128
        self.timestamp = 5
        self.dropout_rate = 0.7
        self.future_day = 50
        self.learning_rate = 0.01
        self.feat_size = 1
        self.output_size = 1
        self.forget_bias = 0.1
        self.model = None


    def get_pars(self):
        return {
            'num_layers': self.num_layers,
            'size_layer': self.size_layer,
            'dropout_rate': self.dropout_rate,
            'epoch': self.epoch,
            'learning_rate': 0.01 ,
            'output_size': self.output_size,
            'feat_size': self.feat_size,
            'forget_bias': self.forget_bias
        }


    def build_model(self):
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
                    [lstm_cell(self.size_layer) for _ in range(self.num_layers)],
                    state_is_tuple = False,)
                    
        self.X = tf.placeholder(tf.float32, (None, None, self.feat_size))
        self.Y = tf.placeholder(tf.float32, (None, self.output_size))
        drop   = tf.contrib.rnn.DropoutWrapper(rnn_cells, output_keep_prob = self.forget_bias)
        self.hidden_layer = tf.placeholder( tf.float32, 
                                            (None, num_layers * 2 * size_layer))
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
        )
        self.logits    = tf.layers.dense(self.outputs[-1], output_size)
        self.cost      = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        self.sess    = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    
    def fit(self):
        for i in range(self.epoch):
            init_value = np.zeros((1, self.num_layers * 2 * self.size_layer))
            for batch_x, batch_y in self.datasampler.train_batch_sampler():
                last_state, _, loss = sess.run(
                    [self.last_state, self.optimizer, self.cost],
                    feed_dict = {
                        self.X: batch_x,
                        self.Y: batch_y,
                        self.hidden_layer: init_value,
                    },
                )
                init_value = last_state
                total_loss += loss
            total_loss /= self.datasampler.get_n_samples_per_batch()
            if (i + 1) % 100 == 0:
                print('epoch:', i + 1, 'avg loss:', total_loss)

    def predict(self,):
        # takes a sampler as the one in datasampler class
        # in this function it should take a sampler 
"""
