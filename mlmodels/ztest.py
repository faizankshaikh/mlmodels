# -*- coding: utf-8 -*-
import copy
import math
import os
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
# import scipy as sci
# import sklearn as sk

####################################################################################################
import tensorflow as tf
import torch as torch

# import autogluon
# import gluonts
####################################################################################################
import mlmodels


####################################################################################################
def get_recursive_files(folderPath, ext='/*model*/*.py'):
  import glob
  files = glob.glob( folderPath + ext, recursive=True) 
  return files


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)



####################################################################################################
from mlmodels.util  import os_package_root_path

  
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





def model_get_list(folder=None, block_list=[]):
  # Get all the model.py into folder  
  folder = os_package_root_path(__file__) if folder is None else folder
  # print(folder)
  module_names = get_recursive_files(folder, r'/*model*/*.py' )                       


  NO_LIST = [  "__init__", "util", "preprocess" ]
  NO_LIST = NO_LIST + block_list

  list_select = []
  for t in module_names :
      t = t.replace(folder, "").replace("\\", ".")

      flag = False     
      for x in NO_LIST :
        if x in t: flag = True

      if not flag  :
       list_select.append( t )


  return list_select


def test_model_structure():
  print("os.getcwd", os.getcwd())
  print(mlmodels) 

  path = mlmodels.__path__[0]
  
  print("############Check structure ############################")
  cmd = f"python {path}/ztest_structure.py"
  os.system( cmd )


def test_import() :
    print(np, np.__version__)
    print(tf, tf.__version__)
    print(torch, torch.__version__)
    print(mlmodels)



def main():
  print("os.getcwd", os.getcwd())

  path = mlmodels.__path__[0]
  print("############Check model ################################")
  model_list = model_get_list(folder=None, block_list=[])
  print(model_list)

  # return 7


  test_list =[    
   ### Tflow
   f"python {path}/model_tf/namentity_crm_bilstm.py",
 

    
   ### Keras
   f"python {path}/model_keras/01_deepctr.py",
  f"python {path}/model_keras/charcnn.py",


   f"python {path}/model_keras/01_deepctr.py",
   f"python {path}/model_keras/textcnn.py",


   ### SKLearn
   f"python {path}/model_sklearn/sklearn.py",


   ### Torch
   f"python {path}/model_tch/03_nbeats.py",
   f"python {path}/model_tch/textcnn.py",
   f"python {path}/model_tch/transformer_classifier.py",


   ### Glueon
   f"python {path}/model_gluon/gluon_deepar.py",
   f"python {path}/model_glufon/gluon_ffn.py",


   #### Too slow
   # f"python {path}/model_gluon/gluon_automl.py",


  ]


  

  path = path.replace("\\", "//")
  
  test_list =[ f"python {path}/"  + t.replace(".","//").replace("//py", ".py") for t in model_list ] 


  for cmd in test_list :
    print("\n\n\n", flush=True)
    print(cmd, flush=True)
    os.system( cmd )




#################################################################################################################
#################################################################################################################
def cli_load_arguments(config_file= None):
    """
        Load CLI input, load config.toml , overwrite config.toml by CLI Input
    """
    import argparse
    from util import load_config
    if config_file is None  :
      cur_path = os.path.dirname(os.path.realpath(__file__))
      config_file = os.path.join(cur_path, "template/test_config.json")
    # print(config_file)

    
    p = argparse.ArgumentParser()
    def add(*w, **kw) :
       p.add_argument(*w, **kw)
    
    add("--config_file", default=config_file, help="Params File")
    add("--config_mode", default="test", help="test/ prod /uat")
    add("--log_file", help="log.log")
    add("--do", default="all", help="test")
    add("--folder", default=None, help="test")
    
    ##### model pars



    ##### data pars


    ##### compute pars


    ##### out pars
    add("--save_folder", default="ztest/",  help=".")
    

    arg = p.parse_args()
    arg = load_config(arg, arg.config_file, arg.config_mode, verbose=0)
    return arg





if __name__ == "__main__":
  arg = cli_load_arguments()
  print(arg.do)

  if arg.do == "all"  :  #list all models in the repo                    
     main()

  if arg.do == "model_structure"  :  #list all models in the repo                    
     test_model_structure()





