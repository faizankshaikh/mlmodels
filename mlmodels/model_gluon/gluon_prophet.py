# -*- coding: utf-8 -*-
"""
GluonTS
# First install package from terminal:  pip install mxnet autogluon
https://autogluon.mxnet.io/tutorials/tabular_prediction/tabular-quickstart.html



"""
import json
import os
from pathlib import Path

import pandas as pd

from gluonts.model.prophet import ProphetPredictor
from gluonts.trainer import Trainer
from mlmodels.model_gluon.util import (
    _config_process, fit, get_dataset, load, log, metrics,
    os_package_root_path, plot_predict, plot_prob_forecasts, predict, save)

VERBOSE = False




########################################################################################################################
#### Model defintion
class Model(object) :
    def __init__(self, model_pars=None, compute_pars=None) :
        ## Empty model for Seaialization
        if model_pars is None and compute_pars is None :
            self.model = None
        else :
            m = model_pars
            self.model = ProphetPredictor(prediction_length=m['prediction_length'],freq=m['freq'])





########################################################################################################################
def get_params(choice="", data_path="dataset/", config_mode="test", **kw):
    if choice == "json":
       return _config_process(data_path, config_mode=config_mode)


    if choice == "test01":
        from mlmodels.util import path_local_setup
        log("#### Path params   ################################################")
        data_path, out_path, model_path = path_local_setup( __file__, sublevel=1,
                                                           out_folder="/ztest/gluon_phrophet/", 
                                                           data_path=data_path)

        train_data_path = data_path + "GLUON-GLUON-train.csv"
        test_data_path = data_path + "GLUON-test.csv"
        start = pd.Timestamp("01-01-1750", freq='1H')
        data_pars = {"train_data_path": train_data_path, "test_data_path": test_data_path, "train": False,
                     'prediction_length': 48, 'freq': '1H', "start": start, "num_series": 245,
                     "save_fig": "./series.png","modelpath":model_path}

        log("## Model params   ################################################")
        model_pars = {"prediction_length": data_pars["prediction_length"], "freq": data_pars["freq"]}
        compute_pars = {}
        outpath = out_path + "result"

        out_pars = {"outpath": outpath, "plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}

    return model_pars, data_pars, compute_pars, out_pars



########################################################################################################################
def test(data_path="dataset/"):
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=0, data_path=data_path)


    log("#### Loading dataset   #############################################")
    gluont_ds = get_dataset(**data_pars)


    log("#### Model init, fit   ###########################################")
    model = Model(model_pars, compute_pars)
    #model=m.model    ### WE WORK WITH THE CLASS (not the attribute GLUON )
    model=fit(model, data_pars, model_pars, compute_pars)

    log("#### save the trained model  #############################################")
    save(model, data_pars["modelpath"])


    log("#### Predict   ###################################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)
    print(ypred)


    log("#### metrics   ################################################")
    metrics_val, item_metrics = metrics(ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   #######################################################")
    plot_prob_forecasts(ypred, out_pars)
    plot_predict(item_metrics, out_pars)



if __name__ == '__main__':
    VERBOSE = True
    test(data_path="dataset/timeseries/" , choice="test01" )
