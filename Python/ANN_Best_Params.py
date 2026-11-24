# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 17:55:02 2026

@author: nhand
"""

def get_best_params(GMType='Mix',num_features=2):
    best_params= {'hidden_units': [192, 96, 48, 24],
                  'learning_rate': 0.0066681836538019165,
                  'batch_size': 1024,
                  'weight_decay': 0.00045139366704422104,
                  }
    if GMType=='Mix':
        if num_features== 5:
            best_params= {'hidden_units': [192, 96, 48, 24],
                          'learning_rate': 0.0066681836538019165,
                          'batch_size': 1024,
                          'weight_decay': 0.00045139366704422104,
                          }
        if num_features== 2:
            best_params= {'hidden_units': [192, 96],
                          'learning_rate': 0.0020930470178492453,
                          'batch_size': 1024,
                          'weight_decay': 0.0006115026336724198,
                          }
    if GMType=='PulseLike':
        if num_features== 5:
            best_params= {'hidden_units': [192, 96],
                          'learning_rate': 0.00439284912574558,
                          'batch_size': 1024,
                          'weight_decay': 4.218699121275044e-06,
                          }
        if num_features== 2:
            best_params= {'hidden_units': [96, 48],
                          'learning_rate': 0.006991552482566878,
                          'batch_size': 4096,
                          'weight_decay': 8.573669051325601e-05,
                          }
    if GMType=='NoPulse':
        if num_features== 5:
            best_params= {'hidden_units': [128],
                          'learning_rate': 0.018685994379512155,
                          'batch_size': 2048,
                          'weight_decay': 0.00010595589391532446,
                          }
        if num_features== 2:
            best_params= {'hidden_units': [192, 96],
                          'learning_rate': 0.0021073721241382003,
                          'batch_size': 1024,
                          'weight_decay': 5.6454712218949895e-06,
                          }
    return best_params