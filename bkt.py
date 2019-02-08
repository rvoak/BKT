import pandas as pd
import numpy as np
import io
import requests
import os
import wget
import numpy as np
from pyBKT.generate import synthetic_data
from pyBKT.fit import EM_fit
from copy import deepcopy
import pandas as pd
import sys
sys.path.append('../')



def DataHelper(skill_name):

  if not (os.path.isfile('data/skill_builder_data.csv')):
    os.mkdir('data')
    os.chdir('data')
    wget.download(http://users.wpi.edu/~yutaowang/data/skill_builder_data.csv)
    os.chdir('..')

  df = pd.read_csv('data/skill_builder_data.csv',encoding='ISO-8859-1')
  # filter by the skill you want, make sure the question is an 'original'
  skill = df[(df['skill_name']==skill_name) & (df['original'] == 1)]
  # sort by the order in which the problems were answered
  df.sort_values('order_id', inplace=True)

  # example of how to get the unique users
  # uilist=skill['user_id'].unique()

  # convert from 0=incorrect,1=correct to 1=incorrect,2=correct
  skill.loc[:,'correct']+=1

  # filter out garbage
  df3=skill[skill['correct']!=3]
  data=df3['correct'].values

  # find out how many problems per user, form the start/length arrays
  steps=df3.groupby('user_id')['problem_id'].count().values
  lengths=np.copy(steps)

  steps[0]=0
  steps[1]=1
  for i in range(2,steps.size):
    steps[i]=steps[i-1]+lengths[i-2]


  starts=np.delete(steps,0)

  resources=[1]*data.size
  resource=np.asarray(resources)

  stateseqs=np.copy(resource)
  lengths=np.resize(lengths,lengths.size-1)
  Data={}
  Data["stateseqs"]=np.asarray([stateseqs],dtype='int32')
  Data["data"]=np.asarray([data],dtype='int32')
  Data["starts"]=np.asarray(starts)
  Data["lengths"]=np.asarray(lengths)
  Data["resources"]=resource


  return (Data)



def main():
    df = pd.read_csv('skill_builder_data.csv',encoding='ISO-8859-1')
    #parameters
    num_subparts = 1
    num_resources = 2
    num_fit_initializations = 10
    observation_sequence_lengths = np.full(500, 100, dtype=np.int)

    #generate synthetic model and data.
    #model is really easy.
    truemodel = {}

    truemodel["As"] =  np.zeros((2, 2, num_resources), dtype=np.float_)
    for i in range(num_resources):
        truemodel["As"][i, :, :] = np.transpose([[0.7, 0.3], [0, 1]])
    truemodel["learns"] = truemodel["As"][:, 1, 0]
    truemodel["forgets"] = truemodel["As"][:, 0, 1]

    truemodel["pi_0"] = np.array([[0.9], [0.1]])
    truemodel["prior"] = truemodel["pi_0"][1][0]

    truemodel["guesses"] = np.full(num_subparts, 0.1, dtype=np.float_)
    truemodel["slips"] = np.full(num_subparts, 0.03, dtype=np.float_)

    #data!
    print("generating data...")
    skill='Pythagorean Theorem'
    data=DataHelper(skill)
    print('fitting! each dot is a new EM initialization')
    best_likelihood = float("-inf")
    fitmodel = deepcopy(truemodel) # NOTE: include this line to initialize at the truth
    (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
    if(log_likelihoods[-1] > best_likelihood):
      best_likelihood = log_likelihoods[-1]
      best_model = fitmodel

    print('')
    print('\ttruth\tlearned')
    for r in range(num_resources):
      print('learn%d\t%.4f\t%.4f' % (r+1, truemodel['As'][r, 1, 0].squeeze(), best_model['As'][r, 1, 0].squeeze()))
    for r in range(num_resources):
      print('forget%d\t%.4f\t%.4f' % (r+1, truemodel['As'][r, 0, 1].squeeze(), best_model['As'][r, 0, 1].squeeze()))

    for s in range(num_subparts):
      print('guess%d\t%.4f\t%.4f' % (s+1, truemodel['guesses'][s], best_model['guesses'][s]))
    for s in range(num_subparts):
      print('slip%d\t%.4f\t%.4f' % (s+1, truemodel['slips'][s], best_model['slips'][s]))
