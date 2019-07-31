# this file was created to run parts of the evaluation on a different, more powerful machine.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns

import scipy.stats as stat
from scipy import interp

from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
# , ExtraTreesRegressor,RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, chi2
from sklearn.linear_model import LassoCV, LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.utils.multiclass import type_of_target
from sklearn.multiclass import OneVsRestClassifier

import time
import os
from datetime import datetime
import json

# need for missing value generation and imputation
from xgbimputer import MissXGB


import itertools

# used for permutation importance, currently not actually used

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families.family import Binomial, Gamma
from statsmodels.tools.tools import add_constant
from statsmodels.stats import proportion

from xgboost import XGBClassifier, XGBRegressor
import pylab
import dill
import gc

np.random.seed(1234)


def generate_gen_out_vec(df, cat_cols):
    x = np.repeat(1, len(df.columns)-len(cat_cols))
    x = np.concatenate((x, (np.max(df[cat_cols].apply(
        lambda x: x.astype('category')).apply(lambda x: x.cat.codes))+1)))
    return x

# you can find all datasets at: http://fairness-measures.org/Pages/Datasets

# Doing 2016 sqf instead because i am lazy :)))
# columns taken from: https://github.com/adewes/fatml-pydata/blob/master/stop-and-frisk.ipynb


def gen_sqf16_data():
    # data:
    raw_sqf16 = pd.read_csv(
        "https://www1.nyc.gov/assets/nypd/downloads/excel/analysis_and_planning/stop-question-frisk/sqf-2016.csv", low_memory=False)

    # last row is empty
    raw_sqf16.drop(raw_sqf16.tail(1).index, inplace=True)

    numeric_attr = [
        # appearance
        'age',      # SUSPECT'S AGE                 N
        'weight',   # SUSPECT'S WEIGHT              N
        # environment
        'timestop_hour',  # N
        'timestop_minute',  # orignally timestop, converted below into two cols N
        # PRECINCT OF STOP (FROM 1 TO 123), actually should be C, but hmm...
        'pct'
    ]

    cat_attr = [
        # appearance
        # 'ht_feet',  # SUSPECT'S HEIGHT (FEET)       C
        'race',     # SUSPECT'S RACE                C
        'sex',      # SUSPECT'S SEX                 C
        'build',    # SUSPECT'S BUILD               C
        # environment
        'inout',    # WAS STOP INSIDE OR OUTSIDE?   C
        'trhsloc',  # WAS LOCATION HOUSING OR TRANSIT AUTHORITY? C
    ]

    yes_no_behavior_attribs = [
        'ac_evasv',  # EVASIVE RESPONSE TO QUESTIONING
        'ac_assoc',  # ASSOCIATING WITH KNOWN CRIMINALS
        'cs_lkout',  # SUSPECT ACTING AS A LOOKOUT
        'cs_drgtr',  # ACTIONS INDICATIVE OF A DRUG TRANSACTION
        'cs_casng',  # CASING A VICTIM OR LOCATION
        'cs_vcrim',  # VIOLENT CRIME SUSPECTED
        'ac_cgdir',  # CHANGE DIRECTION AT SIGHT OF OFFICER
        'cs_furtv',  # FURTIVE MOVEMENTS
        'ac_stsnd',  # SIGHTS OR SOUNDS OF CRIMINAL ACTIVITY
    ]

    yes_no_environment_attribs = [
        'ac_proxm',  # PROXIMITY TO SCENE OF OFFENSE
        'cs_other',  # OTHER
        'ac_rept',   # REPORT BY VICTIM / WITNESS / OFFICER
        'ac_inves',  # ONGOING INVESTIGATION
        'ac_incid',  # AREA HAS HIGH CRIME INCIDENCE
        'ac_time',   # TIME OF DAY FITS CRIME INCIDENCE
    ]

    yes_no_appearance_attribs = [
        'cs_cloth',  # WEARING CLOTHES COMMONLY USED IN A CRIME
        'cs_objcs',  # CARRYING SUSPICIOUS OBJECT
        'cs_bulge',  # SUSPICIOUS BULGE
        'cs_descr',  # FITS A RELEVANT DESCRIPTION
        'rf_attir',  # INAPPROPRIATE ATTIRE FOR SEASON
    ]

    yes_no_frisk_attribs = [
        'rf_othsw',  # OTHER SUSPICION OF WEAPONS
        'rf_knowl',  # KNOWLEDGE OF SUSPECTS PRIOR CRIMINAL BEHAVIOR
        'rf_vcact',  # ACTIONS OF ENGAGING IN A VIOLENT CRIME
        'rf_verbl',  # VERBAL THREATS BY SUSPECT
    ]

    yes_no_target_attribs = [
        'arstmade',  # WAS AN ARREST MADE?
        'frisked',  # WAS SUSPECT FRISKED?
        'sumissue'  # WAS A SUMMONS ISSUED?
    ]

    bin_attr = np.concatenate((yes_no_behavior_attribs, yes_no_environment_attribs,
                              yes_no_appearance_attribs, yes_no_frisk_attribs, yes_no_target_attribs))

    for attrib in bin_attr:
        raw_sqf16[attrib] = raw_sqf16[attrib].map(
            dict(Y=1, N=0)).apply(np.uint8)

    raw_sqf16['timestop'] = raw_sqf16['timestop'].apply(
        str).apply('{:0>4}'.format)
    raw_sqf16['timestop_hour'] = raw_sqf16.timestop.apply(
        lambda x: datetime.strptime(x, '%H%M').hour)
    raw_sqf16['timestop_minute'] = raw_sqf16.timestop.apply(
        lambda x: datetime.strptime(x, '%H%M').minute)
    raw_sqf16.drop('timestop', axis=1, inplace=True)

    for attrib in numeric_attr:
        raw_sqf16[attrib] = pd.to_numeric(raw_sqf16[attrib], errors='coerce')

    raw_sqf16[numeric_attr] = raw_sqf16[numeric_attr].apply(
        lambda x: (x-x.min())/(x.max()-x.min()), axis=0)
    cat_vector = np.concatenate((np.repeat(1, len(numeric_attr)+len(bin_attr)), np.max(
        raw_sqf16[cat_attr].apply(lambda x: x.astype('category')).apply(lambda x: x.cat.codes))+1))

    raw_sqf16 = raw_sqf16[np.concatenate((numeric_attr, bin_attr, cat_attr))]
    sqf16_dum = pd.get_dummies(data=raw_sqf16, columns=cat_attr)

    sqf16_dum.dropna(inplace=True)

    X_train_numpy = sqf16_dum.copy()
    sqf16 = tf.convert_to_tensor(sqf16_dum.values, dtype='float32')

    X_train_onehot = sqf16
    print(X_train_onehot.shape)
    return X_train_onehot, X_train_numpy, cat_vector


def gen_compas_data():
    # Used: https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    # as base for what is important in that dataset as they do investigation into unfairness

    raw_compas = pd.read_csv(
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")

    cmp = raw_compas[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
        'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
    cmp = cmp.query(
        'days_b_screening_arrest <= 30 and days_b_screening_arrest >= -30 and is_recid != -1 and c_charge_degree != "O" and score_text != "N/A"')
    cmp = cmp.reset_index(drop=True)

    # editing such that this will work for gan
    # age_cat and score_text can be derived from other variables and are therefore removed
    tmp = pd.Series(delta.total_seconds(
    )/3600 for delta in (pd.to_datetime(cmp.c_jail_out)-pd.to_datetime(cmp.c_jail_in)))
    cmp['length_of_stay'] = tmp
    cmp = cmp.drop(['c_jail_out', 'c_jail_in',
                   'age_cat', 'score_text'], axis=1)

    # Prepare categorical columns
    cat_cols = ['c_charge_degree', 'race', 'sex', 'is_recid', 'two_year_recid']

    # Generate vector for number of outputs for softmax
    cat_vector = generate_gen_out_vec(cmp, cat_cols)

    # Create dummy variables
    cmp_dum = pd.get_dummies(data=cmp, columns=cat_cols)

    # Rescale numeric columns
    num_cols = len(cmp.columns)-len(cat_cols)
    cmp_dum.iloc[:, :num_cols] = cmp_dum.iloc[:, :num_cols].apply(
        lambda x: (x-x.min())/(x.max()-x.min()), axis=0)  # 1,0

    X_train_numpy = cmp_dum.copy()
    cmp_dum = tf.convert_to_tensor(cmp_dum.values, dtype='float32')

    X_train_onehot = cmp_dum
    print(X_train_onehot.shape)
    return X_train_onehot, X_train_numpy, cat_vector


# https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

def gen_schufa_data():
    raw_schufa = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data", sep=' ', header=None)

    # Prepare categorical columns
    cat_cols = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 19, 20]

    # Generate vector for number of outputs for softmax
    cat_vector = generate_gen_out_vec(raw_schufa, cat_cols)

    # Create dummy variables
    schufa_dum = pd.get_dummies(data=raw_schufa, columns=cat_cols)

    # Rescale numeric columns
    num_cols = len(raw_schufa.columns)-len(cat_cols)

    schufa_dum.iloc[:, :num_cols] = schufa_dum.iloc[:, :num_cols].apply(
        lambda x: (x-x.min())/(x.max()-x.min()), axis=0)  # 1,0

    X_train_numpy = schufa_dum.copy()
    schufa_dum = tf.convert_to_tensor(schufa_dum.values, dtype='float32')

    X_train_onehot = schufa_dum
    print(X_train_onehot.shape)
    return X_train_onehot, X_train_numpy, cat_vector


def gen_chile_data():
    raw_chile = pd.read_csv(
        'C:/Users/Elias/Desktop/UnfairGAN Stuff/Chile_dataset/ADMISION2017_Refractored_replaced.csv', sep=";")

    # Prepare categorical columns
    cat_cols = ['Nationality [P008]',
                'Gender [P009]',
                'Civil status [P019]',
                'Income decile [P034]',
                'Education of father [P037]',
                'Education of mother [P038]',
                'Occupational status of father [P039]',
                'Occupational status of mother [P040]',
                'Main occupation of father [P043]',
                'Main occupation of mother [P044]',
                'Region [P056]',
                'Type of high school [P077]']

    # Generate vector for number of outputs for softmax
    cat_vector = generate_gen_out_vec(raw_chile, cat_cols)

    # Create dummy variables
    chile_dum = pd.get_dummies(data=raw_chile, columns=cat_cols)

    # Rescale numeric columns
    num_cols = len(raw_chile.columns)-len(cat_cols)

    chile_dum.iloc[:, :num_cols] = chile_dum.iloc[:, :num_cols].apply(
        lambda x: (x-x.min())/(x.max()-x.min()), axis=0)  # 1,0

    X_train_numpy = chile_dum.copy()
    chile_dum = tf.convert_to_tensor(chile_dum.values, dtype='float32')

    X_train_onehot = chile_dum
    print(X_train_onehot.shape)
    return X_train_onehot, X_train_numpy, cat_vector


# Load data from file or generate it again
dataset_name = 'chile'

datasets = {'sqf': gen_sqf16_data,
            'cmp': gen_compas_data,
            'schufa': gen_schufa_data,
            'chile': gen_chile_data}


X_train_onehot, X_train_numpy, cat_vector = datasets[dataset_name]()

load_from_file = True  

work_dir_ = os.path.join(
    'C:/Users/Elias/Desktop/UnfairGAN Stuff/', dataset_name)
os.makedirs(work_dir_, exist_ok=True)

# create filename
filename_eval = dataset_name+'_eval.pkl'
filepath_eval = os.path.join(work_dir_, filename_eval)

filename_prot = dataset_name+'_eval_prot.pkl'
filepath_prot = os.path.join(work_dir_, filename_prot)


def sample_from_prob(samples):
    # get point where values change to find first categorical column:
    idx = np.where(cat_vector[:-1] != cat_vector[1:])[0][0]+1
    cat_vec_short = cat_vector[idx:]
    # loop over all categorical variables and sample from distributions
    for i in cat_vec_short:
        tmp = [np.random.choice(i, size=1, p=(j/np.sum(j)))[0]
                                for j in samples.values[:, idx:i+idx]]
        dummy = np.zeros([len(tmp), i])
        dummy[np.arange(len(tmp)), tmp] = 1
        samples.values[:, idx:idx+i] = dummy
        idx += i
    return samples


eval_orig = X_train_numpy.copy()
if(dataset_name == 'chile'):
    eval_gen = pd.read_csv('C:/Users/Elias/Desktop/UnfairGAN Stuff/' +
                           dataset_name+'/data-2019-04-09-08-25.csv', index_col=0)
    eval_gen = sample_from_prob(eval_gen)
elif(dataset_name == 'cmp'):
    eval_gen = pd.read_csv('C:/Users/Elias/Desktop/UnfairGAN Stuff/' +
                           dataset_name+'/data-2019-02-19-09-34.csv', index_col=0)
    eval_gen = sample_from_prob(eval_gen)
elif(dataset_name == 'schufa'):
    eval_gen = pd.read_csv('C:/Users/Elias/Desktop/UnfairGAN Stuff/' +
                           dataset_name+'/data-2019-05-07-08-25.csv', index_col=0)
    eval_gen = sample_from_prob(eval_gen)
elif(dataset_name == 'sqf'):
    eval_gen = pd.read_csv('C:/Users/Elias/Desktop/UnfairGAN Stuff/' +
                           dataset_name+'/data-2019-02-05-14-34.csv', index_col=0)
    eval_gen = sample_from_prob(eval_gen)


idx = 0
drop_cols = []
for i in cat_vector:
    if i > 1:
        drop_cols.append(idx)
        idx += i
    else:
        idx += 1

print(drop_cols)
eval_orig.drop(eval_orig.columns[drop_cols], axis=1, inplace=True)
eval_gen.drop(eval_gen.columns[drop_cols], axis=1, inplace=True)


def add_nan_noise(df_old, missing):
    df = df_old.copy()
    sample_size = np.int32(len(df.index)*len(df.columns)*missing + .5)
    poss_locs = np.array(list(itertools.product(
        range(len(df.index)), range(len(df.columns)))))
    tuples = poss_locs[np.random.choice(
        range(len(poss_locs)), sample_size, replace=False)]
    for s in tuples:
        df.iloc[tuple(s)] = np.nan
    return df


def add_col_nan_noise(df_old, missing):
    df = df_old.copy()
    samp_cols = df.columns.to_series().sample(np.int8(len(df.columns)*(1-missing)+.5))
    df[samp_cols] = np.nan
    return df


def impute_missing(df,dataset_name):
    if(dataset_name=='schufa'):
        imputer = MissXGB(n_estimators=10, max_iter=20,use_gpu=False)
    else:
        imputer = MissXGB(n_estimators=10, max_iter=20)
    return imputer.fit_transform(df)


def create_combined_set(data_real, data_gen, missing, n_percent,dataset_name,full_noise=False):
    data_real = data_real.copy()
    data_gen = data_gen.copy()
    y_cols = data_real[2].columns
    df_real = pd.concat([data_real[0], data_real[2]], axis=1)
    df_gen = pd.concat([data_gen[0], data_gen[2]], axis=1)
    df_real = df_real.sample(frac=n_percent)
    df_gen = df_gen.sample(frac=n_percent)

    df_real_y = df_real[y_cols]
    df_gen_y = df_gen[y_cols]
    df_real_x = df_real.drop(y_cols, axis=1)
    df_gen_x = df_gen.drop(y_cols, axis=1)

    if(full_noise):
        df_real = add_nan_noise(df_real_x, missing)
    else:
        df_real = add_col_nan_noise(df_real_x, missing)

    combined = pd.concat([df_real, df_gen_x], axis=0)
    df_real_imp = impute_missing(combined,dataset_name)[0:len(df_real_x.index), :]
    return [df_gen_x, df_real_imp, df_gen_y, df_real_y]


class CV_pred_protected():

    def __init__(self, eval_orig, eval_gen, prot_dict,dataset_name):
        self.dict = prot_dict
        self.data_orig = eval_orig
        self.data_gen = eval_gen
        self.dataset_name = dataset_name
        self.eval_dict = {}

    def create_var_predset(self, cols):
        data_y_o, data_y_g = [], []

        data_y_o = self.data_orig[cols].astype(np.int8)
        data_y_g = self.data_gen[cols].astype(np.int8)

        data_x_o, data_x_g = self.data_orig.drop(
            cols, axis=1), self.data_gen.drop(cols, axis=1)

        return data_x_o, data_x_g, data_y_o, data_y_g

    def interpRates(self, x):
        fpr, tpr, _ = x
        base_fpr = np.linspace(0, 1, 100)
        tpr_1 = interp(base_fpr, fpr, tpr)
        fpr_1 = interp(base_fpr, tpr, fpr)
        return fpr_1, tpr_1

    def get_cfm_auc(self, data, v):
        fpr = []
        tpr = []
        roc_auc = []

        models = [LogisticRegression(solver='liblinear', C=2e10, max_iter=1000),
                        RandomForestClassifier(n_estimators=100, max_depth=5)]

        if(len(v) >= 2):

            data[2] = pd.get_dummies(data[2])
            data[3] = pd.get_dummies(data[3])
            for model in models:
                model1 = OneVsRestClassifier(model).fit(data[0], data[2])
                pred = model1.predict_proba(data[1])

                for i in range(len(data[2].columns)):
                    fpr_, tpr_ = self.interpRates(
                        roc_curve(data[3].iloc[:, i], pred[:, i]))

                    fpr.append(fpr_)
                    tpr.append(tpr_)
                    roc_auc.append(auc(fpr_, tpr_))

                # Compute micro-average ROC curve and ROC area
                fpr_, tpr_ = self.interpRates(
                    roc_curve(data[3].values.ravel(), pred.ravel()))

                fpr.append(fpr_)
                tpr.append(tpr_)
                roc_auc.append(auc(fpr_, tpr_))

        else:
            for model in models:
    #         model1 = GLM(data[2],add_constant(data[0]),family = Binomial()).fit()
    #         m1_pred = model1.predict(add_constant(data[1]))
    #         m1_round = np.int8(m1_pred > .5)
                model = model.fit(data[0], data[2].values.ravel())
                pred = model.predict_proba(data[1])[:, 1]

                fpr1, tpr1 = self.interpRates(roc_curve(data[3], pred))
                fpr.append(fpr1)
                tpr.append(tpr1)
                roc_auc.append(auc(fpr1, tpr1))

        return fpr, tpr, roc_auc

    def cv_pred_protected_var(self, data_x_o, data_x_g, data_y_o, data_y_g, v):
        random_iter = 10
        skf = StratifiedKFold(n_splits=self.n_splits)
        fpr = []
        tpr = []
        roc_auc = []
        cv_split = 0
        for df1, df2 in zip(skf.split(data_x_o, data_y_o), skf.split(data_x_g, data_y_g)):
            print(cv_split)
            fpr_i = []
            tpr_i = []
            roc_auc_i = []

            data_r = [data_x_o.iloc[df1[0]], data_x_o.iloc[df1[1]],
                data_y_o.take(df1[0]), data_y_o.take(df1[1])]
            data_g = [data_x_g.iloc[df2[0]], data_x_g.iloc[df2[1]],
                data_y_g.take(df2[0]), data_y_g.take(df2[1])]
            # data_gr= [data_g[0],data_r[1],data_g[2],data_r[3]]

            for i in [data_r, data_g]:  # ,data_gr]:
                fpr_, tpr_, roc_auc_ = self.get_cfm_auc(i, v)
                fpr_i.append(fpr_)
                tpr_i.append(tpr_)
                roc_auc_i.append(roc_auc_)

            
            for i in np.arange(1.0, 0.0, -.1):  # from 90% to 10%
                fpr_miss = []
                tpr_miss = []
                auc_miss = []

                fpr_imp_miss = []
                tpr_imp_miss = []
                auc_imp_miss = []
                if(i == 1):
                    # samp_cols = data_x_o.columns.to_series().sample(
                    #     np.int8(len(data_x_o.columns)*i+.5))
                    data_miss = [data_g[0], data_r[1]
                        , data_g[2], data_r[3]]
                    fpr_, tpr_, roc_auc_ = self.get_cfm_auc(data_miss, v)
                    fpr_miss.append(fpr_)
                    tpr_miss.append(tpr_)
                    auc_miss.append(roc_auc_)
                    fpr_i.append(np.mean(fpr_miss, axis=0))
                    tpr_i.append(np.mean(tpr_miss, axis=0))
                    roc_auc_i.append(np.mean(auc_miss, axis=0))
                else:
                    for _ in range(random_iter):
                        samp_cols = data_x_o.columns.to_series().sample(
                            np.int32(len(data_x_o.columns)*i+.5))
                        data_miss = [data_g[0][samp_cols], data_r[1]
                            [samp_cols], data_g[2], data_r[3]]
                        fpr_, tpr_, roc_auc_ = self.get_cfm_auc(data_miss, v)
                        fpr_miss.append(fpr_)
                        tpr_miss.append(tpr_)
                        auc_miss.append(roc_auc_)
                    fpr_i.append(np.mean(fpr_miss, axis=0))
                    tpr_i.append(np.mean(tpr_miss, axis=0))
                    roc_auc_i.append(np.mean(auc_miss, axis=0))
                    for j in [0.5, .3, .1]:
                        for _ in range(random_iter):

                            data_imp_miss = create_combined_set(
                                data_r, data_g, missing=i, n_percent=j,dataset_name=self.dataset_name)
                            fpr_, tpr_, roc_auc_ = self.get_cfm_auc(
                                data_imp_miss, v)
                            fpr_imp_miss.append(fpr_)
                            tpr_imp_miss.append(tpr_)
                            auc_imp_miss.append(roc_auc_)
                        fpr_i.append(np.mean(fpr_imp_miss, axis=0))
                        tpr_i.append(np.mean(tpr_imp_miss, axis=0))
                        roc_auc_i.append(np.mean(auc_imp_miss, axis=0))
            fpr.append(fpr_i)
            tpr.append(tpr_i)
            roc_auc.append(roc_auc_i)
            cv_split+=1
        return fpr, tpr,roc_auc

    def evaluate(self, n_splits):
        self.n_splits = n_splits
        for k, v in self.dict.items():
            gc.collect()
            print('Evaluating:', k,'...')
            data_x_o, data_x_g,data_y_o,data_y_g = self.create_var_predset(v)

            if(type_of_target(data_y_g) =='multilabel-indicator'):
                data_y_g = data_y_g.idxmax(axis=1)
                data_y_o = data_y_o.idxmax(axis=1)

            fpr,tpr,roc_auc = self.cv_pred_protected_var(data_x_o,data_x_g,data_y_o,data_y_g,v)

            self.eval_dict[k] = [fpr, tpr,roc_auc]

    # wilcoxon test for auc:
    def wilcox_eval(self, key,mod,f):
        print(key,file=f)
        models = ['logit', 'rf']
        auc = self.eval_dict[key][2]
        print('model: ', models[mod],file=f)
        v = self.dict[key]
        cnt = 0
        for i in range(len(v)):
            print(v[i],file=f)

            wc_r_gr = stat.wilcoxon([auc[cv][0][cnt+mod::len(models)][0] for cv in range(self.n_splits)], [auc[cv][2][cnt+mod::len(models)][0] for cv in range(self.n_splits)])
            wc_r_g = stat.wilcoxon([auc[cv][0][cnt+mod::len(models)][0] for cv in range(self.n_splits)], [auc[cv][1][cnt+mod::len(models)][0] for cv in range(self.n_splits)])
            wc_g_gr = stat.wilcoxon([auc[cv][1][cnt+mod::len(models)][0] for cv in range(self.n_splits)], [auc[cv][2][cnt+mod::len(models)][0] for cv in range(self.n_splits)])

            for wc, n in zip([wc_r_g,wc_r_gr,wc_g_gr],['wc_r_g','wc_r_gr','wc_g_gr']):
                print(n,file=f)
                if(wc[0] >5 and wc[1]<0.05):
                    print(wc,file=f)
                    print(
                        'H0 rejected, difference does not have a distribution with mean 0',file=f)
                elif(wc[1] >.05):
                    print(wc,file=f)
                    print('not statistically significant',file=f)
                else:
                    print(wc,file=f)
                    print('Cannot reject H0',file=f)
                # return wc_1,wc_2

    def _print_rates(self, fpr,tpr,auc,file):
        print('Mean False positive rate', fpr,'Mean True positive rate', tpr,'Mean AUC', auc,file=file) 

    def summary(self):
        f = open(os.path.join(work_dir_,'prot_ev_summary.txt'),'w')
        print('Mean AUC+Confusion matrix for logit, rf using (real,gen,gr):',file=f)
        models = ['glm', 'rf']
        data_n = ['real', 'gen','train_gen_predict_real',
        'data_gr_90','data_comb_90_0.5','data_comb_90_0.3','data_comb_90_0.1',
        'data_gr_80','data_comb_80_0.5','data_comb_80_0.3','data_comb_80_0.1',
        'data_gr_70','data_comb_70_0.5','data_comb_70_0.3','data_comb_70_0.1',
        'data_gr_60','data_comb_60_0.5','data_comb_60_0.3','data_comb_60_0.1',
        'data_gr_50','data_comb_50_0.5','data_comb_50_0.3','data_comb_50_0.1',
        'data_gr_40','data_comb_40_0.5','data_comb_40_0.3','data_comb_40_0.1',
        'data_gr_30','data_comb_30_0.5','data_comb_30_0.3','data_comb_30_0.1',
        'data_gr_20','data_comb_20_0.5','data_comb_20_0.3','data_comb_20_0.1',
        'data_gr_10','data_comb_10_0.5','data_comb_10_0.3','data_comb_10_0.1']

        for i in range(len(models)):
            print(models[i], '\n---------------------------',file=f)
            for k, v in self.eval_dict.items():
                print(k,file=f)
                fpr = v[0]
                tpr = v[1]
                roc_auc = v[2]

                if(len(self.dict[k]) >1):
                    cnt = 0
                    for l in self.dict[k]:
                        print('Summary for', l,'...',file=f)
                        for j in range(len(data_n)):
                            # print(data_n[j],file=f)    
                            mean_fpr = np.nanmean(
                                [fpr[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                            mean_tpr = np.nanmean(
                                [tpr[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                            mean_roc_auc = np.nanmean(
                                [roc_auc[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                            self._print_rates(mean_fpr, mean_tpr,mean_roc_auc,f)
                        cnt += 1
                    print('micro average?',file=f)
                    for j in range(len(data_n)):
                        # print(data_n[j],file=f)    
                        mean_fpr = np.nanmean(
                            [fpr[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                        mean_tpr = np.nanmean(
                            [tpr[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                        mean_roc_auc = np.nanmean(
                            [roc_auc[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                        self._print_rates(mean_fpr, mean_tpr,mean_roc_auc,f)
                else:
                    for j in range(len(data_n)):
                        # print(data_n[j],file=f)    
                        mean_fpr = np.nanmean([fpr[cv][j][i]
                                            for cv in range(self.n_splits)])
                        mean_tpr = np.nanmean([tpr[cv][j][i]
                                            for cv in range(self.n_splits)])
                        mean_roc_auc = np.nanmean(
                            [roc_auc[cv][j][i] for cv in range(self.n_splits)])
                        self._print_rates(mean_fpr, mean_tpr,mean_roc_auc,f)

            for key in self.eval_dict.keys():
                self.wilcox_eval(key,i,f)


eval_orig_c = eval_orig.copy()
eval_gen_c = eval_gen.copy()

if (dataset_name =='cmp'):
    g1 = ['sex_Male']
    r1 = ['race_Asian', 'race_Caucasian', 'race_Hispanic',  'race_Other'] # 'race_African-American','race_Native American'
    try:

        for i in [eval_orig_c, eval_gen_c]:
            i['age1'] = np.int8(i['age'] <= .33)
            i['age3'] = np.int8(i['age'] > .66)
            i['age2'] = np.int8(np.logical_not(np.logical_or(i['age1'], i['age3'])))
            i.drop(columns='age', inplace=True)
    except:
        print("age was already converted")
    ages = ['age1', 'age2','age3']

    prot_dict = {'gender': g1,'ethnicity':r1,'age':ages}

elif (dataset_name == 'schufa'):
    eval_orig_c.columns = eval_gen_c.columns
    try:
        for i in [eval_orig_c, eval_gen_c]:
            i['age1'] = np.int8(i['12'] <= .33)
            i['age3'] = np.int8(i['12'] > .66)
            i['age2'] = np.int8(np.logical_not(np.logical_or(i['age1'], i['age3'])))
            i.drop(columns='12', inplace=True)
    except:
        print("age was already converted")
    ages = ['age1', 'age2','age3']

    sexes = ['8_A92', '8_A93','8_A94']

    prot_dict = {'gender/marriage': sexes,'age':ages}
elif (dataset_name == 'chile'):

    eval_gen_c.columns = [s.replace('[', '').replace(']', '') for s in eval_gen_c.columns]
    eval_orig_c.columns = [s.replace('[', '').replace(']', '') for s in eval_orig_c.columns]
    n1 = ['Nationality P008_2']

    g1 = ['Gender P009_2']

    r1 = ['Region P056_2']

    id1 = ['Income decile P034_2', 'Income decile P034_3','Income decile P034_4','Income decile P034_5','Income decile P034_6','Income decile P034_7','Income decile P034_8','Income decile P034_9','Income decile P034_10']

    prot_dict = {'nationality': n1,'gender':g1,'region':r1,'income decile':id1}

elif (dataset_name == 'sqf'):

    g1 = ['sex_M', 'sex_Z']
    try:
        for i in [eval_orig_c, eval_gen_c]:
            i['age1'] = np.int8(i['age'] <= .33)
            i['age3'] = np.int8(i['age'] > .66)
            i['age2'] = np.int8(np.logical_not(np.logical_or(i['age1'], i['age3'])))
            i.drop(columns='age', inplace=True)
    except:
        print("age was already converted")
    ages = ['age1', 'age2','age3']
    r1 = ['race_B', 'race_P','race_Q','race_W'] 


    prot_dict = {'gender': g1,'ethnicity':r1,'age':ages}

# SET THIS ONLY FOR THE PURPOSE OF REPLACING OLD FUNCTIONS OF PAST GENERATED RESULTS
load_from_pkl = False

if(not load_from_pkl):
    prot_ev = CV_pred_protected(eval_orig_c, eval_gen_c,prot_dict,dataset_name)
    prot_ev.evaluate(5)
    prot_ev.summary()
    dill.dump(prot_ev, open(filepath_prot,'wb'))

def n_print_rates(self, fpr,tpr,auc,file):
        print('Mean False positive rate', fpr,'Mean True positive rate', tpr,'Mean AUC', auc,file=file)

def n_wilcox_eval(self, key,mod,f):
    print(key,file=f)
    models = ['logit', 'rf']
    auc = self.eval_dict[key][2]
    print('model: ', models[mod],file=f)
    v = self.dict[key]
    cnt = 0
    for i in range(len(v)):
        print(v[i],file=f)

        wc_r_gr = stat.wilcoxon([auc[cv][0][cnt+mod::len(models)][0] for cv in range(self.n_splits)], [auc[cv][2][cnt+mod::len(models)][0] for cv in range(self.n_splits)])
        wc_r_g = stat.wilcoxon([auc[cv][0][cnt+mod::len(models)][0] for cv in range(self.n_splits)], [auc[cv][1][cnt+mod::len(models)][0] for cv in range(self.n_splits)])
        wc_g_gr = stat.wilcoxon([auc[cv][1][cnt+mod::len(models)][0] for cv in range(self.n_splits)], [auc[cv][2][cnt+mod::len(models)][0] for cv in range(self.n_splits)])

        for wc, n in zip([wc_r_g,wc_r_gr,wc_g_gr],['wc_r_g','wc_r_gr','wc_g_gr']):
            print(n,file=f)
            if(wc[0] >5 and wc[1]<0.05):
                print(wc,file=f)
                print(
                    'H0 rejected, difference does not have a distribution with mean 0',file=f)
            elif(wc[1] >.05):
                print(wc,file=f)
                print('not statistically significant',file=f)
            else:
                print(wc,file=f)
                print('Cannot reject H0',file=f)

def new_summary(self):
    f = open(os.path.join(work_dir_,'prot_ev_summary.txt'),'w')
    print('Mean AUC+Confusion matrix for logit, rf using (real,gen,gr):',file=f)
    models = ['glm', 'rf']
    data_n = ['real', 'gen','train_gen_predict_real',
    'data_gr_90','data_comb_90_0.5','data_comb_90_0.3','data_comb_90_0.1',
    'data_gr_80','data_comb_80_0.5','data_comb_80_0.3','data_comb_80_0.1',
    'data_gr_70','data_comb_70_0.5','data_comb_70_0.3','data_comb_70_0.1',
    'data_gr_60','data_comb_60_0.5','data_comb_60_0.3','data_comb_60_0.1',
    'data_gr_50','data_comb_50_0.5','data_comb_50_0.3','data_comb_50_0.1',
    'data_gr_40','data_comb_40_0.5','data_comb_40_0.3','data_comb_40_0.1',
    'data_gr_30','data_comb_30_0.5','data_comb_30_0.3','data_comb_30_0.1',
    'data_gr_20','data_comb_20_0.5','data_comb_20_0.3','data_comb_20_0.1',
    'data_gr_10','data_comb_10_0.5','data_comb_10_0.3','data_comb_10_0.1']

    for i in range(len(models)):
        print(models[i], '\n---------------------------',file=f)
        for k, v in self.eval_dict.items():
            print(k,file=f)
            fpr = v[0]
            tpr = v[1]
            roc_auc = v[2]

            if(len(self.dict[k]) >1):
                cnt = 0
                for l in self.dict[k]:
                    print('Summary for', l,'...',file=f)
                    for j in range(len(data_n)):
                        # print(data_n[j],file=f)    
                        mean_fpr = np.nanmean(
                            [fpr[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                        mean_tpr = np.nanmean(
                            [tpr[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                        mean_roc_auc = np.nanmean(
                            [roc_auc[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                        self._print_rates(mean_fpr, mean_tpr,mean_roc_auc,f)
                    cnt += 1
                print('micro average?',file=f)
                for j in range(len(data_n)):
                    # print(data_n[j],file=f)    
                    mean_fpr = np.nanmean(
                        [fpr[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                    mean_tpr = np.nanmean(
                        [tpr[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                    mean_roc_auc = np.nanmean(
                        [roc_auc[cv][j][cnt+i::len(models)] for cv in range(self.n_splits)])
                    self._print_rates(mean_fpr, mean_tpr,mean_roc_auc,f)
            else:
                for j in range(len(data_n)):
                    # print(data_n[j],file=f)    
                    mean_fpr = np.nanmean([fpr[cv][j][i]
                                        for cv in range(self.n_splits)])
                    mean_tpr = np.nanmean([tpr[cv][j][i]
                                        for cv in range(self.n_splits)])
                    mean_roc_auc = np.nanmean(
                        [roc_auc[cv][j][i] for cv in range(self.n_splits)])
                    self._print_rates(mean_fpr, mean_tpr,mean_roc_auc,f)

        for key in self.eval_dict.keys():
            self.wilcox_eval(key,i,f)

if(load_from_pkl):
    prot_ev = dill.load(open(filepath_prot,'rb'))
    import types  
    prot_ev.summary = types.MethodType(new_summary, prot_ev)
    prot_ev._print_rates = types.MethodType(n_print_rates,prot_ev)
    prot_ev.wilcox_eval = types.MethodType(n_wilcox_eval,prot_ev)

prot_ev.summary()
