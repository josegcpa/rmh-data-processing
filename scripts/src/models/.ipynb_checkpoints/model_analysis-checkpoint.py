import pandas as pd
import numpy as np
import mlflow
import functools
import sklearn.metrics as skm
from fairlearn.metrics import (MetricFrame, false_positive_rate, true_positive_rate, selection_rate, count)
from src.models.preprocess import CorrelatedColumnsRemover
from src.models.fetch_data import load_data
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold

def explain_cat_variable(var, X_test, shap_values):
    idx = pd.Index(X_test.columns).get_loc(var)
    epe_values = shap_values[:,idx]
    epe_data = X_test[var].reset_index(drop=True)
    unique_epe = X_test[var].unique()
    new_shap_values = [np.array(pd.Series(epe_values.values)[epe_data==cat]) for cat in unique_epe]
    
    #Each sublist needs to be the same length
    max_len = max([len(v) for v in new_shap_values])
    new_shap_values = [np.append(vs,[np.nan]*(max_len - len(vs))) for vs in new_shap_values]
    new_shap_values = np.array(new_shap_values)

    #transpost matrix so categories are columns and SHAP values are rows
    new_shap_values = new_shap_values.transpose()

    #replace shap values
    epe_values.values = np.array(new_shap_values)

    #replace data with placeholder array
    epe_values.data = np.array([[0]*len(unique_epe)]*max_len)

    #replace base data with placeholder array
    epe_values.base = np.array([0]*max_len)

    #replace feature names with category labels
    labels = [var+" ({})".format(u) for u in unique_epe]
    try:
        epe_values.feature_names = list(labels)
    except:
        print('error')
    
    return epe_values

def get_group_metrics(model_name, experiment_name, sen, sensitive_attribute, metrics, track_uri="../scripts/mlruns/" , X_train=None, y_train=None, X_test=None, y_test = None, uc=None, sf_test=None, pdt = 0.5, cv=False):
    
    # Load runs from mlflow
    mlflow.set_tracking_uri(track_uri)
    current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id = current_experiment['experiment_id']
    runs = mlflow.search_runs(experiment_id)
    run_id = runs.loc[runs['tags.mlflow.runName'].str.contains(model_name),'run_id'].values[0]

    # Load fitted model
    model_uri = track_uri + experiment_id + '/' + run_id + '/artifacts/model'
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # Load fitted preprocessing pipeline
    pipe_uri = track_uri + experiment_id + '/' + run_id + '/artifacts/preprocess_pipe'
    loaded_pipe = mlflow.sklearn.load_model(pipe_uri)

    # Load data
    data = model_name.split('_')[0]
    uc = int(model_name.split('_')[1][-1])
    sequence = model_name.split('_')[2]

    if ('SIEMENS' in model_name) or ('PHILIPS' in model_name): 
        scanner = model_name.split('_')[3]
    elif 'GE' in model_name:
        scanner = 'GE MEDICAL SYSTEMS'
    else:
        scanner = 'all'

    if 'noERC' in model_name:
        erc = False
    else:
        erc = True
    print('Loading...\tData:', data, '\tUC:', uc, '\tSequence:', sequence, '\tScanner:', scanner, '\tInclude ERC:', erc)

    X_train, y_train, X_test, y_test, descr = load_data(uc = uc,
                                                        dt = data,
                                                        sequence = sequence,
                                                        scanner = scanner,
                                                        erc = erc, 
                                                        return_descr = True, 
                                                        return_ids=True, new=True)
    if uc == 3 or uc == 6 or uc == 8:
        X_train, X_test = pd.concat([X_train, X_test]), pd.concat([X_train, X_test])
        y_train, y_test = pd.concat([y_train, y_test]), pd.concat([y_train, y_test])
        #X_train = pd.concat([X_train, X_test])
        #y_train = pd.concat([y_train, y_test])
    print('Full data set: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    # Process test data
    ids = X_test['PCa-ID']
    X_test = loaded_pipe.transform(X_test)
    X_test['PCa-ID'] = ids
    ids = X_train['PCa-ID']
    X_train = loaded_pipe.transform(X_train)
    X_train['PCa-ID'] = ids
    print('Preprocessed: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    #get sensitive attributes
    sen2 = sen.loc[sen['use_case_form'].str.contains(str(uc)),['PCa-ID', sensitive_attribute]]
    X_test['Target'] = y_test
    temp = X_test.loc[:,['PCa-ID', 'Target']].merge(sen2, on = 'PCa-ID')
    X_test = X_test.loc[X_test['PCa-ID'].isin(temp['PCa-ID']),:]
    y_test = X_test['Target']
    print('Sensitive attribute distribution:', end=' ')
    for x, y in zip(temp[sensitive_attribute].value_counts().index, temp[sensitive_attribute].value_counts().values):
        print(x, y, end='    ')
    print('\n')
    
    if cv:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
        folds = []
        for train_idx, test_idx in splitter.split(X_train, y_train):
            
            # cv split
            train_x = X_train.iloc[train_idx,:]
            train_y = y_train.reset_index(drop=True)[train_idx]
            val_x = X_train.iloc[test_idx,:]
            val_y = y_train.reset_index(drop=True)[test_idx]
            
            # refit on train folds
            loaded_model.fit(train_x.drop(columns=['PCa-ID']), train_y)
            
            # predict on validation folds
            try:
                y_prob = loaded_model.predict_proba(val_x.drop(columns=['PCa-ID']))
                y_pred = y_prob[:,1] > pdt
            except:
                y_pred = loaded_model.predict(val_x.drop(columns=['PCa-ID']))
            
            #get sensitive attributes
            sen2 = sen.loc[sen['use_case_form'].str.contains(str(uc)),['PCa-ID', sensitive_attribute]]
            val_x['Target'] = val_y
            temp = val_x.loc[:,['PCa-ID', 'Target']].merge(sen2, on = 'PCa-ID')
            
            # calculate performance
            grouped_metric = MetricFrame(metrics, val_y, y_pred,
                                         sensitive_features=temp[sensitive_attribute])
            folds.append(pd.DataFrame(grouped_metric.by_group))
        
        result = pd.concat([df.stack() for df in folds], axis=1).mean(axis=1).unstack()
        
        X_train['Target'] = y_train
        temp = X_train.loc[:,['PCa-ID', 'Target']].merge(sen2, on = 'PCa-ID', how='inner')
        temp = temp.loc[temp['PCa-ID'].isin(X_train['PCa-ID']),:].drop_duplicates(subset='PCa-ID')
        label_counts = pd.crosstab(temp[sensitive_attribute], temp['Target'])
        label_counts.columns = ['train_counts_target_'+str(x) for x in label_counts.columns]
        label_counts.reset_index(drop=False, inplace=True)
        result = result.merge(label_counts, on=sensitive_attribute, how='outer')
        
        temp2 = temp.loc[:,[sensitive_attribute]]
        c = pd.DataFrame(temp2.value_counts(), columns=['train_counts']).reset_index()
        result = result.merge(c, on=sensitive_attribute, how='outer')
            
    else:
    
        #get predictions on test set
        y_prob = loaded_model.predict_proba(X_test.drop(columns=['PCa-ID', 'Target']))
        y_pred = y_prob[:,1] > pdt
        
        #calculate performance
        grouped_metric = MetricFrame(metrics,
                                     y_test*1, y_pred*1,
                                     sensitive_features=temp[sensitive_attribute])

        a = pd.DataFrame(grouped_metric.by_group).reset_index()
        b = pd.DataFrame([temp[sensitive_attribute].value_counts().index, temp[sensitive_attribute].value_counts().values]).transpose()
        b.columns = [sensitive_attribute, 'test_counts']
        a = a.merge(b, on=sensitive_attribute, how='outer')
        label_counts = pd.crosstab(temp[sensitive_attribute], temp['Target'])
        label_counts.columns = ['test_counts_target_'+str(x) for x in label_counts.columns]
        label_counts.reset_index(drop=False, inplace=True)
        a = a.merge(label_counts, on=sensitive_attribute, how='outer')
        temp2 = sen2.loc[sen2['PCa-ID'].isin(X_train['PCa-ID']),:].drop_duplicates(subset='PCa-ID').loc[:,[sensitive_attribute]]
        c = pd.DataFrame(temp2.value_counts(), columns=['train_counts']).reset_index()
        result = a.merge(c, on=sensitive_attribute, how='outer')
        result.fillna(0, inplace=True)
    
    
    return result