import numpy as np
import pandas as pd
from scipy import stats, spatial
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold

class CorrelatedColumnsRemover(BaseEstimator, TransformerMixin):
    """
    A class used to represent a remover of correlated 
    columns from a dataframe.

    Methods
    -------
    fit(self, X, y = None)
        Calculates the correlation matrix and finds which 
        columns have correlation above the threshold.
    transform(self, X, y = None)
        Removes correlated columns from the dataframe.
    """
    
    def __init__(self,
                 threshold:float=0.8,
                 corr:str="pearson"):
        """
        Parameters
        ----------
        threshold : float
            Value above which two features are 
            considered correlated.
        corr : str
            Type of correlation to calculate. 
            Options: pearson, spearman or kendall.
        """
        
        self.threshold = threshold
        self.corr = corr
        
    def fit(self,X,y=None):
        """Calculates the correlation matrix and finds which 
        columns have correlation above the threshold.

        Parameters
        ----------
        X : pandas dataframe
            dataframe with data
        y : pandas series, optional
            labels
        """
        
        self.n_samples_,self.n_features_ = X.shape
        if self.corr == "pearson":
            r = np.corrcoef(X.T)
            n = self.n_features_
            t = r*np.sqrt((n-2)/(1-r*r+1e-8))
            corr_mat_values = np.abs(r)
            corr_mat_pvalues = 1 - stats.t.cdf(t, n-2)

        elif self.corr == "spearman":
            corr_mat = stats.spearmanr(X, axis=0)
            corr_mat_values,corr_mat_pvalues = corr_mat
            corr_mat_values = np.abs(corr_mat_values)

        elif self.corr == "kendall":
            corr_mat = stats.kendalltau(X)
            corr_mat_values = np.abs(corr_mat.statistic)
            corr_mat_pvalues = corr_mat.pvalue

        else:
            raise 'corr must be one of ["pearson","spearman","kendall"]'
            
        highly_correlated = np.zeros_like(corr_mat_values,
                                          dtype=bool)
        tri = np.triu_indices(self.n_features_,1)
        highly_correlated[tri] = corr_mat_values[tri] > self.threshold
                
        columns_to_remove = []
        if np.any(highly_correlated):
            for x,y in zip(*np.where(highly_correlated)):
                # remove the column which are more often correlated 
                # with other columns
                x_mean = np.nanmean(corr_mat_values[x])
                y_mean = np.nanmean(corr_mat_values[y])
                if x_mean > y_mean:
                    columns_to_remove.append(x)
                else:
                    columns_to_remove.append(y)
        
        self.columns_to_remove_ = list(set(columns_to_remove))
        return self
    
    def transform(self,X):
        """Removes correlated columns from the dataframe.

        Parameters
        ----------
        X : pandas dataframe
            dataframe with data
        y : pandas series, optional
            labels
        """
        
        new = X.drop(X.columns[self.columns_to_remove_], axis=1)
        return new
    
    def get_feature_names_out(self):
        pass
    
class Booleanizer(BaseEstimator, TransformerMixin):
    """
    A class used to represent a transformer that forces 
    any column with 2 unique values to bool.

    Methods
    -------
    fit(self, X, y = None)
        Finds which columns have 2 unique values.
    transform(self, X, y = None)
        Sets those columns to bool.
    """
    
    def __init__(self, columns = None):
        """
        Parameters
        ----------
        columns : list, optional
            Columns with 2 unique values.
        """
        
        self.to_bool = columns
        
    def fit(self,X,y=None):
        """Finds which columns have 2 unique values.

        Parameters
        ----------
        X : pandas dataframe
            dataframe with data
        y : pandas series, optional
            labels
        """
        
        self.to_bool = [col for col in X.columns if len(X[col].unique())==2]
        
        return self
    
    def transform(self,X):
        """Sets columns with  2 unique values to bool.

        Parameters
        ----------
        X : pandas dataframe
            dataframe with data
        y : pandas series, optional
            labels
        """
        
        for col in self.to_bool:
            X[col] = X[col].astype(bool)
        return X

    def get_feature_names_out(self):
        pass
    
class NumericColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A class used to represent a transformer that separates
    columns by data type and processes only the numeric
    variables with: 
        StandardScaler(), 
        VarianceThreshold(0.01), 
        CorrelatedColumnsRemover(0.9, 'spearman')).

    Methods
    -------
    fit(self, X, y = None)
        .....
    transform(self, X, y = None)
        .....
    """
    
    def __init__(self, pipe = None):
        """
        Parameters
        ----------
        pipe : sklearn pipeline, optional
            Preprocessing pipeline.
        """
        
        self.preprocess = pipe
        
    def fit(self,X,y=None):
        """Fits preprocessing pipeline.

        Parameters
        ----------
        X : pandas dataframe
            dataframe with data
        y : pandas series, optional
            labels
        """
        
        if self.preprocess is None:
        
            numeric_features = X.select_dtypes(include=[int, float]).columns
            numeric_transformer = make_pipeline(StandardScaler(), 
                                                VarianceThreshold(0.01), 
                                                CorrelatedColumnsRemover(0.9, 'spearman'))

            boolean_features = X.select_dtypes(include=[bool]).columns
            boolean_transformer = OrdinalEncoder()

            categorical_features = X.select_dtypes(include=['category']).columns
            categorical_transformer = 'passthrough'

            self.preprocess = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),
                                                           ("bool", boolean_transformer, boolean_features), 
                                                           ("cat", categorical_transformer, categorical_features)], 
                                                           remainder='passthrough')
            self.preprocess.set_output(transform="pandas")
        
        self.preprocess.fit(X, y)
        
        return self
    
    def transform(self,X):
        """...... .

        Parameters
        ----------
        X : pandas dataframe
            dataframe with data
        y : pandas series, optional
            labels
        """
        
        return self.preprocess.transform(X)

    def get_feature_names_out(self):
        pass