from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def transferability_score(y_source, y_target):
    '''
    Args: 
      y_source (ndarray): extracted features of target samples, pseudo source labels or source labels (N x D1)
      y_target (ndarray): ground truth labels (N x D2)
    Return: 
      mse: a scalar for transferability score 
    '''
    linear_reg = LinearRegression().fit(y_source, y_target)
    y_pred = linear_reg.predict(y_source)
    mse = -1.0 * mean_squared_error(y_target, y_pred, squared=True)
    return mse

def transferability_score_ridge(y_source, y_target, alpha, solver='svd'):
    '''
    Args: 
      y_source (ndarray): extracted features of target samples, pseudo source labels or source labels (N x D1)
      y_target (ndarray): ground truth labels (N x D2)
    Return: 
      mse: a scalar for transferability score 
    '''
    n = y_source.shape[0]
    ridge = Ridge(alpha=n * alpha,solver=solver).fit(y_source, y_target)
    y_pred = ridge.predict(y_source)
    mse = -1.0 * mean_squared_error(y_target, y_pred, squared=True)
    return mse
