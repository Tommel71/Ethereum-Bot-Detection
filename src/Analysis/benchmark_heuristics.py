import pandas as pd

def metric(prediction: pd.DataFrame, target: pd.DataFrame):

    return (prediction == target).mean()

def accuracy_prediction(prediction: pd.DataFrame, target: pd.DataFrame):

    assert (x == y for x,y in zip(prediction.index, target.index))

    best_score = metric(prediction.values, target.values)

    return best_score
