import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from gan_thesis.evaluation.machine_learning import *
from gan_thesis.data.load_data import load_data

N_ESTIM = 10

MODELS = {'logreg': LogisticRegression(penalty='none', max_iter=100),
          'rforest': RandomForestClassifier(n_estimators=N_ESTIM),
          'gboost': AdaBoostClassifier(n_estimators=N_ESTIM)}

MODELS = {'logreg': LogisticRegression(max_iter=100)}


def pMSE_ratio(real_df, synth_df, discrete_columns):
    # Extract all discrete features, first line ensures that get all features from both real and synth
    features = real_df.columns.to_list() + list(set(synth_df.columns.to_list()) - set(real_df.columns.to_list()))
    discrete_used_feature_indices = [feature for feature in features if feature in discrete_columns]

    one_hot_real = pd.get_dummies(real_df, columns=discrete_used_feature_indices)
    one_hot_synth = pd.get_dummies(synth_df, columns=discrete_used_feature_indices)

    ratio = {}
    for model in MODELS:
        pmse, k = pMSE(one_hot_real, one_hot_synth, MODELS[model])
        if model == 'logreg':
            N = len(real_df.index) + len(synth_df.index)
            c = len(synth_df.index)/N
            null = (k-1) * (1-c)**2 * (c/N)
        else:
            null = null_pmse(model)
        ratio[model] = pmse/null
        print('pmse: ', pmse)
        print('null: ', null)
        print('ratio: ', ratio)


    return ratio


def pMSE(real_df, synth_df, model, shuffle=False, polynomials=True):
    # This should be implemented with multiple models chosen by some variable
    # For now it will be done with logistic regression as there is an analytical solutionn to the null value.
    # ind_var is the name of the indicator variable

    df = add_indicator(real_df, synth_df, shuffle)
    predictors = df.iloc[:, :-1]
    target = df.iloc[:, -1]

    if polynomials:
        poly = PolynomialFeatures(degree=2)
        poly.fit_transform(predictors)

    model.fit(predictors, target)
    prediction = model.predict_proba(predictors)
    c = len(synth_df.index) / len(df.index)
    pmse = sum((prediction[:, 1] - c) ** 2) / len(df.index)

    return pmse, model.coef_.size


def null_pmse_est(real_df, synth_df, n_iter):
    # Randomly assigns the indicator variable
    null = {}
    for model in MODELS:
        pmse = 0
        for i in range(n_iter):
            if i % 10 == 0: print('iteration {0}'.format(str(i)))
            pmse += pMSE(real_df, synth_df, MODELS[model], shuffle=True)
        pmse /= n_iter
        null[model] = pmse
    print(null)
    df = pd.DataFrame(data=null, index=range(1))
    save_path = os.path.join(os.path.dirname(__file__), 'null_pmse_est_{0}_{1}.csv'.format(str(N_ESTIM), str(n_iter)))
    df.to_csv(save_path, index=False)
    return pmse


def null_pmse(model):
    load_path = os.path.join(os.path.dirname(__file__), 'null_pmse_est.csv')
    df = pd.read_csv(load_path)
    print(df[model][0])

    return df[model][0]


def add_indicator(real_df, synth_df, shuffle=False):
    """ Helper function which combines real data with synthetic and adds a corresponding indicator feature.
    :param shuffle: If True. shuffle entire dataset after adding indicator (used when calculating null-pMSE)."""
    r_df = real_df.copy()
    r_df['ind'] = 0
    s_df = synth_df.copy()
    s_df['ind'] = 1
    df = pd.concat((r_df, s_df), axis=0).reset_index(drop=True)

    if shuffle:
        df['ind'] = df['ind'].sample(frac=1).reset_index(drop=True)

    return df


if __name__ == '__main__':
    # This should be a dataset in which samples exists
    dataset = load_data('mvn-test2')
    real = dataset.train
    samples = dataset.samples.get('ctgan')
    null_pmse_est(real, samples, n_iter=100)
    #N_ESTIM = 100
    #null_pmse_est(real, samples, n_iter=1000)
    # Our trials show that these give similar null_pmse estimates
