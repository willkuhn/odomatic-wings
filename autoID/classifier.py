
def setup_classifier(clfMethod):
    """Set up a classifier"""
    # Random Forest, calibrated & optimized
    if clfMethod == 'rf':
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier
        return CalibratedClassifierCV(RandomForestClassifier(oob_score=True,
              max_features=0.1,n_estimators=90,n_jobs=-1),method='sigmoid')

    # Naive Bayes, calibrated, can't be optimized
    elif clfMethod == 'nb':
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.naive_bayes import GaussianNB
        return CalibratedClassifierCV(GaussianNB(),method='sigmoid')

    # Support Vector Machine with Radial Basis Function kernel, optimized for genus
    elif clfMethod == 'svm_rbf_g':
        from sklearn.svm import SVC
        return SVC(kernel='rbf', C=1e6, gamma=1e-05, probability=True)

    # SVM with RBF kernel, optimized for genus+species or genus+species+sex
    elif clfMethod == 'svm_rbf_gs':
        from sklearn.svm import SVC
        return SVC(kernel='rbf',C=100., gamma=0.1,probability=True)

    # SVM with linear kernel, optimized
    elif clfMethod == 'svm_lin':
        from sklearn.svm import SVC
        return SVC(kernel='linear',probability=True,C=10.0,gamma=1.0)

    # Bernoulli RBM NeuralNetwork + Logistic Regression, calibrated & optimized
    elif clfMethod == 'nn_log':
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline
        from sklearn.neural_network import BernoulliRBM
        from sklearn.linear_model import LogisticRegression
        return CalibratedClassifierCV(Pipeline(steps=[('rbm', BernoulliRBM(
              n_components=256,batch_size=30,learning_rate=0.01,n_iter=5)),
              ('logistic',LogisticRegression(C=1e5))]),method='sigmoid')

    # Bernoulli RBM NeuralNetwork + Linear SVM
    elif clfMethod == 'nn_svm':
        from sklearn.pipeline import Pipeline
        from sklearn.neural_network import BernoulliRBM
        from sklearn.svm import SVC
        return Pipeline(steps=[('rbm', BernoulliRBM(n_components=256,
              batch_size=30,learning_rate=0.01,n_iter=5)),
              ('linsvm', SVC(kernel='linear',probability='True',C=10.0,gamma=1.0))])

    else:
        raise ValueError('Value for `clfMethod` not recognized.')
