from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()

x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,
                                                 random_state=2,test_size=0.2)


tpot = TPOTClassifier(verbosity=2,generations=5,
                      population_size=20)





tpot.fit(x_train,y_train)


print(tpot.score(x_test,y_test))

tpot.export("mydigits.py")


# RESAULT
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9798199767711961
exported_pipeline = MLPClassifier(alpha=0.1, learning_rate_init=0.01)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
