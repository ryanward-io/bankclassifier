# IMPORTS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import (
                                 GaussianNB,
                                 MultinomialNB,
                                 ComplementNB,
                                 BernoulliNB
                                )
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# VARIABLES

original_file = "BankTransactions.csv"
training_file = "transactions_train.csv"
output_file = "output_file.csv"
replaceList = [' - Receipt', ' - Visa']

vectorizers = [
			  	CountVectorizer,
				TfidfVectorizer
			 ]

models = [
            GaussianNB,
            MultinomialNB,
            ComplementNB,
            BernoulliNB,
            RandomForestClassifier,
            DecisionTreeClassifier
         ]

# CLASS
class Classifier:

    def __init__(self, model=models[-1], vectorizer=vectorizers[0]):
        if model is RandomForestClassifier:
            self.model = model(n_estimators=100)
        else:
            self.model = model()

        self._model_str = str(self.model)
        self.model_name = self._model_str[0:self._model_str.find("(")]
        self.vectorizer = vectorizer
        self.df_train, self.le = self._load_and_label_train_data(training_file)
        self.x_train, self.x_test, self.y_train, self.y_test = self._test_and_train_data(self.df_train)
        self.x_train_tfidf, self.y_train_le, self.tfidf = self._vectorize()
        self.clf = self._train_model()

    def _load_and_label_train_data(self, file, testSize=0.4, randomState=1):
        df = pd.read_csv(file)
        le = LabelEncoder()
        le.fit(df.y_train)
        return df, le

    def _test_and_train_data(self, df_train, testSize=0.2, randomState=1):
        x_train, x_test, y_train, y_test = train_test_split(
                                                df_train.x_train,
                                                df_train.y_train,
                                                test_size=testSize,
                                                random_state=randomState
                                            )
        return x_train, x_test, y_train, y_test

    def _vectorize(self):
        tfidf = self.vectorizer()
        x_train_tfidf = tfidf.fit_transform(self.x_train)
        y_train_le = self.le.transform(self.y_train)
        return x_train_tfidf, y_train_le, tfidf

    def _train_model(self):
        clf = self.model
        clf.fit(self.x_train_tfidf.todense(), self.y_train_le)
        return clf

    def self_evaluate(self, print_output=True, return_output=False):
        x_train_tfidf, y_train_le = self.x_train_tfidf, self.y_train_le
        x_test_tfidf = self.tfidf.transform(self.x_test)
        y_test_le = self.le.transform(self.y_test)
        # Next few lines are calculating values to print out in the result
        predicted = self.clf.predict(x_test_tfidf.todense())
        accuracy_pc = accuracy_score(y_test_le, predicted)
        y_test_predicted = self.le.inverse_transform(predicted)
        if print_output is True:
            print("Accuracy of {} is {:.1%}".format(
                                                    self.model_name,
                                                    accuracy_pc
                                                   ))
            print(classification_report(self.y_test, y_test_predicted))
        if return_output is True:
            return classification_report(
                                         self.y_test,
                                         y_test_predicted,
                                         output_dict=True
                                        )

    def load_and_clean_bank_data(
                                 self,
                                 file=original_file,
                                 replaceList=replaceList
                                ):
        df = pd.read_csv(file)
        # Creates a new col to store clean transaction info for the model
        for i in range(0, len(replaceList)):
            # CrNeed to create a new column or replace existing new column
            if i == 0:
                descriptionCol = 'Description'
            else:
                descriptionCol = 'x'
            # Cleaning description by removing junk
            df['tempIndex'] = df[descriptionCol].str.find(replaceList[i])
            # Amending end slice for rows if no change needed (as they are -1)
            df['tempIndex'] = [1000 if df['tempIndex'][i] < 0
                               else df['tempIndex'][i]
                               for i in range(0, len(df))
                               ]
            # Make a new column by removing the junk text
            df['x'] = df.apply(lambda
                               x: x[descriptionCol][0:x['tempIndex']],
                               1)
        del df['tempIndex']
        return df

    def classify(self, inputSeries):
        x_data = self.tfidf.transform(inputSeries)
        predictions = self.clf.predict(x_data.todense())
        y_test_output = self.le.inverse_transform(predictions)
        return y_test_output


# FUNCTION

def compare_models():
    df_comparison = pd.DataFrame()
    for i in models:
        mod = Classifier(model=i)
        precision = mod.self_evaluate(print_output=False, return_output=True)
        df_temp = pd.DataFrame.from_dict(precision, orient="index")
        df_comparison[mod.model_name] = df_temp['precision']
    return df_comparison


def model_heatmap():
    df = compare_models()
    sns.heatmap(df, annot=True)
    plt.show()

# SCRIPT

def main():
    myModel = Classifier()
    model_heatmap()

if __name__ == '__main__':
    main()