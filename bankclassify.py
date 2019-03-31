#IMPORTS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#VARIABLES
original_file = "BankTransactions.csv"
training_file = "transactions_train_public.csv"
output_file = "output_file.csv"
replaceList = [' - Receipt', ' - Visa']
models = [GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, RandomForestClassifier]

#CLASS
class Classifier:
    def __init__(self, model = models[-1], report=True):
        self.model = model()
        self.report = report
        self.df_train, self.le = self._load_and_label_training_data(training_file)
        self.x_train, self.x_test, self.y_train, self.y_test = self._test_and_train_data(self.df_train)
        self.x_train_tfidf, self.y_train_le, self.tfidf = self._vectorize()
        self.clf = self._train_model()

    def _load_and_label_training_data(self, file, testSize=0.4, randomState=1):
        df = pd.read_csv(file)
        le = LabelEncoder()
        le.fit(df.y_train)
        return df, le

    def _test_and_train_data(self, df_train, testSize=0.2, randomState=1):
        x_train, x_test, y_train, y_test = train_test_split(df_train.x_train, df_train.y_train, test_size = testSize, random_state = randomState)
        return x_train, x_test, y_train, y_test

    def _vectorize(self):
        tfidf = TfidfVectorizer()
        x_train_tfidf = tfidf.fit_transform(self.x_train)
        y_train_le = self.le.transform(self.y_train)
        return x_train_tfidf, y_train_le, tfidf

    def _train_model(self):
        clf = self.model
        print(clf)
        clf.fit(self.x_train_tfidf.todense(), self.y_train_le)
        return clf

    def self_evaluate(self):
        x_train_tfidf, y_train_le = self.x_train_tfidf, self.y_train_le
        x_test_tfidf = self.tfidf.transform(self.x_test)
        y_test_le = self.le.transform(self.y_test)
        predicted = self.clf.predict(x_test_tfidf.todense())
        y_test_predicted = self.le.inverse_transform(predicted)
        if self.report == True:
            print(classification_report(self.y_test, y_test_predicted))
        print("Accuracy score is {:.1%}".format(accuracy_score(y_test_le, predicted)))
        #df=pd.DataFrame(model.le.transform(model.le.classes_))
        #df['label'] = model.le.classes_

    def classify(self, inputSeries):
        x_data = self.tfidf.transform(inputSeries)
        predictions = self.clf.predict(x_data.todense())
        y_test_output = self.le.inverse_transform(predictions)
        return y_test_output

    def load_and_clean_bank_data(self, file = original_file, replaceList = replaceList):
        df = pd.read_csv(file)
        #Creates a new column to store a clean transaction description in for the ML model to read
        for i in range(0,len(replaceList)):
            #Creates a new column if it's the first replacing item, otherwise uses the existing new list
            if i == 0:
                descriptionCol = 'Description'
            else:
                descriptionCol = 'x'
            #Cleaning description by removing everything after certain keywords which is just junk
            df['tempIndex'] = df[descriptionCol].str.find(replaceList[i])
            #Changing the end slice for rows where no change is needed to not modify the description later (otherwise it has value of -1)
            df['tempIndex'] = [1000 if df['tempIndex'][i]<0 else df['tempIndex'][i] for i in range(0,len(df))]
            #Make a new column by removing the junk text
            df['x'] = df.apply(lambda x: x[descriptionCol][0:x['tempIndex']], 1)
        del df['tempIndex']
        return df


#Running for real data

#SCRIPT
def main():
    myModel = Classifier()
    myModel.self_evaluate()

if __name__ == '__main__':
    main()