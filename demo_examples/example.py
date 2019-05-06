from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

features = [[7, 0],
            [11, 1],
            [9, 1],
            [10, 1]
            ]

feature_names = ["diameter", "colour"]

labels = [0, 0, 1, 1]

label_names = ["Apple", "Orange"]

classifier = DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

with open("example.txt", "w") as f:
    f = export_graphviz(classifier,
    					out_file=f,
    					feature_names=feature_names,
    					class_names=label_names,
    					filled=True,
    					rounded=True,
    					special_characters=True
    				   )

# dot -Tpdf example.txt -o example.pdf