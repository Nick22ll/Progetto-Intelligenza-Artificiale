import pandas
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import scikitplot as skplot
from resources import preprocessing

results = open("resources/results/results.txt", "w")
#Test con Bernoulli Naive Bayes
results.write("RISULTATI NAIVE BAYES BERNOULLI\n")
for i in ["PRIMA", "DOPO"]:
    X_train, y_train, X_test, y_test = preprocessing.prepare(train_path="resources/kddcup.data_10_percent.gz", test_path="resources/corrected.gz")

    if i == "PRIMA":
        y_train = preprocessing.target_to_five(y_train)
        y_test = preprocessing.target_to_five(y_test)
        esempi = y_train.value_counts(normalize= True)
        results.write(f"Esempi per classe presenti nel dataset di training:\n{esempi*100}\n")
        esempi = y_test.value_counts(normalize=True)
        results.write(f"Esempi per classe presenti nel dataset di testing:\n{esempi * 100}\n")


    #Bernoulli Naive Bayes
    model = BernoulliNB()
    model.fit(X_train, y_train)
    results.write(f"Numero classi:{model.classes_}\nNumero features: {model.n_features_}\nNumero di esempi:{X_train.shape[0]}\n")
    results_train = pandas.Series(model.predict(X_train), name = 'target')
    results_test =pandas.Series(model.predict(X_test), name = 'target')

    if i == "DOPO":
        y_train= preprocessing.target_to_five(y_train)
        y_test = preprocessing.target_to_five(y_test)
        results_train = preprocessing.target_to_five(results_train)
        results_test = preprocessing.target_to_five(results_test)

    accuracy_train = accuracy_score(y_train, results_train)
    accuracy_test = accuracy_score(y_test, results_test)

    results.write(f"\nRisultati con suddivisione delle classi di attacco {i} la predizione:\n")
    results.write(f"\nAccuratezza/Misclassification Training: {accuracy_train,1-accuracy_train}\nAccuratezza/Misclassification Test: {accuracy_test,1-accuracy_test}\n")
    skplot.metrics.plot_confusion_matrix(y_test, results_test, normalize= "all")
    plt.savefig("resources/results/BernoulliConfusionMatrix"+i)


#Test con Multinomial Naive Bayes
results.write("\n\n\nRISULTATI NAIVE BAYES MULTINOMIALE\n")
for i in ["PRIMA", "DOPO"]:
    X_train, y_train, X_test, y_test = preprocessing.prepare(train_path="resources/kddcup.data_10_percent.gz", test_path="resources/corrected.gz", model = "MULTINOMIAL")

    if i == "PRIMA":
        y_train = preprocessing.target_to_five(y_train)
        y_test = preprocessing.target_to_five(y_test)
        esempi = y_train.value_counts(normalize= True)
        results.write(f"Esempi per classe presenti nel dataset di training:\n{esempi*100}\n")
        esempi = y_test.value_counts(normalize=True)
        results.write(f"Esempi per classe presenti nel dataset di testing:\n{esempi * 100}\n")

    #Multinomial Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)
    results.write(f"Numero classi:{model.classes_}\nNumero features: {model.n_features_}\nNumero di esempi:{X_train.shape[0]}\n")
    results_train = pandas.Series(model.predict(X_train), name = 'target')
    results_test =pandas.Series(model.predict(X_test), name = 'target')

    if i == "DOPO":
        y_train= preprocessing.target_to_five(y_train)
        y_test = preprocessing.target_to_five(y_test)
        results_train = preprocessing.target_to_five(results_train)
        results_test = preprocessing.target_to_five(results_test)


    accuracy_train = accuracy_score(y_train, results_train)
    accuracy_test = accuracy_score(y_test, results_test)
    results.write(f"\nRisultati con suddivisione delle classi di attacco {i} la predizione:\n")
    results.write(f"\nAccuratezza/Misclassification Training: {accuracy_train,1-accuracy_train}\nAccuratezza/Misclassification Test: {accuracy_test,1-accuracy_test}\n")
    #skplot.metrics.plot_confusion_matrix(y_test, results_test, normalize="all")
    skplot.metrics.plot_confusion_matrix(y_test, results_test, normalize= 'all')
    plt.savefig("resources/results/MultinomialConfusionMatrix" + i)

#Test con Gaussian Naive Bayes
results.write("\n\n\nRISULTATI NAIVE BAYES GAUSSIANO\n")
for i in ["PRIMA", "DOPO"]:
    X_train, y_train, X_test, y_test = preprocessing.prepare(train_path="resources/kddcup.data_10_percent.gz", test_path="resources/corrected.gz", model = "GAUSSIAN")

    if i == "PRIMA":
        y_train = preprocessing.target_to_five(y_train)
        y_test = preprocessing.target_to_five(y_test)
        esempi = y_train.value_counts(normalize= True)
        results.write(f"Esempi per classe presenti nel dataset di training:\n{esempi*100}\n")
        esempi = y_test.value_counts(normalize=True)
        results.write(f"Esempi per classe presenti nel dataset di testing:\n{esempi * 100}\n")

    #Gaussian Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    results.write(f"Numero classi:{model.classes_}\nNumero features: {model.sigma_.shape[1]}\nNumero di esempi:{X_train.shape[0]}\n")
    results_train = pandas.Series(model.predict(X_train), name = 'target')
    results_test =pandas.Series(model.predict(X_test), name = 'target')

    if i == "DOPO":
        y_train= preprocessing.target_to_five(y_train)
        y_test = preprocessing.target_to_five(y_test)
        results_train = preprocessing.target_to_five(results_train)
        results_test = preprocessing.target_to_five(results_test)


    accuracy_train = accuracy_score(y_train, results_train)
    accuracy_test = accuracy_score(y_test, results_test)
    results.write(f"\nRisultati con suddivisione delle classi di attacco {i} la predizione:\n")
    results.write(f"\nAccuratezza/Misclassification Training: {accuracy_train,1-accuracy_train}\nAccuratezza/Misclassification Test: {accuracy_test,1-accuracy_test}\n")
    skplot.metrics.plot_confusion_matrix(y_test, results_test, normalize="all")
    skplot.metrics.plot_confusion_matrix(y_test, results_test, normalize= 'all')
    plt.savefig("resources/results/GaussianConfusionMatrix" + i)
