# Progetto-Intelligenza-Artificiale
Studio classificatori Naive Bayes

#####   SYSTEM REQUIREMENTS  #####
Per eseguire al meglio il programma è necessario:
1. Scaricare il dataset di training da http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
2. Scaricare il dataset di test da  http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz
3. Inserire i dataset nella cartella 'resources' che dovrà essere presente nella directory di esecuzione del file 'main.py'

##### RUNNING SIMPLE #####
Il programma è totalmente autosufficiente; una volta lanciato produrrà i risultati, che si troveranno nella cartela 'results' nella directory di esecuzione del file 'main.py'.
Eventuali messaggi di warning sono dovuti alla struttura stessa del dataset KDD99 e non pregiudicano l'esito dei test.

##### main.py #####
Si articola in tre fasi: ogni fase testa un modello di classificatore bayesiano diverso e salva i risultati nella cartella 'results'.

##### resources/preprocessing.py #####
File contenente tutte le funzioni di pre-processazione per i dataset: oneHotEncoding, scaling, ecc...
La funzione 'prepare()' prende in input i dataset,di training e di test, e il modello che si vuole utilizzare per la predizione, ed in base a quest'ultimo, restituisce dei dataset trasformati in modo ottimale per la predizione con il modello scelto.

