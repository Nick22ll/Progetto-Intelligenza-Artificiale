
import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.feature_selection import  f_classif, SelectPercentile



def prepare(train_path, test_path, name_path = "resources/names.txt", model = "BERNOULLI"):
    train, test = exctract(train_path, test_path, name_path)

    if model == "MULTINOMIAL":
        train, test = best_features(train, test, 60)
        train, test = low_correlation(train, test, 0.90)
        train, test, discretized = discretize(train, test)
        X_train, y_train = dataset_splitter(train)
        X_test, y_test = dataset_splitter(test)
        X_train, X_test = myTransformer(X_train, X_test, to_convert=feature_names()[1]['symbolic'])
    elif model == "GAUSSIAN":
        train, test = best_features(train, test, 55)
        train, test = low_correlation(train, test, 0.65)
        X_train, y_train = dataset_splitter(train)
        X_test, y_test = dataset_splitter(test)
        X_train, X_test = myTransformer(X_train, X_test, to_convert=feature_names()[1]['symbolic'])
    else:
        train, test = best_features(train, test, 60)
        train, test = low_correlation(train, test, 0.85)
        train, test = oneHot(train, test, to_convert = ['protocol_type', 'service', 'flag'])
        train, test, discretized = discretize(train, test)
        train, test = oneHot(train, test, to_convert = discretized, numeric=True)
        X_train, y_train = dataset_splitter(train)
        X_test, y_test = dataset_splitter(test)
    return X_train, y_train, X_test, y_test


#Funzione che permette l'estrazione dei dati da file .csv o .txt restituendoli come pandas.Dataframe
def exctract(train_path, test_path, name_path = 'resources/names.txt'):
    names = feature_names(name_path)[0]
    # reading the datas from a text file
    training_dataset = pandas.read_csv(train_path, delimiter=',', header=None)
    test_dataset = pandas.read_csv(test_path, delimiter=',', header=None)
    # giving a name to the columns to read the dataframe easily
    training_dataset.set_axis(names, axis='columns', inplace=True)
    test_dataset.set_axis(names, axis='columns', inplace=True)
    return training_dataset, test_dataset



#Funzione che separa il dataset in 2 parti: le features e il target
def dataset_splitter(dataset):
    target = dataset['target']
    data = dataset.drop('target', axis = 'columns')
    return data, target


#Estrae i nomi e i tipi delle features da un percorso file dato in ingresso: i nomi sono restituiti in una lista, i tipi in un dizionario
def feature_names(path = 'resources/names.txt'):
    features_type = {'symbolic': [], 'numeric':[]}
    with open(path, 'r') as names_file:
        lines = names_file.readlines()
    lines.pop(0)
    for i in range(len(lines)):
        if lines[i][lines[i].find(':')+2:lines[i].find('.')] == 'symbolic':
            features_type['symbolic'].append(lines[i][0:lines[i].find(':')])
        else:
            features_type['numeric'].append(lines[i][0:lines[i].find(':')])
        lines[i] = lines[i][0:lines[i].find(':')]
    features_type['symbolic'].append('target')
    lines.append('target')
    return lines, features_type


#Funzione che scala features continue in modo che abbiano distribuzione Normale( N(0,1) )
def scaling(train_data, test_data):
    scaler = StandardScaler(with_mean = 0, with_std = 1)
    #scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data



#Funzione che discretizza dati numerici continui presenti nei dataset
def discretize(train_data, test_data):
    to_discretize = []
    for el in train_data.columns:
        if el in feature_names()[1]['numeric']:
            to_discretize.append(el)
    for col in to_discretize:
        if len(train_data[col].value_counts()) <= 2:
            to_discretize.remove(col)
    discretizer = KBinsDiscretizer(n_bins= 24, encode= 'ordinal', strategy= 'uniform')
    train_data[to_discretize] = discretizer.fit_transform(train_data[to_discretize])
    test_data[to_discretize] = discretizer.transform(test_data[to_discretize])
    return train_data, test_data, to_discretize


#Funzione che implementa la trasformazione OneHot delle features indicate nel parametro 'to_convert'
def oneHot(train_data, test_data, to_convert, numeric = False):
    transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
    for el in to_convert:
        if el not in train_data.columns:
            to_convert.remove(el)
    if len(to_convert) == 0:
        return train_data, test_data
    transformer.fit(train_data[to_convert])
    new_features = []
    if numeric:
        for el in range(len(to_convert)):
            lista = []
            for i in transformer.categories_[el]:
                #creo la lista contenente i nomi della feature in posizione el per ogni categoria incontrata: "nome_feature"+"nome_categoria"
                lista.append(to_convert[el]+'_'+str(i))
            #sostituisco al nome della feature originale una lista di nomi "one-code"
            new_features = new_features+lista
    else:
        for e in transformer.categories_:
            for f in e:
                new_features.append(f)
    new_dataframe = pandas.DataFrame(transformer.transform(train_data[to_convert]), columns=new_features, dtype=bool)
    train_data = train_data.join(new_dataframe)
    train_data.drop(to_convert, axis='columns', inplace=True)
    new_dataframe = pandas.DataFrame(transformer.transform(test_data[to_convert]), columns=new_features, dtype= bool)
    test_data = test_data.join(new_dataframe)
    test_data.drop(to_convert, axis='columns', inplace=True)
    return train_data, test_data

#Funzione che trasforma dati categoriali in dati numerici ordinati, gestendo anche il ritrovamento nei dati di test di categorie non
# trovate nei dati di training
def myTransformer(train, test, to_convert):
    for el in to_convert:
        if el not in train.columns:
            to_convert.remove(el)
    if 'is_guest_login' in to_convert:
        to_convert.remove('is_guest_login')
    for feature in to_convert:
        categories = list(train[feature].unique())
        #creazione lista delle categorie anomale nel dataset di test
        test_anomalies = list(test[feature].unique())
        #rimozione delle categorie "legali"
        for el in categories:
            if el in test_anomalies:
                test_anomalies.remove(el)
        #creazione dizionari per la codifica ordinale all'interno dei dataset
        dictionary = {}
        anomalies_dictionary = {}
        #riempimanto dizionari delle categorie "legali"
        for el in range(len(categories)):
            dictionary[categories[el]] = el
        #riempimanto del dizionario delle anomalie: verranno raggruppate in un unica categoria (-666)
        for el in range(len(test_anomalies)):
            anomalies_dictionary[test_anomalies[el]] = -666

        #sostituzione dei valori all'interno del dataset
        train.replace({feature: dictionary}, inplace= True)
        test.replace({feature: dictionary}, inplace= True)
        test.replace({feature: anomalies_dictionary}, inplace= True)
    return train, test


#Funzione che mantiene le features con una correlazione al di sotto di una certa soglia
def low_correlation(train, test, perc):
    corr_matrix = train.corr()
    for row in range(len(corr_matrix.index)):
        for col in range(row+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iat[row,col])>= perc:
                if corr_matrix.columns[col] in train.columns:
                    train.drop(corr_matrix.columns[col], axis= 'columns', inplace= True)
                    test.drop(corr_matrix.columns[col], axis= 'columns', inplace= True)
    return train, test


def target_to_five(dataset):
    if type(dataset) == pandas.Series:
        dataset.replace(
            {'normal.':'normal','neptune.': 'DoS', 'smurf.': 'DoS', 'pod.': 'DoS', 'teardrop.': 'DoS', 'land.': 'DoS', 'back.': 'DoS',
             'apache2.': 'DoS', 'udpstorm.': 'DoS', 'processtable.': 'DoS', 'mailbomb.': 'DoS',
             'buffer_overflow.': 'U2R', 'loadmodule.': 'U2R', 'perl.': 'U2R', 'rootkit.': 'U2R', 'spy.': 'U2R',
             'xterm.': 'U2R', 'ps.': 'U2R', 'httptunnel.': 'U2R', 'sqlattack.': 'U2R', 'worm.': 'U2R',
             'snmpguess.': 'U2R', 'guess_passwd.': 'R2L', 'ftp_write.': 'R2L', 'phf.': 'R2L', 'imap.': 'R2L',
             'multihop.': 'R2L', 'warezmaster.': 'R2L', 'warezclient.': 'R2L', 'snmpgetattack.': 'R2L', 'named.': 'R2L',
             'xlock.': 'R2L', 'xsnoop.': 'R2L', 'sendmail.': 'R2L', 'portsweep.': 'Probe', 'ipsweep.': 'Probe',
             'satan.': 'Probe', 'nmap.': 'Probe', 'saint.': 'Probe', 'mscan.': 'Probe'}, inplace=True)

    if type(dataset) == pandas.DataFrame:
        dataset.replace({'target': {'normal.':'normal','neptune.': 'DoS', 'smurf.': 'DoS', 'pod.': 'DoS', 'teardrop.': 'DoS',
                                    'land.': 'DoS', 'back.': 'DoS', 'apache2.': 'DoS', 'udpstorm.': 'DoS',
                                    'processtable.': 'DoS', 'mailbomb.': 'DoS', 'buffer_overflow.': 'U2R',
                                    'loadmodule.': 'U2R', 'perl.': 'U2R', 'rootkit.': 'U2R', 'spy.': 'U2R',
                                    'xterm.': 'U2R', 'ps.': 'U2R', 'httptunnel.': 'U2R', 'sqlattack.': 'U2R',
                                    'worm.': 'U2R', 'snmpguess.': 'U2R', 'guess_passwd.': 'R2L', 'ftp_write.': 'R2L',
                                    'phf.': 'R2L', 'imap.': 'R2L', 'multihop.': 'R2L', 'warezmaster.': 'R2L',
                                    'warezclient.': 'R2L', 'snmpgetattack.': 'R2L', 'named.': 'R2L', 'xlock.': 'R2L',
                                    'xsnoop.': 'R2L', 'sendmail.': 'R2L', 'portsweep.': 'Probe', 'ipsweep.': 'Probe',
                                    'satan.': 'Probe', 'nmap.': 'Probe', 'saint.': 'Probe', 'mscan.': 'Probe'}},
                        inplace=True)
    return dataset



#Funzione che elimina le righe duplicate in un dataset
def duplicate_elimination(dataset):
    dataset = dataset[dataset.duplicated() == False]
    dataset.set_index(numpy.array(range(len(dataset.index))), inplace=True)
    return dataset


#Funzione che mantine una certa percentuale (parametro 'perc') di features in base alla loro correlazione con le classi target
def best_features(train, test, perc):
    temp_trans = OrdinalEncoder(dtype='int')
    train[['protocol_type', 'service', 'flag', 'target']] = temp_trans.fit_transform(train[['protocol_type', 'service', 'flag', 'target']])
    trans = SelectPercentile(f_classif, percentile= perc)
    trans.fit(train.drop('target', axis='columns'), train['target'])
    train[['protocol_type', 'service', 'flag', 'target']] = temp_trans.inverse_transform(train[['protocol_type', 'service', 'flag', 'target']])
    eliminated_columns = trans.get_support()
    bad_features =[]
    for i in range(len(eliminated_columns)):
        if not eliminated_columns[i]:
            bad_features.append(train.columns[i])
    train.drop(bad_features,axis = 'columns', inplace=True)
    test.drop(bad_features,axis = 'columns', inplace=True)
    return train, test