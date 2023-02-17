# -*- coding: utf-8 -*-

"""Script d'extraction de candidats trigrammes inhabituels pour annotation"""

import re
import tqdm
from math import log2
from xml.dom.expatbuilder import Namespaces
from lxml import etree
from lxml.builder import *

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

path_corpus = "../Corpus_Reddit/Corpus_Reddit_long.xml" #/!\ Ligne qui change selon l'environnement
path_corpus_out = "../SortiesPropres/CalculsTfIdf.csv" #/!\ Ligne qui change selon l'environnement

def get_messages(path):
    """Fonction pour récuperer les messages et les numéroter
    Produit un nouveau document XML où l'attribut @id correspond au numéro du message
    Retourne une liste avec tous les messages prétraités"""
    #On utilise la bibliothèque etree pour pouvoir parser le corpus reddit-TIFU qui est au format xml
    #Ici le corpus est appelé "Corpus_reddit.xml" et est placé and un dossier "Corpus_Reddit/"
    document = etree.parse(path_corpus)
    root = document.getroot()
    message = root.findall('message')
    #Les différents messages sont stockés dans un itératif "message"
    newroot = E("Corpus_numerote")

    #Initialisation d'une liste où les postes seront stockés. Ici : un poste=un document. 
    liste = [] 
    stats = []
    incrID = 0

    #On itère dans le corpus pour ajouter le texte de chaque message non vide à une liste
    for element in message:
            text = element.text
            if text is not None: #On enlève les textes vides : 79750 messages
                message2 = etree.SubElement(newroot, "message", attrib={"id":str(incrID)})
                incrID += 1 #On rajoute 1 au numéro d'id
                #Lignes de traitement de texte pour enlever les tabulations, retours à la ligne et les URLs
                text = text.rstrip("\n")
                text = re.sub("\t"," ",text)
                text = re.sub("\n", " ",text)
                text=re.sub(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', '<url>', text) #on remplace les url par une balise <url>
                message2.text = text
                liste.append(text.lower())
                

    CorpusNumerote = etree.ElementTree(newroot) # Lignes pour numéroter les messages au format XML, qu'on pourra récupérer plus tard
    CorpusNumerote.write("../SortiesPropres/Corpus_Redditall_num.xml", pretty_print=True, xml_declaration=True, encoding="utf-8")
    return liste

def get_tfidf(sparse_tf, dense_idf, feature_names):
    """Fonction qui produit un DataFrame dense du tf/idf à partir du tf et de l'idf"""
    dataset = []

    num_row = sparse_tf.shape[0]

    for idDoc in tqdm.tqdm(range(num_row)):
        dense_tf = pd.DataFrame(sparse_tf[idDoc,].toarray()[0], index = feature_names)

        tf_over_idf = dense_tf/dense_idf
        tf_over_idf.rename(columns={0:"tfidf"}, inplace=True)

        tf_over_idf_non_zero = tf_over_idf[(tf_over_idf != 0).values].copy()
        tf_over_idf_non_zero["logtfidf"] = tf_over_idf_non_zero["tfidf"].apply(log2)

        tf_over_idf_non_zero["idDoc"] = idDoc

        tf_over_idf_non_zero = tf_over_idf_non_zero.reset_index().rename(columns={"index": "3gram"})

        dataset.append(tf_over_idf_non_zero)

    return pd.concat(dataset, ignore_index=True)



messages = get_messages(path_corpus)

#Création du dataframe à vectoriser dans l'étape suivante:
#Pour Disco on utilisait input='filename' pour pouvoir traiter le contenu de notre collection de document txt directement, 
#ici on n'a pas besoin parce que le contenu est accessible directement dans le dataframe créé

df = pd.DataFrame(messages, columns=["Message"])


#Initialisation d'un vectoriseur pour le nouveau dataframe (où chaque ligne représente un poste)
count_vectorizer = CountVectorizer(
    max_df = 0.1,
    analyzer="char",
    ngram_range = (3,3)
)

#création d'un vectoriseur TF.IDF
tfidf_transformer = TfidfTransformer(use_idf = True)
#ET d'un vectoriseur seulement TF : on va combiner les deux pour avoir une matrice TF/IDF
tf_transformer = TfidfTransformer(use_idf = False)

#Fit sur nos données
count_vecto = count_vectorizer.fit_transform(df["Message"])

#On utilise l'algorithme tf_idf sur la matrice de trigrammes de char "count_vecto"
tfidf = tfidf_transformer.fit(count_vecto)
#Dont on prend seulement les poids idf (méthode .idf)

#On enregistre les poids idf pour chaque trigramme dans un DataFrame
#Qu'on transpose : on interverti les axes vertical et horizontal
feature_names = count_vectorizer.get_feature_names_out()
dense_idf = pd.DataFrame(tfidf.idf_, index = feature_names)

#On a nos poids idf pour le calcul tf/idf, on passe maintenant à la matrice tf :
#Création d'une matrice avec les ngrammes et leur tf (chaque ligne correspond à un document)
sparse_tf = tf_transformer.fit_transform(count_vecto) #matrice creuse

print("getting TF/IDF")
Tfidf = get_tfidf(sparse_tf, dense_idf, feature_names)

print("Saving CalculsTfIdf")
Tfidf.to_csv(
    "../SortiesPropres/CalculsTfIdf.csv",
    index = False,
    sep=","
)
#Puis on exporte en .csv

print("Loading and applying threshold")


data = pd.read_csv("../SortiesPropres/CalculsTfIdf.csv", encoding="UTF-8", quotechar='"', sep=",", decimal=".", low_memory = False)

x = list(data["logtfidf"])
x = np.array(x)
threshold = float(np.nanquantile(x, 0.02))
data2 = data.query(f"logtfidf < {threshold}")


idDocs_to_annotate = set(data2["idDoc"].values)

df["docId"] = df.index
df_to_annotate = df[df["docId"].isin(idDocs_to_annotate)]

print("Saving results toAnnotate.csv")
df_to_annotate.to_csv(
    "../SortiesPropres/toAnnotate.csv",
    index = False,
    sep=","
)

"""Méthode XML pour récupérer les messages avec leurs ID et les ajouter dans un dataframe"""

arbre = etree.parse("../SortiesPropres/Corpus_Redditall_num.xml")    #On parse l'arbre du XML du fichier
racine = arbre.getroot()

def recupMess (ligne) :
    """Fonction qui récupère les messages depuis le fichier XML numéroté"""
    expr = "//message[@id= $ID]"
    IdLigne = str(int(ligne["idDoc"]))
    Message = racine.xpath(expr, ID = IdLigne)[0].text
    #print(Message)
    return(Message)

Messages = data2.apply(recupMess, axis = 1)
data2["message"] = Messages

stackCol = list(data2.columns) #On récupère le nom des colonnes pour l'export

data2.to_csv("../SortiesPropres/3gramAndMessage.csv",
                  header = stackCol,
                  index = False,
                  sep=",")
#data2 contient maintenant les trigrammes, les métriques, l'id du document et le texte du message

"""Si on n'envoie pas les messages entiers en annotation
alors il faut raisonner au niveau du trigramme extrait/de son token pour construire les extraits à envoyer
-> On place un seuil à la valeur minimum du tf/idf par message pour diminuer le nombre d'éléments à annoter"""

byDoc2 = data2.groupby("idDoc")["logtfidf"].min().reset_index()
DicMini = {}
#On construit un dictionnaire des valeurs minimum par document
def add2Dic (ligne) :
    """Fonction pour récupérer les valeurs minimum du tfidf par document et le mettre dans un dictionnaire"""
    DicMini[int(ligne["idDoc"])] = ligne["logtfidf"]

byDoc2.apply(add2Dic, axis=1)

stackCol = list(data2.columns)
min_df = pd.DataFrame(columns=stackCol)
#Pour mettre ces informations dans un nouveau dataframe

def keepMinimum (ligne):
    """Fonction pandas pour produire un Dataframe qui garde les trigrammes ayant le tf/idf minimum
    à partir du dataframe avec les informations tf/idf de data2 (métriques + messages)"""
    global min_df
    first = ligne["logtfidf"]
    id1 = ligne["idDoc"]
    if (id1, first) in DicMini.items():
        
        insert_row = {"3gram":ligne["3gram"], 
                      "tfidf":ligne["tfidf"], 
                      "logtfidf":ligne["logtfidf"], 
                      "idDoc":ligne["idDoc"], 
                      "message":ligne["message"]}
        min_df = pd.concat([min_df, pd.DataFrame([insert_row])])
        
data2.apply(keepMinimum, axis= 1)

data2 = min_df

res_df = pd.DataFrame(columns=["id", "trigram", "message", "extract", "token", "tfidf", "logtfidf"])
ecart = 35
# Création d'un nouveau DataFrame pour avoir l'information du token

def get_tokens(tri, msg):
    """Fonction pour avoir l'information du token qui contient le trigramme extrait"""
    correct_ngram = ""
    for char in tri:
        if char in "$^+?*()[].\\" :
            correct_ngram += "\\"
        correct_ngram += char
    myregex = rf'((?:^|\W)[\w\"\\\/.?!:\-&]*{correct_ngram}[\w\"\\\/.?!:\-&]*(?:$|\W))'
    match = re.findall(myregex, msg)
    return match

for x in range(0, len(data2['idDoc'])) :   
    tri = str(data2.iloc[x]["3gram"])
    msg = data2.iloc[x]["message"]
    ide = data2.iloc[x]["idDoc"]
    tfidf = data2.iloc[x]["tfidf"]
    logtfidf = data2.iloc[x]["logtfidf"]
    if msg is not None:
        msg = re.sub(r"\s+", " ", msg)
        match = get_tokens(tri, msg)
        for elem in set(match):
            correct_token = ""
            for char in elem:
                if char in "$^+?*()[].\\/|" :
                    correct_token += "\\"
                correct_token += char
            ecart = 35
            regex2 = r"(?:\S*\s){0," + str(ecart) + "}" + correct_token.strip() + "(?:\s\S*){0," + str(ecart) + "}"
            match = re.search(regex2, msg)
            if match :
                if elem not in list(res_df["token"]):
                    insert_row = {"id":ide, "trigram":tri, "message":msg, "extract":match.group(), "token":elem, "tfidf":tfidf, "logtfidf":logtfidf}
                    res_df = pd.concat([res_df, pd.DataFrame([insert_row])])

"""Des erreurs sont survenues avec la partie précédente du code, notamment avec les regex, on les nettoie"""

Col2 = list(res_df.columns)
df_extraits_courts = pd.DataFrame(columns = Col2)
def seuilLong (ligne):
    global df_extraits_courts
    extrait = len(str(ligne["extract"]))
    if extrait >30: #Les extraits ne peuvent pas avoir moins de 30 caractères
        insert_row = {"id":ligne["id"], 
                      "trigram":ligne["trigram"], 
                      "message":ligne["message"], 
                      "extract":ligne["extract"], 
                      "token":ligne["token"], 
                      "tfidf":ligne["tfidf"],
                      "logtfidf":ligne["logtfidf"], 
                     }
        df_extraits_courts = pd.concat([df_extraits_courts, pd.DataFrame([insert_row])])
res_df.apply(seuilLong, axis= 1)



stopdf_dis = pd.DataFrame(columns = Col2)
inverse_df = pd.DataFrame(columns = Col2)
#On créé deux nouveaux dataframes pour enlever les erreurs qui proviennent de certains trigrammes :
#ceux qui comportent des barres de disjonction notamment

def disjonction (ligne):
    global stopdf_dis
    global inverse_df
    x = ligne["trigram"]
    y = str(ligne["token"])
    if ("|" or "/\_") in x or y == "|_/\_|" :
        #print("match")
        insert_row2 = {"id":ligne["id"], 
                  "trigram":ligne["trigram"], 
                  "message":ligne["message"], 
                  "extract":ligne["extract"], 
                  "token":ligne["token"], 
                  "tfidf":ligne["tfidf"],
                  "logtfidf":ligne["logtfidf"], 
                 }
        stopdf_dis = pd.concat([stopdf_dis, pd.DataFrame([insert_row2])])
    else :
        insert_row2 = {"id":ligne["id"], 
                  "trigram":ligne["trigram"], 
                  "message":ligne["message"], 
                  "extract":ligne["extract"], 
                  "token":ligne["token"], 
                  "tfidf":ligne["tfidf"],
                  "logtfidf":ligne["logtfidf"]}
        inverse_df = pd.concat([inverse_df, pd.DataFrame([insert_row2])])

df_extraits_courts.apply(disjonction, axis=1)





inverse_df.to_csv("../SortiesPropres/tfidf_Vfin.csv",index = False, sep=",")

import json 

sortie = open('toAnnotate.jsonl', "w+") 
# remplissage du dictionnaire Python + conversion en jsonl

msgs = dict()   # dictionnaire de dictionnaire 

sortie = open('toAnnotate.jsonl', "w+") # fichier de sortie du script 

for index in range(0, len(inverse_df['Message'])) :   # je parcours le data frame 
    msgs[index] = {'text' : inverse_df.loc[index]["Message"],'meta' : [{"identifiant" : index, "methode" : tfidf}] } # je remplis le dictionnaire
    
    json.dump(msgs[index],sortie)  # conversion du fichier au format jsonl + écriture dans le fichier 
    sortie.write("\n") # jajoute un saut de ligne