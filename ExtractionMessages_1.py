import re
#import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from lxml import etree
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from math import log2
import sys, glob, re
from xml.dom.expatbuilder import Namespaces
from lxml.builder import *

#On utilise la bibliothèque etree pour pouvoir parser le corpus reddit-TIFU qui est au format xml
#Ici le corpus est appelé "Corpus_reddit.xml" et est placé and un dossier "Corpus_Reddit/"
document = etree.parse("Corpus_Reddit_long.xml")
root = document.getroot()
message = root.findall('message')
#Les différents messages sont stockés dans un itératif "message"
newroot = E("Corpus_numerote")


#On utilise le tokenizer de SpaCy pour avoir des statistiques sur le nombre de tokens dans le corpus
#nlp = English()
#tokenizer = nlp.tokenizer

#Ci-dessous une alternative :
#on définit une fonction pour tokéniser toutes les chaînes de caractères alphanumériques
#( = chaînes de chiffres, lettres, ou underscores d'une taille de 1 ou +)

# def tokenisation (y):
#    x = re.split("\w+", y)
#    return (len(x))

#Initialisation d'une liste où les postes seront stockés. Ici : un poste=un document. 
liste = [] 
stats = []
incrID = 0

#On itère dans le corpus pour ajouter le texte de chaque message non vide à une liste
for element in message:
        text = element.text
        if text is not None:
             #(tokenisation(str(text)))> 200 :if
                #Formule très lourde pour enlever 15% des post les moins long du calcul
                #Serait sans doute mieux de le faire sur la base des charachtères (pas de tokenisation)
            message2 = etree.SubElement(newroot, "message", attrib={"id":str(incrID)})
            incrID += 1 #On rajoute 1 au numéro d'id
            text = text.rstrip("\n")
            text = re.sub("\t"," ",text)
            text = re.sub("\n", " ",text)
            text=re.sub(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', '<url>', text) #on remplace les url par une balise <url>
            message2.text = text
            liste.append(text.lower())
 
CorpusNumerote = etree.ElementTree(newroot)
#CorpusNumerote.write("Corpus_Reddit_long_url.xml", pretty_print=True, xml_declaration=True, encoding="utf-8") #document xml où les url sont remplacés.

#Création du dataframe à vectoriser dans l'étape suivante:
#Pour Disco on utilisait input='filename' pour pouvoir traiter le contenu de notre collection de document txt directement, 
#ici on n'a pas besoin parce que le contenu est accessible directement dans le dataframe créé
df = pd.DataFrame (liste, columns=["Message"])


#au cas-où j'ai rajouté une colonne pour avoir le numéro du post
#df['NbDoc'] = np.arange(len(df))

#Initialisation d'un vectoriseur pour le nouveau dataframe (où chaque ligne représente un poste)
count_vectorizer = CountVectorizer(max_df = 0.1,
                                    analyzer="char",
                                    ngram_range = (3,3))

#Fit sur nos données
count_vecto = count_vectorizer.fit_transform(df["Message"])

#création d'un vectoriseur TF.IDF
tfidf_transformer = TfidfTransformer(use_idf = True)
#ET d'un vectoriseur seulement TF : on va combiner les deux pour avoir une matrice TF/IDF
tf_transformer = TfidfTransformer(use_idf = False)

#On utilise l'algorithme tf_idf sur la matrice de trigrammes de char "count_vecto"
tfidf = tfidf_transformer.fit(count_vecto)
#Dont on prend seulement les poids idf (méthode .idf)

#On enregistre les poids idf pour chaque trigramme dans un DataFrame
variable = pd.DataFrame(tfidf.idf_, index = count_vectorizer.get_feature_names())

#Qu'on transpose : on interverti les axes vertical et horizontal
variable = variable.transpose()

#On a nos poids idf pour le calcul tf/idf, on passe maintenant à la matrice tf :

#Création d'une matrice avec les ngrammes et leur tf (chaque ligne correspond à un document)
sparse_tf = tf_transformer.fit_transform(count_vecto) #matrice creuse
dense_tf = pd.DataFrame(sparse_tf.todense(),
                        columns = count_vectorizer.get_feature_names()) #matrice dense

#On défini maintenant la formule pour avoir la matrice tf/idf

def tfoveridf (colonne):
    Trigram = colonne.name
    ValIdf = variable[Trigram][0]
    ColSeries = [x/ValIdf for x in colonne]
    return ColSeries

#Qu'on applique

FrameTfidf = dense_tf.apply(tfoveridf, axis = 0)

#On retient le log : impossible à appliquer sur les zéros

#On fait une petite manipulation où on enlève les retour à la ligne et les tabulations des noms de colonnes : pour l'affichage postérieur en .csv

head = list(FrameTfidf.columns)
Header = [re.sub(r"\s", "", x) for x in head]

#Maintenant on choisit de stacker le dataframe pour avoir tous les tfidf sur la même colonne
#Et pour pouvoir réduire la taille du df

Tfidf = FrameTfidf.stack().reset_index() #On stacke

Tfidf = Tfidf.rename(columns = {0:"tfidf","level_0":"idDoc", "level_1":"3gram"}) #On renomme

Tfidf = Tfidf[Tfidf.tfidf !=0] #On enlève les 0



#Maintenant qu'on a plus de 0 on peut utiliser un log sur le tf/idf

def LogFrame (ligne) :
    x = ligne[2]
    print(x)
    if float(x) > 0:
        Log = log2(float(x))
        print(Log)
    else : Log = 0
    return(Log)

# Qu'on applique

LogSeries = Tfidf.apply(LogFrame, axis = 1) #Le résultat du log est stocké dans une Series pandas

Tfidf["logtfidf"] = LogSeries #On remplace la colonne du df par la Series

#On passe maintenant à l'exportation en .csv, notamment pour des calculs avec R

stackCol = list(Tfidf.columns) #On récupère le nom des colonnes pour l'export

Tfidf.to_csv("CalculsTfIdf.csv",
                  header = stackCol,
                  index = False,
                  sep=",")

#Puis on export en .csv

#On utilise la bibliothèque etree pour pouvoir parser le corpus reddit-TIFU qui est au format xml
#Ici le corpus est appelé "Corpus_reddit.xml" et est placé and un dossier "Corpus_Reddit/"

message = newroot.findall('message')
#Les différents messages sont stockés dans un itératif "message"

CorpusNumerote = etree.ElementTree(newroot)
#CorpusNumerote.write("Corpus_Reddit_20k_num.xml", pretty_print=True, xml_declaration=True, encoding="utf-8")

data = pd.read_csv("CalculsTfIdf.csv", encoding="UTF-8", header=0, quotechar='"', sep=",", decimal=".", low_memory = False)

data2 = data.query("logtfidf<-7")

messages = newroot.findall('message')


def recupMess (ligne) :
    expr = "//message[@id= $ID]"
    IdLigne = str(ligne[0])
    Message = newroot.xpath(expr, ID = IdLigne)[0].text
    return(Message)

test = data2.apply(recupMess, axis = 1)


data2["message"] = test

stackCol = list(data2.columns) #On récupère le nom des colonnes pour l'export

data2.to_csv("toAnnotate.csv",
                  header = stackCol,
                  index = False,
                  sep=",")


