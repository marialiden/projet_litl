#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Informations sur les calculs:
Calcul du log likelihood repris de: http://termostat.ling.umontreal.ca/doc_termostat/doc_termostat_en.html

Il y a deux versions du code : une version où l'entrée est le document json contenant tous le corpus Reddit-TIFU tel qu'il a été téléchargé sur Huggingface,
et un autre version avec le xml généré par nous qui ne contient que les postes de ce corpus.

Commande: python3 Loglikelihood_Reddit.py 
   
@author: M2 LITL 2022-2023
"""
import re
from lxml import etree
import pandas as pd
import glob
import numpy as np
import sys
import glob
from math import log
from tqdm import tqdm
import json

#On crée un dictionnaire pour stocker les trigrammes et leur fréquence brute dans tout le corpus
trigrammes_tot = dict()
#et une liste avec les messages du corpus.
corpus=[]
#et un compteur pour avoir accés à la longueur (en trigrammes) du corpus (chaque message -2*le nb de messages dans le corpus)
corpus_len=0

#Initialisation d'une liste pour stocker les lignes du fichier json
#posts = []
#with open('tifu_all_tokenized_and_filtered.json', 'r') as fp:
#    for line in fp:
#        posts.append(json.loads(line))
#
## On parcourt les entrées du fichier Json 
#for p in tqdm(posts): 
#    selftext=[] #On initialise une liste pour stocker tous les caractères de chaque message - elle se réinitialise pour chaque poste
#    
#    for i in p["selftext"]: #extraction de la clé de la valeur "selftext" qui contient le poste en entier, sans tokénisation
#        selftext.append(i) 
#    
#    #On colle tous les caractères du messages en une seule variable (string)
#    #On nettoie le texte (text)
#    string = ''.join(selftext)
#    text = string.rstrip("\n")
#    text = re.sub("\t"," ",text)
#    text = re.sub("\n", " ",text)
#    text=re.sub(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', '<url>', text) #on remplace les url par une balise <url>
#    text=text.lower()
#    #On stocke les messages dans la liste corpus
#    corpus.append(text)
#    #On stocke tous les trigrammes du corpus dans un dictionnaire
#    for i in range(0,len(text)-2):
#        corpus_len=corpus_len+1 #1 est ajout pour chaque caractère parcouru dans chaque message, on ne prend pas en compte les deux derniers caractères, qui ne sont pas pris en compte dans le dictionnaire de trigrammes
#        trigramme=text[i:i+3]
#        trigrammes_tot[trigramme]=trigrammes_tot.get(trigramme,0)+1
               
#sortie = open('Trigrammes_frequences_Reddit.csv', 'w')
#sortie.write("Trigramme"+"\t"+"Fréquence brute"+"\n")  
#for x in trigrammes_tot:
#    sortie.write(str(x)+"\t"+str(trigrammes_tot[x])+"\n")  

#Version avec xml au lieu de json - si utilisé, met la version json en commentaire
#Le document est le fichier xml en question
document = etree.parse("Corpus_Reddit_long.xml")
#on parse le fichier xml
root = document.getroot()
message = root.findall('message')
for element in tqdm(message):
        text = element.text
        if text is not None:
            #On nettoie les textes et remplace les urls
            text = text.rstrip("\n")
            text = re.sub("\t"," ",text)
            text = re.sub("\n", " ",text)
            text=re.sub(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', '<url>', text) #on remplace les url par une balise <url>
            text=text.lower()
            #On stocke les messages dans une liste
            corpus.append(text)
            #On stocke les trigrammes dans le dictionnaire avec leur fréquence brute dans le corpus
            for i in range(0,len(text)-2):
                corpus_len=corpus_len+1
                trigramme=text[i:i+3]
                trigrammes_tot[trigramme]=trigrammes_tot.get(trigramme,0)+1

#On crée une liste avec les valeurs des trigrammes du corpus pour faire des stats 
values=list(trigrammes_tot.values())

#Pour ne pas avoir un dataframe trop volumineux, 
#on stocke uniquement les trigrammes qui ont une fréquence brute dans le corpus équivalente ou inférieure à la médiane
#Ici on identifie le seuil qu'on va utiliser plus tard dans le script
seuil=np.quantile(values, 0.25)

#On initialise une liste de liste où on va stocker les informations de chaque trigramme (id, phrase, trigramme, fréquence dans le message, fréquence dans le corpus, c, d, loglikelihood)
stats_tot=[]

#On initialise un compteur de message pour avoir l'identifiant de chaque message
Nb=0

#Version 1: On initialise un dictionnaire où stocke les messages et leurs identifiants pour créer le jsonl à annoter
#messages=dict()

#Version 2: On initialise un dictionnaire où stocke les phrases et l'identifiant du message dans lequel elle apparaît pour créer le jsonl à annoter
sents=dict()

#On initialise un dictionnaire où stocke les tokens et l'identifiant du message dans lequel il apparaît pour créer un dataframe supplémentaire
tokens=dict()

#On parcourt les messages stockés dans la liste corpus, un message à la fois
for x in tqdm(corpus):

    #On crée un dictionnaire où on stocke les trigrammes du message (réinitialisé pour chaque message)
    trigrammes_mess=dict()
    for i in range(0,len(x)-2):
        trigramme=x[i:i+3]
        trigrammes_mess[trigramme]=trigrammes_mess.get(trigramme,0)+1
        
    #On parcourt le dictionnaire des trigrammes du message
    for y in trigrammes_mess:
        #Alternatif : On n'affiche uniquement les stats pour les trigrammes avec une fréquence brute de moins de 5 (pour ne pas avoir trop de données)
#        if trigrammes_mess[y] < 5:
#        a = la fréquence du trigramme dans votre message
#        b = sa fréquence dans le corpus
#        c = la taille (nb de trigrammes) du message (i.e.  len(message)-2)- fréq brute de trigramme dans le message, i.e "Fréquences autres trigrammes"
#        d = la taille (nb de trigrammes) du corpus (i.e. corpus_len)- fréq brute de trigramme dans le corpus, I.e "Fréquences autres trigrammes"
        a= trigrammes_mess[y]
        b= trigrammes_tot.get(y)
        c=(len(x)-2)-a
        d=(corpus_len)-b
        N=int(a+b+c+d)
        #khi2= N*(a*d-b*c)**2/((a+b)*(c+d)*(a+c)*(b+d))
        E1 = ((a+c)*(a+b))/((a+c)*(b+d))
        E2 = ((b+d)*(a+b))/((a+c)*(b+d))
        LL = 2*((a*log(a/E1)) + (b*log(b/E2)))

        #On se limite aux trigrammes dont leur fréquence brute dans le corpus est inférieure ou égal au premier quartile
        if b <= seuil:
            #Si le log likelihood est supérieur à 6.63,
            #la probabilité que le résultat, à savoir la diffèrence entre les deux fréquences, se produit par hasard est inférieure à 1 % 
            #On peut donc être sûr à 99% que nos résultats veulent dire quelque chose
            if LL > 6.63: 
                #messages[x]=Nb #Stockage des messages à annoter dans un dictionnaire (valeur=identifiant, clé=message)
                #On identifi où se trouve le trigramme dans le message pour pouvoir recupérer les mots autour
                match=(re.search(r''+re.escape(y)+r'', x))
                
                
                for i in range(0,len(x)-2):
                    
#                    #On recupère le début et la fin de la phrase      
                    debut=match.start()
                    minimum=match.start()-200
                    while debut>0 and re.search(r'[^\.\?!]',x[debut]) and debut>minimum: #On idenifie le début de la phrase, on s'arrête après 200 caractères si la phrase est trop longue
                        debut=debut-1

                    fin=match.end()
                    maximum= match.end()+200
                    while fin< len(x)-2 and re.search(r'[^\.\?!]',x[fin]) and fin<maximum:#et la fin de la phrase, on s'arrête après 200 caractères si la phrase est trop longue
                        fin=fin+1
#                        
#                   #On recupère le début et la fin du token 
                    start=match.start()                    
                    while start>0 and re.search(r'[^\s\.,!\?\"\\]',x[start]): #On idenifie le début du token
                        start=start-1
                   
                    end=match.end()
                    while end< len(x)-2 and re.search(r'[^\s\.,!\?\"\\]',x[end]):#et la fin du token
                        end=end+1
                        
                sent=x[debut:fin+1] #La phrase à stocker correspond au span qui va du caractère de l'index du debut au caractère de l'index de la fin +1
                token=x[start:end] #Le mot à stocker correspond au span qui va du caractère de l'index du debut au caractère de l'index de la fin
                tokens[token]=Nb
                #print(token)
                sents[sent]=Nb #Stockage des phrases à annoter dans un dictionnaire (valeur=identifiant du message, clé=phrase)
                #On stocke dans la LoL les informations qui nous intéressent pour le trigramme en question
                stats=[Nb,sent,token, y,a,b,c,d,LL] 
                stats_tot.append(stats)
    #Quand toutes les informations d'un message ont été stockées, on ajoute 1 au compteur d'identifiant et on passe au message suivant.                  
    Nb=Nb+1
    
#df_messages= pd.DataFrame(messages.items(), columns=['Message', 'docId'])
#df_messages_tocsv = df_messages.to_csv("toAnnotate_LL_avecseuil5.csv", sep="\t")
#    
df_tokens= pd.DataFrame(tokens.items(), columns=['Token', 'docId'])
df_tokens_tocsv = df_tokens.to_csv("Tokens_Reddit.csv", sep="\t")
#    
##création d'un dataframe avec les phrases contenant les trigrammes extraits
df_phrases= pd.DataFrame(sents.items(), columns=['Phrase', 'docId'])
df_phrases_tocsv = df_phrases.to_csv("toAnnotate_LL_phrases.csv", encoding="UTF-8", quotechar='"', sep="\t", decimal=".", line_terminator="\n")

    
#Création de dataframe avec les trigrammes et les stas
df = pd.DataFrame(stats_tot, columns = ['Id', 'Phrase', 'Token','Trigramme','Fréquence message','Fréquence corpus', 'c', 'd', 'Log likelihood'])
#On tri le dataframe selon la valeur LL
final_df = df.sort_values(by=['Log likelihood'], ascending=False)
csv_data = final_df.to_csv("Reddit_loglikelihood_long_Phrases_Tokens.csv", encoding="UTF-8", quotechar='"', sep="\t", decimal=".", line_terminator="\n")


#On stocke les phrases du dataframe dans un json (avec l'identifiant du document en tant que méta-donnée)
phrases=dict()
sortie = open('toAnnotate_LL_Phrases_Tokens.jsonl', "w+", encoding="utf8") 
for i in range(0, len(df['Phrase'])):
    p=df['Phrase'][i]
    p=re.sub('&nbsp;', ' ',p)
    iddoc=df['Id'][i]
    if p not in phrases:
        phrases[p] = {'text' : str(p),'meta' : {"identifiant" : str(iddoc)}} 
    
        json.dump(phrases[p],sortie, ensure_ascii=False)  # conversion du fichier au format jsonl + écriture dans le fichier 
        sortie.write("\n")