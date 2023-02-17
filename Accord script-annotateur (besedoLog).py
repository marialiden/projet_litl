#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import json
import numpy as np


#Ouverture du fichier .jsonl contenant les annotations des annotateurs de Besedo sur les messages issus du script basé sur la métrique du log likelihood
besedo = pd.read_csv("besedo_annot.csv", sep="\t", lineterminator="\n")
besedo.drop(besedo.filter(regex="Unname"),axis=1, inplace=True)

#Dans le fichier 'besedo', on ne garde que les données issus de la méthode "LL" (donc on drop les données de la méthode "tfidf")
besedoLog = besedo.drop(besedo[besedo["method"]=='tfidf'].index)
besedoLog

#Suppression d'une colonne d'index inutile et réinitialisation des indexs
#Création d'un dataframe dans lequel se trouve les phrases étant identiques entre les phrases du dataframe 'df_log' et du dataframe 'besedo'
df_log = pd.read_csv('Reddit_loglikelihood_long_Phrases_Tokens.csv', sep="\t", lineterminator="\n")
df_log.drop(df_log.filter(regex="Unname"),axis=1, inplace=True)
df_log2 = df_log[df_log['Phrase'].isin(besedo['text'])]
df_log2.reset_index(drop=True, inplace=True)
df_log2



#Extraction du contenu textuel entre les index de début et de fin
liste_index=[]
listeColle = []
for index in range(0, len(besedoLog["text"])):
    txt = besedoLog.loc[index]["text"]
    colle=""
    debut = int(besedoLog.loc[index]["start"])
    fin = int(besedoLog.loc[index]["end"])
    etiquette = besedoLog.loc[index]["label"]

    if etiquette != "None":
        for i in txt :
            liste_index.append(i)
        elem = liste_index[debut:fin+1] #liste d'offsets
        for c in elem : 
            colle+= c
        liste_index=[]
        listeColle.append(colle)
   
    elif etiquette == 'None' : 
        listeColle.append('None')

besedoLog["Chaine"] = listeColle
besedoLog




#Identification des annotations identiques, chevauchées, superposées et différentes
c=[]
AnnotSuperposees = []
AnnotChevau = []
labelsSup = []
labelsChev = [] 
TokensNonAnnot = []

import re 

for elem in df_log2["Id"] : 
    c.append(elem)

for y in besedoLog.index :
    ide = besedoLog.loc[y]["identifiant"]
    ch = besedoLog.loc[y]["Chaine"]
    label = besedoLog.loc[y]["label"]

    if ide in c :
        x = c.index(ide)
        tok = df_log2.loc[x]["Token"]
        ph = df_log2.loc[x]["Phrase"]

        #annotations superposées
        if ch in tok or tok in ch : 
            AnnotSuperposees.append(ch)
            labelsSup.append(label)

        if ch in ph : 
            c_deb = ph.index(ch)
            c_fin = c_deb + len(ch)            
            t_deb = ph.index(tok)
            t_fin = t_deb + len(tok)
            
            #annotations chevauchées
            if c_deb < t_deb < c_fin : 
                AnnotChevau.append(ch)
                labelsChev.append(label)
            if t_deb < c_fin < t_fin : 
                AnnotChevau.append(ch)
                labelsChev.append(label)
        
        #annotations différentes
        elif ch not in tok : 
            TokensNonAnnot.append(ch)



#Affichage des différentes informations sur les annotations
sup = len(AnnotSuperposees)
chev = len(AnnotChevau)
tot = len(besedoLog)
diff = len(besedoLog)-(chev+sup) 
sil = len(TokensNonAnnot)

freqS = dict() 
for elem in labelsSup : 
    freqS[elem]=freqS.get(elem,0)+1

freqC = dict()
for elem in labelsChev : 
    freqC[elem]=freqC.get(elem,0)+1
    
print("Tokens détectés par le script mais non annotés :", sil)
print("\n")
    
print("Annotations totales besedoLog: ", tot)
print("\n")

print("Annotation parfaitement identique au script: 0")
print("\n")

print("Annotations superposées log likelihood : ", sup)
for elem in freqS : 
    print(elem, freqS[elem])
print("\n")

print("Annotations chevauchées log likelihood : ", chev)
for elem in freqC : 
    print(elem, freqC[elem])
print("\n")

print("Annotations des tokens qui n'ont pas été détectés par le script :", diff)
print("\n")




