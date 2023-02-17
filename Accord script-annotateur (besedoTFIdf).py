#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import json
import numpy as np


#Ouverture du fichier .jsonl contenant les annotations des annotateurs de Besedo sur les messages issus des deux scripts d'extractions de CCI
besedo = pd.read_csv("besedo_annot.csv", sep="\t", lineterminator="\n")
besedo.drop(besedo.filter(regex="Unname"),axis=1, inplace=True)

#Dans le fichier 'besedo', on ne garde que les données issus de la méthode "tfidf" (donc on drop les données de la méthode "LL")
besedoTfidf = besedo.drop(besedo[besedo["method"]=='LL'].index)
besedoTfidf.reset_index(drop=True, inplace=True)
besedoTfidf

#Suppression d'une colonne d'index inutile et réinitialisation des indexs
#Création d'un dataframe dans lequel se trouve les phrases étant identiques entre les phrases du dataframe 'df_log' et du dataframe 'besedo'
df_tfidf = pd.read_csv('tfidf_Vfin.csv', sep=",", lineterminator="\n")
df_tfidf.drop(df_tfidf.filter(regex="Unname"),axis=1, inplace=True)
df_tfidf2 = df_tfidf[df_tfidf['extract'].isin(besedo['text'])]
df_tfidf2.reset_index(drop=True, inplace=True)
df_tfidf2



#Extraction du contenu textuel entre les index de début et de fin
liste_index=[]
listeColle = []
for index in range(0, len(besedoTfidf["text"])):
    txt = besedoTfidf.loc[index]["text"]
    colle=""
    debut = int(besedoTfidf.loc[index]["start"])
    fin = int(besedoTfidf.loc[index]["end"])
    etiquette = besedoTfidf.loc[index]["label"]

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

besedoTfidf["Chaine"] = listeColle
besedoTfidf



#Identification des annotations identiques, chevauchées, superposées et différentes
cc=[]
Annot_Superposees = []
Annot_Chevau = []
Labelssup = []
Labelschev = [] 
tokensNonAnnot = []

import re 

for elem in df_tfidf2["id"] :ex
    cc.append(elem)

for y in df_tfidf2.index :
    ide = besedoTfidf.loc[y]["identifiant"]
    ch = besedoTfidf.loc[y]["Chaine"]
    label = besedoTfidf.loc[y]["label"]
    
    if ide in cc :
        x = cc.index(ide)
        tok = df_tfidf2.loc[x]["token"]
        ph = df_tfidf2.loc[x]["message"]

        #annotations superposées
        if ch in tok or tok in ch : 
            Annot_Superposees.append(ch)
            Labelssup.append(label)

        #on récup les offsets des annotations pour pouvoir chopper les annotations chevauchées
        if ch in ph : 
            c_deb = ph.index(ch)
            c_fin = c_deb + len(ch)            
            t_deb = ph.index(tok)
            t_fin = t_deb + len(tok)
            
            #annotations chevauchées
            if c_deb < t_deb < c_fin : 
                Annot_Chevau.append(ch)
                Labelschev.append(label)
            if t_deb < c_fin < t_fin : 
                Annot_Chevau.append(ch)
                Labelschev.append(label)

        #annotations différentes
        elif ch not in tok :
            tokensNonAnnot.append(ch)



#Affichage des différentes informations sur les annotations
supp = len(Annot_Superposees)
chevv = len(Annot_Chevau)
tott = len(besedoTfidf)
difff = len(besedoTfidf)-(chevv+supp) 
sill = len(tokensNonAnnot)

freqSs = dict() 
for elem in Labelssup : 
    freqSs[elem]=freqSs.get(elem,0)+1

freqCc = dict()
for elem in Labelschev : 
    freqCc[elem]=freqCc.get(elem,0)+1
    
print("Tokens détectés par le script mais non annotés :", sill)
print("\n")
    
print("Annotations totales besedoTfidf: ", tott)
print("\n")

print("Annotation parfaitement identique au script: 0")
print("\n")

print("Annotations superposées tfidf : ", supp)
for elem in freqSs : 
    print(elem, freqSs[elem])
print("\n")

print("Annotations chevauchées tfidf : ", chevv)
for elem in freqCc : 
    print(elem, freqCc[elem])
print("\n")

print("Annotations des tokens qui n'ont pas été détectés par le script :", difff)
print("\n")



