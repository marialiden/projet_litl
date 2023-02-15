#!/usr/bin/env python
# coding: utf-8

# <H1> Formats d'entrée</H1> 
# 
# <H2>V1 (pour la fonction get_dataframe_V1):</H2> 
# 
# {"text":". the email is the guy\u2019sname@adam\u2019soldcompany.","meta":"[{'identifiant': '13540', 'methode': 'LL'}]","_input_hash":176898049,"_task_hash":1048993731,"_session_id":"unusual-char-m2-v1-YWxlamFuZHJvLmN1ZXJ2b0BiZXNlZG8uY29t","_view_id":"ner_manual","answer":"accept","spans":"[{'start': 19, 'end': 45, 'label': 'Other'}]"} 
# 
# 
# <H2>V2 (pour la fonction get_dataframe_V2):</H2>  
# {"text":". the email is the guy\u2019sname@adam\u2019soldcompany.","meta":[{'identifiant': '13540', 'methode': 'LL'}],"_input_hash":176898049,"_task_hash":1048993731,"_session_id":"unusual-char-m2-v1-YWxlamFuZHJvLmN1ZXJ2b0BiZXNlZG8uY29t","_view_id":"ner_manual","answer":"accept","spans":[{'start': 19, 'end': 45, 'label': 'Other'}]
# 
# **À NOTER:**
# - Spans peut contenir plusieurs annotations, à savoir plusieurs étiquettes "start", "end" et "label"
# - La clé du spans est un string dans V1, mais une liste dans V2


#Importation des librairies nécéssaires
import pandas as pd
import json
import re
import numpy as np
from sklearn.metrics import cohen_kappa_score


#On ouvre le fichier à traiter, à savoir le jsonl contenant les annotations de l'équipe besedo
besedo = open("unusual-char-m2-besedo.jsonl", 'r', encoding='utf8')

#Fonction qui crée un dataframe du fichier jsonl, en séparant les valeurs du début et de la fin de l'annotation dans deux colonnes différentes
#Chaque ligne du dataframe correspond à une annotation 
def get_dataframe_V1(fichier):
    texts_task=[] #liste de liste où on stocke le texte + le task_hash pour pouvoir enlever les doublons
 
    AnalyseAnnot = pd.DataFrame(columns=["text", "identifiant", "method", "start", "end", "label", "answer"]) #En-tête du dataframe
    for line in fichier.readlines() :    # lecture du fichier ligne par ligne
        json_object = json.loads(line)   # on stocke la ligne sous forme de dictionnaire dans la variable json_object

        # on peut maintenant accéder à chaque élément du dictionnaire 
        
        txt = json_object["text"]     # bout de texte annoté 
        anno = json_object["answer"]  # action finale de l'annotateur (valider, ignorer, rejeter)
        task= json_object["_task_hash"] #la valeur de _task_hash
        txt_hash=[txt,task] #liste de liste contenant le texte et la valeur de _task_hash
        
        #Si le même texte avec le même _task_hash n'a pas déjà été vu, alors on commence à stocker les informations qui nous intéressent dans une ligen du dataframe
        if txt_hash not in texts_task: 
            texts_task.append(txt_hash)
            
            myregex = r"(\d+).*methode.*\b(\w+)"      # pour isoler les éléments de "meta" : identifiant & methode
            match = re.search(myregex, json_object["meta"])
            if match : 
                ide = match.group(1)
                met = match.group(2)
            
            # Si spans est vide, ou s'il y a des crichets sans contenu dedans, on stocke un 0 dans l'index de début et de fin
            # On met une étiquette 'None' (string) dans la colonne label
            # Vu qu'on fait des tranches de listes pour recupérer le span annoté dans le texte plus tard, les index à 0 nous permettent de rien recupérer si l'annotateur n'a rien annoté
            if json_object["spans"] is None or json_object["spans"]=="[]":    
                deb = int(0)
                fin = int(0) 
                tag ="None"

                insert_row = {"text":txt,      # on stocke les variables dans un dictionnaire 
                          "identifiant":ide, 
                          "method":met, 
                          "start":deb,
                          "end":fin,
                          "label":tag,
                          "answer":anno
                         }

                AnalyseAnnot = pd.concat([AnalyseAnnot, pd.DataFrame([insert_row])])   # on ajoute la ligne au dataframe
            
            #On continue avec les cas où spans n'est pas vide
            elif json_object["spans"] != None :      # pour isoler les éléments de "spans" : index début & fin & label

                myregex2 = r"start\': (\d+), \'end\': (\d+), \'label\': \'(\w+ ?\w+)"
                #Vu que certains texts ont plusieurs annotations, on donne chaque annotation sa propre ligne dans le dataframe que l'on crée
                match2 = re.finditer(myregex2, json_object["spans"])
                if match2:
                    for x in match2:
                        deb = int(x.group(1))   
                        fin = int(x.group(2))
                        tag = x.group(3)
                        insert_row = {"text":txt,      # on stocke les variables dans un dictionnaire 
                                      "identifiant":ide, 
                                      "method":met, 
                                      "start":deb,
                                      "end":fin,
                                      "label":tag,
                                      "answer":anno
                                     }

                        AnalyseAnnot = pd.concat([AnalyseAnnot, pd.DataFrame([insert_row])])   # on ajoute la ligne au dataframe

    AnalyseAnnot.reset_index(drop=True, inplace=True) #Pour que l'index du dataframe soit correcte, on le réinitialise
 
    return AnalyseAnnot


#Création du dataframe des anotations de Besedo
df_besedo=get_dataframe_V1(besedo)


#On vérifie que tout est correcte dans le dataframe 
df_besedo

#Pour une raison inconnue, la colonne label a des valeurs None s'il n'y a pas d'annotations, 
#même si on a mis None comme un string dans la fonction ci-dessus. 
#Pour régler le problème, les None sont transformés en 'None'
df_besedo['label'] = df_besedo['label']. replace(np. nan, 'None')


#Exportation du dataframe sous format csv
df_besedo.to_csv('besedo_annot.csv', sep="\t", encoding="utf8")

#Création et exportation d'un dataframe contenant uniquement les annotations "other"
other_df= df_besedo[df_besedo['label']=='Other']
#other_df.to_csv('besedo_other.csv', sep="\t", encoding="utf8")

#Fonction qui permet de lire le fichier jsol comme un objet json, mais ligne par ligne (bien pour pouvoir exploiter le fait que ce soit des dictionnaires et des listes dedans)
def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

litl = load_jsonl("char-m2-litl.jsonl") #On utilise la fonction sur notre fichier jsonl

#Fonction qui permet de créer un dataframe si le fichier jsonl n'a pas de guillemets autour de la liste de la clé "spans"
def get_dataframe_V2(fichier):
    text_task=[]
    AnalyseAnnot = pd.DataFrame(columns=["text", "identifiant", "method", "start", "end", "label", "answer"]) 

    #x=un dictionnaire
    for x in fichier:
        ide="" #variable pour l'identifiant
        met="" #variable pour la méthode
        anno=x['answer'] #variable pour la valeur de la clé "answer"
        txt=x['text'] #variable pour la valeur de la clé "text"
        task= x['_task_hash']
        txt_hash=[txt,task]
        
        if txt_hash not in text_task:
            text_task.append(txt_hash)
            for j in x:    

                if j=="meta":
                    for o in x[j]:
                        #z=clé du dictionnaire o (start, end, token_start, token_end, label)
                        for z in o:
                            if z =="identifiant": #stockage de l'index de début
                                ide=o[z] #la valeur de la clé z du dictionnaire o

                            elif z =="methode": #stockage de l'index de fin
                                met=o[z]

                elif j=="spans":
                    #x[j]= la valeur associée à la clé "spans" (c'est une liste des dictionnaires)
                    for o in x[j]:

                        #z=clé du dictionnaire o (start, end, token_start, token_end, label)
                        for z in o:
                            #print(len(o))
                            if z =="start": #stockage de l'index de début
                                deb=int(o[z]) #la valeur de la clé z du dictionnaire o
                            elif z =="end": #stockage de l'index de fin
                                fin=int(o[z])
                            elif z=="label":
                                tag=o[z]
                    #print(txt, ide, met,deb, fin, tag, sep="\t")
                        insert_row = {"text":txt,      # on stocke les variables dans un dictionnaire 
                                  "identifiant":ide, 
                                  "method":met, 
                                  "start":deb,
                                  "end":fin,
                                  "label":tag,
                                  "answer":anno
                                 }
                        AnalyseAnnot = pd.concat([AnalyseAnnot, pd.DataFrame([insert_row])])#on utilise le dictionnaire pour créer une nouvelle ligne du dataframe


            if "spans" not in x:
                deb=int(0)
                fin=int(0)
                tag='None'
                
                #print(txt,ide, met,deb, fin, tag, sep="\t")
                insert_row = {"text":txt,      # on stocke les variables dans un dictionnaire 
                              "identifiant":ide, 
                              "method":met, 
                              "start":deb,
                              "end":fin,
                              "label":tag,
                              "answer":anno
                             }
                AnalyseAnnot = pd.concat([AnalyseAnnot, pd.DataFrame([insert_row])])

    AnalyseAnnot.reset_index(drop=True, inplace=True) #Pour que l'index du dataframe soit correcte, on le réinitialise
    
    return AnalyseAnnot



#Création du dataframe des annotations de l'équipe litl
df_litl=get_dataframe_V2(litl)
df_litl



#Exportation du dataframe sous format .csv
df_litl.to_csv('litl_annot.csv', sep=",", encoding="utf8")


#Répartition des labels dans le dataframe Besedo (toutes les annotations)
df_besedo['label'].value_counts()


#Répartition des methodes dans le dataframe Besedo (toutes les annotations)
df_besedo['method'].value_counts()


#On créé un nouveau dataframe de Besedo qui contient uniquement les textes annotés par l'équipe litl
df_besedo2= (df_besedo[df_besedo.text.isin(df_litl.text)])
df_besedo2
df_besedo2.reset_index(drop=True, inplace=True)

#Pour ne pas risquer de comparer des textes annotés par litl mais pas annotés par Besedo
#On fait un nouveau dataframe de litl contenant uniquement les textes annotés par l'équipe Besedo
df_litl2=(df_litl[df_litl.text.isin(df_besedo2.text)])
df_litl2
df_litl2.reset_index(drop=True, inplace=True)

#Fonction qui permet de parcourir un dataframe où chaque ligne correspond à une annotation et donner en sortie un dictionnaire contenant une clé par extrait annoté
#La clé= l'extrait
#La valeur : "SANS CCI" si l'extrait contient des labels autres que 'None' et "CCI" s'il contient 'None'
def cci_ou_pas(df):
    ligne=dict()
    for x in df.index:
        if df['label'][x]=='None':
            ligne[df['text'][x]]='SANS CCI'
        else:
            ligne[df['text'][x]]='CCI'
        
    return ligne

bes_annot= cci_ou_pas(df_besedo2)
litl_annot=cci_ou_pas(df_litl2)

#Transformation en dataframe
df_bes= pd.DataFrame(bes_annot.items(), columns=['text', 'type'])
df_lit= pd.DataFrame(litl_annot.items(), columns=['text', 'type'])

#Listes pour stocker les étiquettes associés à chaque exrait pour faire le calcul de kappa
bes_liste=[]
litl_liste=[]
couples=dict()

#On parcourt les annotations de litl
for i in df_lit.index:
    type_l=df_lit['type'][i]
    text= df_lit['text'][i]
    
    #Pour chaque ligne de df_litl, on parcourt les lignes du df_besedo2
    for j in df_bes.index:
        #Si le texte du df_besedo2 correspond au texte du df_litl, alors on crée des variables pour les valeurs qui nous intéressent
        if df_bes['text'][j]==text:
            type_b=df_bes['type'][j]
            couples[(type_b,type_l)]=couples.get((type_b,type_l),0)+1
            bes_liste.append(type_b)
            litl_liste.append(type_l)

#Affichage du score de kappa global
score_global= cohen_kappa_score(bes_liste,litl_liste)
print('score_global', round(score_global,2))
#Affichage des détails du score de kappa
for x in couples:
    print(x, couples[x])

#Pour savoir combien d'extraits annotés par Litl n'ont pas été annotés par Besedo

#On stocke tous les textes annotés par Litl dans un ensemble 
extrait1=set()
for i in range(0, len(df_litl['text'])):
    text=str(df_litl['text'][i])
    extrait1.add(text)
    
#et tous les textes annotés par Litl ET Besedo dans un autre ensemble 
extrait=set()
for i in range(0, len(df_besedo2['text'])):
    text=str(df_besedo2['text'][i])
    extrait.add(text)
    
#On affiche le nombre de cas qui diffère
print(len(extrait1 - extrait))



#Affichage des labels de l'équipe Besedo
df_besedo2.label.value_counts()


#Affichage des labels de l'équipe Litl
df_litl2.label.value_counts()


# <H2> Comparaison des annotations </H2>
# 
# Catégories d'annotation:
# - Catégorie 1 : Accord. Nous considérons qu’un accord correspond à une annotation dont l’étiquette est pareille et dont le span est soit identique, soit superposé ou chevauché. Dans cette catégorie on inclut également tous les cas où les deux annotateurs ont considéré qu’il n’y a pas de CCI dans l’extrait.
# - Catégorie 2 : Accord concernant la présence d’une CCI, mais désaccord concernant l’étiquette attribuée. Tout comme dans la catégorie 1, nous incluons les annotations superposées et chevauchées.
# - Catégorie 3 : Désaccord. Il s’agit des annotations pour lesquelles le segment ayant été annoté par une équipe n’a pas été annoté du tout par l’autre équipe. Autrement dit, l'une des équipes a détecté une CCI mais l’autre équipe ne l’a pas détectée.  

#On ajoute une colonne avec l'équipe d'annotateurs pour pouvoir analyser le dataframe de la catégorie 3 plus tard 
df_litl2['annotateur']='LITL'
df_besedo2['annotateur']='BESEDO'

#création des nouveaux dataframes
df_litl_2 = pd.DataFrame(columns=list(df_litl.columns))
df_besedo_2 = pd.DataFrame(columns=df_besedo2.columns)

df_commun = pd.DataFrame(columns=list(df_litl.columns) + ["cat"] + list(df_besedo2.columns))

#Initialisation d'une liste par catégorie d'annotation
Annot_cat1=[]
Annot_cat2=[]

cat_1=0
cat_2=0

#Initialisation d'une liste par catégorie et par équipe d'annotateur
fs_b=[]
fs_l=[]
other_b=[]
other_l=[]
none_b=[]
none_l=[]
ks_b=[]
ks_l=[]
emp_b=[]
emp_l=[]
ono_b=[]
ono_l=[]
mis_b=[]
mis_l=[]

#Initialisation d'une liste par équipe d'annotateur, toutes les catégories confodues
lab_l=[]
lab_b=[]

#Fonction pour remplir les listes si catégorie 1
def stockage_cat1(besedo, cat, liste_litl, liste_besedo):
    if besedo==cat:
        liste_litl.append('OUI')
        liste_besedo.append('OUI')
    if besedo!=cat:
        liste_litl.append('NON')
        liste_besedo.append('NON')
        
#Fonction pour remplir les listes si catégorie 2
def stockage_cat2(besedo,litl, cat, liste_litl, liste_besedo):
    if besedo==cat:
        liste_litl.append('NON')
        liste_besedo.append('OUI')
    elif litl==cat:
        liste_litl.append('OUI')
        liste_besedo.append('NON')
    else:
        liste_litl.append('NON')
        liste_besedo.append('NON')
        

#Fonction pour remplir les listes si catégorie 2
def stockage_cat3(label, cat, liste_besedo, liste_litl):    
    if label==cat:
        liste_besedo.append('OUI')
        liste_litl.append('NON')
    elif label!=cat:
        liste_besedo.append('NON')
        liste_litl.append('NON')

#On parcourt les annotations de litl
for i in df_litl2.index:
    start_l= int(df_litl2['start'][i]) #Transformation en int nécéssaire, sinon pas possible de recupérer le span annoté dans le texte
    end_l=int(df_litl2['end'][i])
    label_l=df_litl2['label'][i]
    met_l=df_litl2['method'][i]
    text= df_litl2['text'][i]
    id_l=df_litl2['identifiant'][i]
    answer_l=df_litl2['answer'][i]
    annot_l=df_litl2['annotateur'][i]
    line_l = [text, id_l, met_l, start_l, end_l, label_l, answer_l,annot_l] #On crée une liste contenant les informations qu'on va stocker dans le nouveau dataframe
    
    #Pour chaque ligne de df_litl, on parcourt les lignes du df_besedo2
    for j in df_besedo2.index:
        #Si le texte du df_besedo2 correspond au texte du df_litl, alors on crée des variables pour les valeurs qui nous intéressent
        if df_besedo2['text'][j]==text:
            start_b= int(df_besedo2['start'][j])
            end_b=int(df_besedo2['end'][j])
            label_b=df_besedo2['label'][j]
            met_b=df_besedo2['method'][j]
            id_b=df_besedo2['identifiant'][j]
            answer_b=df_besedo2['answer'][j]
            annot_b=df_besedo2['annotateur'][j]
            line_b=[text, id_b, met_b, start_b, end_b, label_b, answer_b, annot_b]
            
            litl=["litl",text,text[start_l:end_l], start_l,end_l,label_l, met_l]
            besedo=["besedo",df_besedo2['text'][j], df_besedo2['text'][j][start_b:end_b], start_b,end_b,label_b, met_b]
            
            # CAT 1 : label identique et span :
            # identique
            if label_b==label_l and start_b==start_l and end_b==end_l: 
                df_litl_2.loc[len(df_litl_2)] = line_l
                df_besedo_2.loc[len(df_besedo_2)]=line_b
                df_commun.loc[len(df_commun)] = line_l
                df_commun.loc[len(df_commun)] = line_b
                cat_1+=1
                Annot_cat1.append(litl)
                Annot_cat1.append(besedo)
                lab_l.append(label_l)
                lab_b.append(label_b)
                stockage_cat1(label_b, 'Funny spelling', fs_l, fs_b)
                stockage_cat1(label_b, 'Emphasis', emp_l, emp_b)
                stockage_cat1(label_b, 'Onomatopoeia', ono_l, ono_b)
                stockage_cat1(label_b, 'Other', other_l, other_b)
                stockage_cat1(label_b, 'Key smashing', ks_l, ks_b)
                stockage_cat1(label_b, 'None', none_l, none_b)
                stockage_cat1(label_b, 'Mistake', mis_l, mis_b)

            # superposé
            elif label_b == label_l and (start_l >= start_b and end_l <= end_b or start_l <= start_b and end_l >= end_b):
                df_litl_2.loc[len(df_litl_2)] = line_l
                df_besedo_2.loc[len(df_besedo_2)]=line_b
                df_commun.loc[len(df_commun)] = line_l
                df_commun.loc[len(df_commun)] = line_b
                cat_1+=1
                Annot_cat1.append(litl)
                Annot_cat1.append(besedo)
                lab_l.append(label_l)
                lab_b.append(label_b)
                stockage_cat1(label_b, 'Funny spelling', fs_l, fs_b)
                stockage_cat1(label_b, 'Emphasis', emp_l, emp_b)
                stockage_cat1(label_b, 'Onomatopoeia', ono_l, ono_b)
                stockage_cat1(label_b, 'Other', other_l, other_b)
                stockage_cat1(label_b, 'Key smashing', ks_l, ks_b)
                stockage_cat1(label_b, 'None', none_l, none_b)
                stockage_cat1(label_b, 'Mistake', mis_l, mis_b)

                    
            # chevauché
            elif label_b == label_l and (start_l > start_b and end_l > end_b and start_l < end_b or start_l < start_b and end_l < end_b and end_l > start_b):
                df_litl_2.loc[len(df_litl_2)] = line_l
                df_besedo_2.loc[len(df_besedo_2)]=line_b
                df_commun.loc[len(df_commun)] = line_l
                df_commun.loc[len(df_commun)] = line_b
                cat_1+=1
                Annot_cat1.append(litl)
                Annot_cat1.append(besedo)
                lab_l.append(label_l)
                lab_b.append(label_b)
                stockage_cat1(label_b, 'Funny spelling', fs_l, fs_b)
                stockage_cat1(label_b, 'Emphasis', emp_l, emp_b)
                stockage_cat1(label_b, 'Onomatopoeia', ono_l, ono_b)
                stockage_cat1(label_b, 'Other', other_l, other_b)
                stockage_cat1(label_b, 'Key smashing', ks_l, ks_b)
                stockage_cat1(label_b, 'None', none_l, none_b)
                stockage_cat1(label_b, 'Mistake', mis_l, mis_b)

            # CAT 2 : Label différent et span :
            #identique
            elif start_b==start_l and end_b==end_l and label_b!=label_l:
                cat_2+=1
                df_litl_2.loc[len(df_litl_2)] = line_l
                df_besedo_2.loc[len(df_besedo_2)]=line_b
                df_commun.loc[len(df_commun)] = line_l
                df_commun.loc[len(df_commun)] = line_b
                Annot_cat2.append(litl)
                Annot_cat2.append(besedo)
                lab_l.append(label_l)
                lab_b.append(label_b)
                stockage_cat2(label_b, label_l, 'Funny spelling', fs_l, fs_b)
                stockage_cat2(label_b, label_l, 'Emphasis', emp_l, emp_b)
                stockage_cat2(label_b, label_l, 'Onomatopoeia', ono_l, ono_b)
                stockage_cat2(label_b, label_l,'Other', other_l, other_b)
                stockage_cat2(label_b, label_l,'Key smashing', ks_l, ks_b)
                stockage_cat2(label_b, label_l,'None', none_l, none_b)
                stockage_cat2(label_b, label_l,'Mistake', mis_l, mis_b)

            #superposé
            elif label_b != label_l and (start_l >= start_b and end_l <= end_b or start_l <= start_b and end_l >= end_b):
                if end_b>0 and end_l>0:
                    cat_2+=1
                    df_litl_2.loc[len(df_litl_2)] = line_l
                    df_besedo_2.loc[len(df_besedo_2)]=line_b
                    df_commun.loc[len(df_commun)] = line_l
                    df_commun.loc[len(df_commun)] = line_b
                    Annot_cat2.append(litl)
                    Annot_cat2.append(besedo)
                    lab_l.append(label_l)
                    lab_b.append(label_b)
                    stockage_cat2(label_b, label_l, 'Funny spelling', fs_l, fs_b)
                    stockage_cat2(label_b, label_l, 'Emphasis', emp_l, emp_b)
                    stockage_cat2(label_b, label_l, 'Onomatopoeia', ono_l, ono_b)
                    stockage_cat2(label_b, label_l,'Other', other_l, other_b)
                    stockage_cat2(label_b, label_l,'Key smashing', ks_l, ks_b)
                    stockage_cat2(label_b, label_l,'None', none_l, none_b)
                    stockage_cat2(label_b, label_l,'Mistake', mis_l, mis_b)

            # chevauché 
            elif label_b != label_l and (start_l > start_b and end_l > end_b and start_l < end_b or start_l < start_b and end_l < end_b and end_l > start_b):
                if end_b>0 and end_l>0:
                    cat_2+=1
                    df_litl_2.loc[len(df_litl_2)] = line_l
                    #df_litl_2.loc[len(df_litl_2)]=df_litl.iloc[[i]]
                    df_commun.loc[len(df_commun)] = line_l
                    df_commun.loc[len(df_commun)] = line_b
                    df_besedo_2.loc[len(df_besedo_2)]=line_b
                    Annot_cat2.append(litl)
                    Annot_cat2.append(besedo)
                    lab_l.append(label_l)
                    lab_b.append(label_b)
                    stockage_cat2(label_b, label_l, 'Funny spelling', fs_l, fs_b)
                    stockage_cat2(label_b, label_l, 'Emphasis', emp_l, emp_b)
                    stockage_cat2(label_b, label_l, 'Onomatopoeia', ono_l, ono_b)
                    stockage_cat2(label_b, label_l,'Other', other_l, other_b)
                    stockage_cat2(label_b, label_l,'Key smashing', ks_l, ks_b)
                    stockage_cat2(label_b, label_l,'None', none_l, none_b)
                    stockage_cat2(label_b, label_l,'Mistake', mis_l, mis_b)

#Création d'un dataframe contenant toutes les annotations de la catégorie 3(à savoir toutes les annotations qui ne font pas partie de la catégorie 1 ou 2)
#On concatène d'abord le df avec les annotation de la catégorie 1 et 2 et le dataframe des deux annotateurs
#Puis on enlève tous les doublons (à savoir tous les extraits qui figurent dans les deux catégories)
annot_cat3= pd.concat([df_litl2, df_besedo2, df_commun], axis=0).drop_duplicates(keep=False)
annot_cat3.reset_index(inplace = True,drop = True)

for i in range(0, len(annot_cat3)):
    label=annot_cat3['label'][i]
    stockage_cat3(label, 'Emphasis', emp_b, emp_l) 
    stockage_cat3(label, 'Onomatopoeia', ono_b, ono_l)    
    stockage_cat3(label, 'Other', other_b, other_l)    
    stockage_cat3(label, 'Key smashing', ks_b, ks_l)    
    stockage_cat3(label, 'Funny spelling', fs_b, fs_l)    
    stockage_cat3(label, 'Mistake', mis_b, mis_l)    
    stockage_cat3(label, 'None', none_b, none_l)    
    lab_b.append(label)
    lab_l.append('NULL')

#Nombre d'annotations de chaque catégorie
print("Catégorie 1:", cat_1)
print("Catégorie 2:", cat_2)
print("Catégorie 3:", len(annot_cat3))


#Pourcentage d'annotations de catégorie 1 et 2 
(cat_1+cat_2)/(cat_1+cat_2)+len(annot_cat3))

#Affichage du dataframe de la catégorie 3
annot_cat3

#Exportation du dataframe de la catégorie 3
annot_cat3.to_csv('Annotations_cat3.csv', sep="\t", encoding="utf8")

#Calcul de kappa de cohen, global et par étiquette
score_global= cohen_kappa_score(lab_b,lab_l)
score_fs= cohen_kappa_score(fs_b,fs_l)
score_other= cohen_kappa_score(other_b,other_l)
score_none= cohen_kappa_score(none_b,none_l)
score_mistake= cohen_kappa_score(mis_b,mis_l)
score_emp= cohen_kappa_score(emp_b,emp_l)
score_ks= cohen_kappa_score(ks_b,ks_l)
score_ono= cohen_kappa_score(ono_b,ono_l)
print('score_global', round(score_global,2))
print('score fs', round(score_fs,2))
print('score other', round(score_other,2))
print('score none', round(score_none,2))
print('score mistake', round(score_mistake,2))
print('score emphasis', round(score_emp,2))
print('score ks', score_ks)
print('score onomatopoeia',round(score_ono,2))


#Affichage de la répartition des étiquettes dans annot_cat3
annot_cat3.label.value_counts()


#Création d'un dataframe de la catégorie 1 (pour l'analyse)
cat1_df=pd.DataFrame(Annot_cat1, columns=["annotateur","text","span","start", "end", "label", "method"])


#Répartition des labels dans la catégorie 1
cat1_df.label.value_counts()

#Export du dataframe en csv pour la catégorie 1
cat1_df.to_csv('Annotations_cat1.csv', sep="\t", encoding="utf8")


#Création d'un dataframe de la catégorie 2 (pour l'analyse)
cat2_df=pd.DataFrame(Annot_cat2, columns=["annotateur","text","span","start", "end", "label", "method"])


#Répartition des labels dans la catégorie 2
cat2_df['label'].value_counts()


#Export du dataframe en csv pour la catégorie 2
cat2_df.to_csv('Annotations_cat2.csv', sep="\t", encoding="utf8")


#Création d'un dictionnaire où on stocke toutes les paires d'annotations possibles et leur comptage,
#utilisé pour l'analyse des confusions
paire=dict()
for i in range(0, len(lab_b)):
    p=(lab_b[i], lab_l[i])
    paire[p]=paire.get(p,0)+1
for x in paire:
    print(x, paire[x])

