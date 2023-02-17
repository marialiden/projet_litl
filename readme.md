# Projet : Détection des chaînes de caractères inhabituelles

## Description 
Lorsque des utilisateurs souhaitent échapper à la censure et publier librement sur une plateforme tout en ayant  conscience qu’ils ne respectent pas les règles d’utilisation, ils mettent en place des stratégies de contournement de filtres de modération. Au fil du temps, les modérateurs de contenu ont pris connaissance de certaines de ces stratégies, et des dispositifs permettant de les détecter automatiquement ont été mis en place. En revanche, les stratégies qui ne sont pas connues des modérateurs ne peuvent être traitées : ce problème est à l’origine de ce projet. Nous partons de l’hypothèse que les tentatives de contournement de filtres de modération sont caractérisées par la présence de chaînes de caractères inhabituelles. Afin de confirmer ou d’infirmer l’hypothèse, nous explorons donc les chaînes de caractères inhabituelles.
Pour mener ce travail, deux méthodes basées sur des métriques de fréquence (TF/IDF et log likelihood) ont été utilisées afin d'identifier les CCI. Ensuite, une campagne d'annotation a eu lieu : deux équipes ont été chargées d'identifier et de catégoriser les CCI selon une typologie. Enfin, une analyse entre les annotations des annotateurs a été menée, ainsi qu'une analyse entre les annotations des annotateurs et les sorties des scripts d'extraction de CCI.

## Contenu de l'archive 
Cette archive contient trois scripts :
ExtractionMessages1.py -> un script d'extraction de trigrammes en fonction de leur score tf/idf
Loglikelihood_Reddit.py -> un script d'extraction de trigrammes selon leur fréquence brute et leur score log likelihood
Accord_Annotateurs.py -> un script pour analyser et comparer les annotations faites par deux équipes sur Prodigy

### ExtractionMessages1.py
- Données d'entrée: fichier .xml contenant tous les postes du Reddit TIFU-SHORT
- Données de sortie:
CalculsTfIdf.csv -> fichier contenant les métriques en détail de chaque trigramme
toAnnotate.csv -> document final contenant les messages à annoter
toAnnotate.jsonl -> document final transformé en jsonl pour l'annotation dans Prodigy

### Loglikelihood_Reddit.py
- Données d'entrée: fichier .xml contenant tous les postes du Reddit TIFU-SHORT ou bien le jsonl
-Données de sortie:
Reddit_loglikelihood_Phrases_Tokens.csv -> fichier contenant les métriques en détail de chaque trigramme
toAnnotate_LL_Phrases_Tokens.jsonl -> document final transformé en jsonl pour l'annotation dans Prodigy

### Accord_Annotateurs.py
- Données d'entrée: les deux fichiers jsonl. contenant les annotations de Prodigy
- Données de sortie:
besedo_annot.csv -> document csv contenant toutes les annotations de l'équipe Besedo, doublons enlevés. Chaque ligne représente une annotation
litl_annot.csv -> document csv contenant toutes les annotations de l'équipe LITL, doublons enlevés. Chaque ligne représente une annotation
Nombre d'annotations de chaque catégorie d'annotation (accord, accord partiel, désaccord)
Le score de kappa de cohen (accord inter-annotateur concernant l'absence ou la présence des CCI)
Les détails des confusions entre les catégories

## Participants 
Équipe LITL : Leïla Fabre, Wissam Kerkri, Maria Lidén, Gabriel Mével, Judith Villedey
Équipe Besedo : Jade Moillic, Roxane Bois, Evgeny Bazarov, Mohamed Bamouh
Enseignante référente : Lydia-Mai Ho-dac
