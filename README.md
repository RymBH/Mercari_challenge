# Mercari_challenge

----------- Présentation et contexte du projet------------------

Dans le cadre de notre projet mémoire de fin de formation MLOPS, nous avons réalisé un nouveau projet de machine Learning et de Deep Learning pour relever un challenge lancé par Mercari, le site de e-commerce Japonais.
L’objectif de notre projet est de créer une api destinée aux vendeurs du site. L’api doit pouvoir proposer des suggestions de prix aux vendeurs pour chaque article en fonction des éléments indiqués par le vendeur. 

Ce projet a été développé par:
- Amina ABBI ([GitHub](https://github.com/Amouna95) / [LinkedIn](https://www.linkedin.com/in/amina-abbi?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAACIrt3EBXzTjLFA4D0G7knBANZ0DV9LBqI4&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_all%3BIsfWB9uARUONCWdZ7TYsKQ%3D%3D))
- Eleonora FABRIS ([GitHub](https://github.com/elfabris) / [LinkedIn](https://www.linkedin.com/in/eleonora-fabris?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAAA6z0dABJ84tlvxAqU9GjNE5TVH-VZQU6ik&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_all%3BJbHa5w%2FOQA6hMWchuaJ8%2BA%3D%3D))
- Rym BEN HASSINE ([GitHub](https://github.com/RymBH) / [LinkedIn](https://www.linkedin.com/in/rym-ben-hassine-136b34109/))

Organisation du Projet
------------

Nous avons utilisé Cookiecutter pour définir la structure de notre projet



    ├── README.md          <- The top-level README for developers using this project.
    ├── DATA               <- The Data used for this project:The files are too big and cannot be uploaded in 
    │                        github (> 25MB). The data can be downloaded from Kaggle: 
    │                       https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/data
    ├── FASTAPI
    │  ├── mercari_2.py       <- The mercari API.
    │  ├── train.tsv       <- The original  train dataset.
    │  ├─- test.csv        <- The original test dataset.
    │  ├── model_lgbm.joblib    <- The lgbm model used to predict the price.
    │  ├── model_svr.joblib     <- The svr model used to predict the price.
    │  ├── train_final.npz      <- The final train dataset after the preprocessing step.
    │  ├── test_final.npz       <- The final test dataset after the preprocessing step.
    │  ├── y_cv.npy             <- The validation data after training the model.
    │  ├── x_cv.npz             <- The validation data after training the model.
    │  ├── cv_final.npy         <-  The final validation dataset after the preprocessing step
    │  ├── Dockerfile           <- The Dockerfile.
    │  ├── y_train.npy          <-  The  data used to train the model 
    │  ├── test_mercari.py      <- The unit test for the model
    │  └── test_mercari_api.py  <-  The unit test for the roots of the API
    │  └── historique.json      <-  The historical data of the pricing prediction
    │
    ├── specifications
    │
    ├── MODELS               <- Trained and serialized models, model predictions
    │   ├── model_svr.joblib      <- The lgbm model used to predict the price
    │   └── model_lgbm.joblib     <- The svr model used to predict the price
    │
    ├── NOTEBOOKS             <- Jupyter notebooks. 
    │  ├── Mercari_challenge_EDA_MLOPS.ipynb       <-  Analysis of the data
    │  └── Mercari_challenge_ML&DL_MLOPS.ipynb     <- The models tested 
    │                       
    │
    │ 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │                       
    │
    │
    ├── Mercari_Challenge_Presentation  <- Slides to Present OUR MLOPS project
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


----------- Choix du modèle------------------

Nous avons testé 5 modèles :
Modèles Machine Learning :
- Ridge
- SVR	 
- Random Forest
- LightGBM
  
- Deep Learning : MLP ( multilayer perceptron)

--------- Définition des métriques et exigences de performances-----

Nous avons obtenu la plus faible RMSE avec le modèle de deep learning multilayer perceptron : 0,38.  Cependant, nous avons décidé de garder les deux modèles SVR et Lightgbm pour la partie fastapi qui ont un  RMSE également faible i.e. 0,45 et qui ont  l’avantage d’avoir un temps d’exécution plus rapide que le modèle multilayer perceptron.

------------ Lancement de ce REPO Github ------------------

En local avec Visual Studio
```
$ git clone https://github.com/Amouna95/Projet_mlops_Mercarichallenge.git.
```


=> cette commande permet de copier tous les dossiers et fichiers du REPO dans un nouveau dossier en local.

```
pip install -r requirements.txt
```


------------ Lancement de Fastapi ------------------

La commande suivante permet de lancer l’api :
```
uvicorn mercari_2:api –reload
```

aller sur http://localhost:8000/docs

--------------- Tests réalisés ----------------------

Dans le cadre de ce projet, nous avons écrit plusieurs tests unitaires pour garantir le bon fonctionnement du modèle et de l'API. Voici une description de ces tests:

----Tests du modèle
Les tests du modèle vérifient le chargement des données, la sauvegarde du modèle, la prédiction du modèle et l'évaluation du modèle.

1 - Test de chargement des données: Ce test vérifie que les données de formation et de validation sont correctement chargées à partir des fichiers respectifs. Il vérifie également que les dimensions des données d'entrée (X_train, X_cv) correspondent à celles des cibles (y_train, y_cv).

2 - Test de sauvegarde du modèle: Ce test vérifie que le fichier du modèle (model_svr.joblib) a été correctement sauvegardé et peut être récupéré.

3 - Test de prédiction du modèle: Ce test charge le modèle sauvegardé et fait une prédiction sur les données de validation. Il vérifie ensuite que le nombre de prédictions correspond au nombre d'échantillons dans les données de validation.

4 - Test d'évaluation du modèle: Ce test évalue le modèle sur les données de validation et vérifie que le score RMSE est entre 0 et 1.

----Tests des routes de l'API
Les tests de l'API vérifient que les différentes routes de l'API fonctionnent comme prévu.

5 - Test de la racine: Ce test vérifie que la route racine de l'API renvoie le bon message.

6 - Test de récupération des utilisateurs: Ce test vérifie que la route /vendors de l'API renvoie une réponse avec un statut 200 (succès) et que la réponse est de type dictionnaire.

7 - Test du statut de l'utilisateur: Ce test vérifie que la route /status de l'API renvoie le bon statut pour un utilisateur authentifié.

8 - Test de prédiction avec un utilisateur authentifié: Ce test vérifie que la route /prediction de l'API renvoie une prédiction de prix pour un article Mercari lorsque l'utilisateur est correctement authentifié.

9 - Test de prédiction avec un utilisateur non authentifié: Ce test vérifie que la route /prediction de l'API renvoie le bon message d'erreur lorsque l'utilisateur n'est pas correctement authentifié.

------------ Docker ------------------

La commande suivante permet la création du container mercari :

```
#Creation de l'image mercari_docker
docker build -t mercari_docker .
```

```
#Vérification de la création de l'image mercari_docker
docker image ls
```
```
#Lancement de l'api avec docker
docker run -p 8000:8000 mercari_docker
```

----------- Nous aurions aimé faire bien plus -------------

Si nous avions plus de temps, nous aurions souhaité :
- Mettre notre api sur un Cloud afin de simplifier la procédure d'installation. 
- Rajouter un front-end

----------- Conclusion ------------------------------------

Notre projet s'est achevé avec la mise en œuvre réussie d'une API fonctionnelle capable de prédire le prix de vente d'un article en se basant sur ses caractéristiques principales. Nous avons l’intention de continuer à améliorer l'expérience utilisateur en développant une interface plus intuitive et conviviale, permettant ainsi à l’utilisateur d'estimer facilement le prix de vente de l’article qu’il souhaite déposer. Cette réalisation correspond parfaitement à l'objectif initial du challenge lancé par Mercari.

Au cours de ce projet, nous avons eu l'occasion de mettre en pratique la plupart des concepts et des techniques abordés lors de notre cursus expert MLOps. Cette expérience nous a non seulement permis de consolider nos acquis, mais aussi de comprendre de manière plus concrète la mise en œuvre de ces notions dans un projet réel.

Malgré le manque de temps, nous avons accompli de nombreux progrès et identifié plusieurs domaines d’amélioration pour le futur, notamment au niveau du modèle, tels que la révision du preprocessing et l'automatisation de l'extraction des nouvelles données du site Mercari pour la mise à jour du modèle.

Du côté de l'API, nos ambitions sont de la déployer sur le cloud pour simplifier le processus d'installation et d'ajouter une interface front-end. Bien que le temps nous ait fait défaut pour réaliser ces améliorations, ces défis représentent de nouvelles opportunités pour l'avenir.

En somme, notre travail dans ce projet illustre notre engagement à continuer d'apprendre, d'innover et de repousser les limites de nos capacités en MLOps.
