# TP2 - Inférence Bayésienne - MCMC

Lien de la page GitLab avec la documentation  https://m2cosmo-tp2-167ad6.pages.in2p3.fr/index.html


# Contents:

- [README](#readme)
- [Fonctionnalités](#fonctionnalites)
- [Installation](#installation)
- [Utilisation](#bruitblanc)
- [Contributions](#contributions)
- [License](#license)
- [Documentation](#documentation)
- [Réponses aux Questions](#reponse-aux-questions)


# README

Le readme du projet

## Fonctionnalites

Le logiciel répond aux questions du TP2, sur la détection de surdensité, et la mesure du redshift d'une galaxie.

## Installation

1. Clonez le dépôt GitHub :
   
    ```bash
    $ git clone git@gitlab.in2p3.fr:dimitri.buffat/m2cosmo_tp2.git
    ```
   
2. Allez dans le dossier du package :

    ```bash
    $ cd m2cosmo_tp2
    ```
   
3. Installez le package à l'aide de pip :

    ```bash
    $ pip install .
    ```
   
    Si vous n'avez pas pip, vous pouvez utiliser : 
   
    ```bash
    $ python3 setup.py install
    ```

## Utilisation

Après installation, il suffit de taper 
  ```bash
      $ minimisation
  ```
Fit les données, minimise le chi2 et trace les contours de la fonction de distribution (Partie 1.1)

  ```bash
      $ mcmc
  ```
Lance une seule chaine de Markov et trace rho_0 en fonction des pas du mcmc et de r_p (Partie 1.2)

  ```bash
      $ multi_mcmc
  ```
Lance dix chainesde Markov et trace rho_0 en fonction des pas du mcmc et de r_p (Bonus Partie 1.2)

  ```bash
      $ emcee
  ```
Effectue 10 chaînes MCMC à l’aide de la librairie emcee. Génère des graphiques d’autocorrélation pour chaque paramètre, avec une courbe par chaîne. Supprime la phase de burn-in et applique les pas d’autocorrélation pour échantillonner les points indépendants. Calcule les contours de la fonction de distribution de probabilité et superpose-les à ceux obtenus par une méthode de minimisation pour permettre une comparaison directe. (Partie 1.3)

  ```bash
      $ spectro
  ```
Pour mesurer le redshift par MCMC, d'une galaxie (KISSR 127) (Partie 2)

## Contributions

Les contributions sont les bienvenues! Pour contribuer à 2048 veuillez suivre ces étapes :

1. Forkez le projet.

2. Créez une branche pour votre fonctionnalité 
	```bash
	$ git checkout -b feature-nouvelle-fonctionnalité
	```

3. Commitez vos changements.
	```bash
	$ git commit -m Ajout d une nouvelle fonctionnalité'
	```

4. Poussez votre branche.
	```bash
	$ git push origin feature-nouvelle-fonctionnalité
	```


5. Ouvrez une Pull Request.

## Contributeurs

Merci aux personnes ayant contribué à ce projet :  
- THOMEER Matthieu

## License

Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus de détails.

# Documentation

# tp2_pkg package

## Submodules

## tp2_pkg.mcmc module

### *class* tp2_pkg.mcmc.MCMC(init_sigma, multi_chain=False)

Bases : `object`

Classe pour l’exécution d’une analyse MCMC sur un modèle de données. Cette classe inclut des méthodes pour
l’échantillonnage MCMC, la convergence des chaînes, l’autocorrélation, ainsi que la gestion du suivi du temps et
de l’empreinte carbone associée aux calculs.

#### data_fit

Instance de la classe Minimisation contenant les données à ajuster et le modèle de base.

* **Type:**
  object

#### psigma

Sigma initiale pour les propositions de nouveaux paramètres dans l’échantillonnage.

* **Type:**
  numpy.ndarray

#### theta_current

Valeurs actuelles des paramètres du modèle lors de l’échantillonnage.

* **Type:**
  numpy.ndarray

#### theta_new

Valeurs proposées des paramètres du modèle à chaque itération.

* **Type:**
  numpy.ndarray

#### sampler_theta

Chaîne des paramètres échantillonnés après le MCMC.

* **Type:**
  numpy.ndarray

#### sampler

L’échantillonneur MCMC utilisant l’algorithme de l’ensemble de chaînes de Markov (emcee).

* **Type:**
  emcee.EnsembleSampler

#### sampler_chain

Chaîne d’échantillons obtenus après avoir effectué un sous-échantillonnage de la chaîne MCMC.

* **Type:**
  numpy.ndarray

#### autocorr_func

Fonction d’autocorrélation calculée pour chaque chaîne et chaque paramètre.

* **Type:**
  numpy.ndarray

#### conv

Valeurs de convergence pour chaque paramètre.

* **Type:**
  numpy.ndarray

#### ndim

Nombre de dimensions des paramètres du modèle.

* **Type:**
  int

#### nwalkers

Nombre de « walkers » (chaînes) utilisées dans l’échantillonnage MCMC.

* **Type:**
  int

#### nech

Nombre d’itérations (pas) dans l’échantillonnage MCMC.

* **Type:**
  int

#### burn_in

Période de « burn-in » des échantillons à ignorer pour l’analyse finale.

* **Type:**
  int

#### pas_inde

Seuils d’indépendance pour chaque paramètre en fonction de l’autocorrélation.

* **Type:**
  numpy.ndarray

Initialisation de la classe MCMC.

### Description

Cette méthode initialise l’instance de la classe, charge les données, et prépare les configurations nécessaires
pour l’analyse MCMC.

* **param init_sigma:**
  Valeurs initiales de sigma pour les propositions de nouveaux paramètres.
* **type init_sigma:**
  numpy.ndarray
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_init_\_(init_sigma, multi_chain=False)

Initialisation de la classe MCMC.

#### Description

Cette méthode initialise l’instance de la classe, charge les données, et prépare les configurations nécessaires
pour l’analyse MCMC.

* **param init_sigma:**
  Valeurs initiales de sigma pour les propositions de nouveaux paramètres.
* **type init_sigma:**
  numpy.ndarray
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_compile_data()

Charge et prépare les données nécessaires à l’analyse MCMC.

#### Description

Cette méthode crée une instance de la classe Minimisation et charge les données à ajuster. Elle
initialise aussi les valeurs de paramètres theta_new à partir de l’ajustement initial.

* **param None:**
* **param Cette méthode ne prend pas de paramètre.:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### Main_mcmc()

#### Main_emcee()

#### \_\_function_proposal()

Propose de nouveaux paramètres en fonction de la distribution normale autour des valeurs actuelles.

#### Description

Cette méthode génère de nouvelles valeurs pour les paramètres theta_new en utilisant une distribution normale
centrée sur les valeurs actuelles des paramètres, avec un écart-type défini par psigma.

* **param None:**
* **param Cette méthode ne prend pas de paramètre.:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_update_theta()

Met à jour les valeurs des paramètres du modèle en fonction de la proposition.

#### Description

Cette méthode copie les valeurs actuelles des paramètres dans theta_current et propose ensuite de nouvelles
valeurs pour les paramètres à l’aide de la méthode \_\_function_proposal.

* **param None:**
* **param Cette méthode ne prend pas de paramètre.:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_backup_theta()

Sauvegarde l’état actuel des paramètres du modèle.

#### Description

Cette méthode crée une copie des paramètres actuels dans theta_current pour les restaurer en cas de rejet
de la proposition de nouveaux paramètres.

* **param None:**
* **param Cette méthode ne prend pas de paramètre.:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_log_prior(theta=None)

Calcule la probabilité a priori des paramètres du modèle.

#### Description

Cette méthode évalue la probabilité a priori des paramètres theta en fonction de leurs limites. Si un ou
plusieurs paramètres sont en dehors des limites spécifiées, la probabilité est renvoyée comme -np.inf.

* **param theta:**
  Les paramètres pour lesquels calculer la probabilité a priori. Si None, utilise theta_new.
* **type theta:**
  numpy.ndarray, optional
* **returns:**
  La probabilité a priori des paramètres theta.
* **rtype:**
  float

#### \_\_log_likelihood(theta=None)

Calcule la vraisemblance du modèle en fonction des données observées.

#### Description

Cette méthode évalue la vraisemblance du modèle en utilisant le chi-carré entre les données observées et
le modèle avec les paramètres theta.

* **param theta:**
  Les paramètres pour lesquels calculer la vraisemblance. Si None, utilise theta_new.
* **type theta:**
  numpy.ndarray, optional
* **returns:**
  La vraisemblance du modèle avec les paramètres theta.
* **rtype:**
  float

#### \_\_log_acceptance()

Calcule la probabilité d’acceptation des nouveaux paramètres proposés dans l’échantillonnage.

#### Description

Cette méthode calcule la différence de log-vraisemblance entre les propositions actuelles et nouvelles de
paramètres. La probabilité d’acceptation est ensuite calculée en fonction de cette différence.

* **param None:**
* **param Cette méthode ne prend pas de paramètre.:**
* **returns:**
  La log-probabilité d’acceptation des nouveaux paramètres.
* **rtype:**
  float

#### \_\_accept_or_reject(log_alpha)

Accepte ou rejette les nouveaux paramètres selon la probabilité d’acceptation.

#### Description

Cette méthode accepte les nouveaux paramètres avec une probabilité égale à exp(log_alpha). Sinon, les
paramètres actuels sont restaurés.

* **param log_alpha:**
  La log-probabilité d’acceptation des nouveaux paramètres.
* **type log_alpha:**
  float
* **returns:**
  Les paramètres soit acceptés soit restaurés.
* **rtype:**
  numpy.ndarray

#### \_\_metropolis_hastings(N=10000)

Effectue un échantillonnage MCMC en utilisant l’algorithme de Metropolis-Hastings.

#### Description

Cette méthode exécute l’algorithme de Metropolis-Hastings pour échantillonner les paramètres du modèle.
Elle génère une chaîne d’échantillons des paramètres.

* **param N:**
  Le nombre d’itérations de l’échantillonnage MCMC. Par défaut, N=10000.
* **type N:**
  int, optional
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_multi_metropolis_hastings(M=10, N=10000)

Effectue un échantillonnage MCMC sur plusieurs chaînes en utilisant l’algorithme de Metropolis-Hastings.

#### Description

Cette méthode exécute l’algorithme de Metropolis-Hastings pour échantillonner les paramètres d’un modèle
sur plusieurs chaînes indépendantes. Chaque itération propose une nouvelle valeur des paramètres selon
une distribution de proposition et accepte ou rejette cette valeur en fonction de la probabilité d’acceptation
calculée. Les chaînes sont stockées dans un tableau multidimensionnel de taille (M, N, ndim), où ndim est
le nombre de dimensions du paramètre.

* **param M:**
  Le nombre de chaînes indépendantes à générer. Par défaut, M=10.
* **type M:**
  int, optional
* **param N:**
  Le nombre d’échantillons à générer par chaîne. Par défaut, N=10000.
* **type N:**
  int, optional
* **returns:**
  Cette méthode ne retourne rien. Les chaînes générées sont enregistrées dans l’attribut self.sampler_theta sous forme d’un tableau NumPy.
* **rtype:**
  None

#### PlotMCMC()

Affiche les évolutions des paramètres du modèle durant l’échantillonnage MCMC.

#### Description

Cette méthode génère deux graphiques montrant l’évolution des paramètres rho0 et rp au cours des
itérations de l’échantillonnage MCMC.

* **param None:**
* **param Cette méthode ne prend pas de paramètre.:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_log_probability(theta=None)

Calcule la probabilité totale du modèle, combinant a priori et vraisemblance.

#### Description

Cette méthode calcule la probabilité totale en combinant la log-probabilité a priori et la log-vraisemblance
du modèle.

* **param theta:**
  Les paramètres pour lesquels calculer la probabilité totale. Si None, utilise theta_new.
* **type theta:**
  numpy.ndarray, optional
* **returns:**
  La probabilité totale du modèle pour les paramètres theta.
* **rtype:**
  float

#### \_\_mcmc_emcee()

Effectue un échantillonnage MCMC en utilisant l’échantillonneur emcee.

#### Description

Cette méthode utilise l’échantillonneur EnsembleSampler de la bibliothèque emcee pour effectuer l’échantillonnage
MCMC. Elle initialise les positions des « walkers » et lance l’échantillonnage.

* **param None:**
* **param Cette méthode ne prend pas de paramètre.:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_autocorr()

Calcule la fonction d’autocorrélation pour chaque « walker » et chaque paramètre.

#### Description

Cette méthode calcule la fonction d’autocorrélation pour chaque « walker » dans la chaîne MCMC et pour chaque paramètre.

* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### PlotAutoCorr()

Affiche les fonctions d’autocorrélation pour chaque paramètre en fonction du nombre de pas.

#### Description

Cette méthode génère un graphique pour chaque paramètre montrant l’autocorrélation en fonction des pas d’échantillonnage.

* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_convergence()

Calcule et affiche la convergence de chaque paramètre dans les chaînes MCMC.

#### Description

Cette méthode évalue la convergence des chaînes MCMC en calculant le critère de Gelman-Rubin pour chaque paramètre.

* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_selection()

Effectue un sous-échantillonnage des chaînes MCMC en fonction de la période de « burn-in » et de l’indépendance.

#### Description

Cette méthode supprime les premiers échantillons (période de burn-in) et applique un sous-échantillonnage
pour obtenir des échantillons indépendants.

* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### GetDist()

Affiche un graphique des distributions des paramètres à partir des échantillons MCMC.

#### Description

Cette méthode utilise la bibliothèque getdist pour générer un graphique des distributions des paramètres
obtenus après le sous-échantillonnage de la chaîne MCMC.

* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### SuperposeDist()

Superpose les distributions des paramètres obtenues par MCMC et par minimisation du $chi^2$.

#### Description

Cette méthode utilise la bibliothèque getdist pour tracer un diagramme en triangle
des distributions marginales des paramètres. Les distributions obtenues par l’échantillonnage
MCMC (via self.sampler_chain) sont comparées à celles obtenues par un tirage simulé autour
des valeurs minimisant le $chi^2$ (via self.data_fit._tirage()). Les deux distributions
sont superposées et affichées sur le même graphique avec des légendes explicites.

* **param None:**
* **returns:**
  Cette méthode ne retourne rien. Le graphique est affiché à l’écran.
* **rtype:**
  None

list of weak references to the object (if defined)

## tp2_pkg.minimisation module

### *class* tp2_pkg.minimisation.Minimisation

Bases : `object`

### Descrition

> Initialise tout les attributs sur None et lance les calculs.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_init_\_()

#### Descrition

> Initialise tout les attributs sur None et lance les calculs.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_compile_data()

#### Description

> Methode prive qui lance les calculs preliminaires.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### Main()

#### Description

> Methode qui realise tout les print et l’affichage des graphiques.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_load_data()

#### Description

> Methode prive. Charge les donnees .npz.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### PrintData()

#### Description

> Trace les donnees.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_bruit_blanc(n=300)

#### Description

> Methode privee. Genere un bruit blanc gaussien.
* **param n:**
* **type n:**
  Integer, optional : the default is 300.

#### Retourne

> Un array numpy de taille n.

#### \_\_bruit_colore(n=300)

#### Description

> Methode privee .
> Genere un bruit colore a partir d’un bruit blanc gaussien et de la PSD issue des donnees.”
* **param n:**
* **type n:**
  Integer, optional : the default is 300.

#### Retourne

> Un array numpy de taille n

#### \_\_covariance()

#### Description

> Methode prive. Evalue la matrice de covariance.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_inv_covariance()

#### Description

> Methode privee. Calcul la matrice inverse de la matrice de covariance.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_model(r, p0, rp, A, mu, sigma)

#### Description

> Methode protegee.
> Definition d’un modele mathematiques pour nos donnees.”
* **param r:**
  Ensemble des r a appliquer sur ce modele.
* **type r:**
  Numpy array
* **param p0:**
  Amplitude de la fonction densite.
* **type p0:**
  Float
* **param rp:**
  Rayon caracteristique de l’amas de galaxie etudie.
* **type rp:**
  Float
* **param A:**
  Amplitude de la fonction gaussienne.
* **type A:**
  Float
* **param mu:**
  Valeur centree de la fonction gaussienne.
* **type mu:**
  Float
* **param sigma:**
  Ecart-type de la fonction gaussienne.
* **type sigma:**
  Float
* **param Retourne:**
* **param ——–:**
  Un array numpy.
  Points du modele evaluee en r.

#### \_\_fit(p0=1, rp=1000, A=0.02, mu=1500, sigma=200)

#### Description

> Methode prive. Realise un fit des donnees a partir du modele.
* **param p0:**
  Amplitude de la fonction densite.
* **type p0:**
  Float, optional : the default is 1.
* **param rp:**
  Rayon caracteristique de l’amas de galaxie etudie.
* **type rp:**
  Float, optional : the default is 1000.
* **param A:**
  Amplitude de la fonction gaussienne.
* **type A:**
  Float, optional : the default is 0.02.
* **param mu:**
  Valeur centree de la fonction gaussienne.
* **type mu:**
  Float, optional : the default is 1500.
* **param sigma:**
  Ecart-type de la fonction gaussienne.
* **type sigma:**
  Float, optional : the default is 200.
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### PrintFit()

#### Description

> Trace les donnees et le fit sur le meme graphique.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_chisq()

#### Description

> Methode protegee. Calcul le $chi^2$.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_chisq_red()

#### Description

> Methode protegee. Calcul le $chi^2_{reduit}$.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### PrintChisq()

#### Description

> Affiche les valeurs du $chi^2$ et du $chi^2_{reduit}$
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_tirage()

#### Description

> Methode protegee.
> On realise 1000 tirages de lot des 5 parametres du probleme.
> Chaque tirage est realise sur une loi normale, centree sur les valeurs du fit et d’ecart type la matrice de covariance calculer pendant le fit.”
* **param None:**

#### Retourne

> Un array numpy

#### GetDist()

#### Description

> Trace les courbes de confiance associee a nos ajustements.
* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None


list of weak references to the object (if defined)

## tp2_pkg.spectro module

### *class* tp2_pkg.spectro.SpectroAnalysis

Bases : `object`

Classe pour l’analyse spectroscopique d’un spectre de galaxie. Cette classe inclut des fonctionnalités pour
la lecture des données, le traitement de bruit, l’ajustement d’un modèle gaussien avec un continuum linéaire,
et l’analyse des chaînes MCMC.

#### filename

Chemin vers le fichier contenant les données spectroscopiques.

* **Type:**
  String

#### data

Données spectroscopiques chargées depuis le fichier.

* **Type:**
  pandas.DataFrame

#### noise_signal

Sous-ensemble des données représentant la région du bruit.

* **Type:**
  pandas.DataFrame

#### lin_fit_params

Paramètres de l’ajustement linéaire sur le bruit (pente et intercept).

* **Type:**
  tuple

#### data_centered

Données après soustraction du modèle linéaire.

* **Type:**
  pandas.DataFrame

#### noise_signal_centered

Données de bruit centrées après soustraction du modèle linéaire.

* **Type:**
  pandas.DataFrame

#### core_data

Sous-ensemble des données représentant la région centrale contenant les pics OIII.

* **Type:**
  pandas.DataFrame

#### fit_params

Paramètres du modèle gaussien ajusté aux données centrales.

* **Type:**
  numpy.ndarray

#### theta_origin

Meilleurs paramètres initiaux pour MCMC, issus de l’ajustement initial.

* **Type:**
  numpy.ndarray

#### theta_new

Meilleurs paramètres finaux obtenus après les analyses MCMC.

* **Type:**
  numpy.ndarray

#### thin

Largeur de corrélation utilisée pour le sous-échantillonnage des chaînes MCMC.

* **Type:**
  int

### Description

Méthode d’initialisation de la classe.
Charge le fichier de données, initialise les paramètres et lance l’analyse complète.

* **param filename:**
  Chemin du fichier avec les données de spectrométrie.
* **type filename:**
  String
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### \_\_init_\_()

#### Description

Méthode d’initialisation de la classe.
Charge le fichier de données, initialise les paramètres et lance l’analyse complète.

* **param filename:**
  Chemin du fichier avec les données de spectrométrie.
* **type filename:**
  String
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### run_analysis()

#### Description

Coeur du programme, effectue l’analyse complète : lecture des données, ajustements, traitement du bruit,
et analyse MCMC.

* **param None:**
* **returns:**
  Flux ajusté à partir du modèle gaussien.
* **rtype:**
  numpy.ndarray

#### read_csv()

#### Description

Lit les données spectroscopiques depuis un fichier CSV
et les stocke dans un DataFrame self.data.

* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### selection(type_data, lambda_min, lambda_max=7274.57784357302)

#### Description

Sélectionne un sous-ensemble des données dans la plage donnée
de longueurs d’onde [lambda_min, lambda_max].

* **param type_data:**
  Ensemble de données spectroscopiques.
* **type type_data:**
  pandas.DataFrame
* **param lambda_min:**
  Longueur d’onde minimale pour la sélection.
* **type lambda_min:**
  float
* **param lambda_max:**
  Longueur d’onde maximale pour la sélection, par défaut 7274.57784357302.
* **type lambda_max:**
  float, optionnel
* **returns:**
  Données filtrées dans la plage donnée.
* **rtype:**
  pandas.DataFrame

#### *static* linear_model(x, a, b)

#### Description

> Définit un modèle linéaire sous la forme y = ax + b.
* **param x:**
  Tableau des abscisses (longueurs d’onde).
* **type x:**
  numpy.ndarray
* **param a:**
  Pente du modèle linéaire.
* **type a:**
  float
* **param b:**
  Ordonnée à l’origine du modèle linéaire.
* **type b:**
  float
* **returns:**
  Valeurs de y correspondant aux valeurs de x.
* **rtype:**
  numpy.ndarray

#### fit_linear_model()

#### Description

Ajuste un modèle linéaire sur les données de bruit et stocke
les paramètres ajustés dans self.lin_fit_params.

* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### subtract_linear_model(dataset)

#### Description

Soustrait le modèle linéaire défini par self.lin_fit_params
des flux du dataset donné.

* **param dataset:**
  Données auxquelles soustraire le modèle linéaire.
* **type dataset:**
  pandas.DataFrame
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### calculate_std_dev()

#### Description

Calcule l’écart type des flux dans les données de bruit centrées.

* **param None:**
* **returns:**
  Écart type des flux.
* **rtype:**
  float

#### gaussian_model(wavelength, A, sigma_g, z, a, b)

#### Description

Définit un modèle avec deux pics gaussiens correspondant aux
raies OIII (4959 et 5007 Å) sur un continuum linéaire.

* **param wavelength:**
  Tableau des longueurs d’onde.
* **type wavelength:**
  numpy.ndarray
* **param A:**
  Amplitude du premier pic gaussien.
* **type A:**
  float
* **param sigma_g:**
  Largeur à mi-hauteur des pics gaussiens.
* **type sigma_g:**
  float
* **param z:**
  Redshift.
* **type z:**
  float
* **param a:**
  Pente du continuum linéaire.
* **type a:**
  float
* **param b:**
  Ordonnée à l’origine du continuum linéaire.
* **type b:**
  float
* **returns:**
  Valeurs du modèle pour les longueurs d’onde données.
* **rtype:**
  numpy.ndarray

#### fit_OIII()

#### Description

Ajuste un modèle gaussien avec un continuum linéaire sur les pics OIII
et stocke les paramètres ajustés dans self.fit_params.

* **param None:**
* **returns:**
  Flux ajusté à partir du modèle.
* **rtype:**
  numpy.ndarray

#### plot_data(title, x, y, x_label, y_label, labels, colors, styles=None)

#### Description

Génère un graphique personnalisé avec les courbes données.

* **param title:**
  Titre du graphique.
* **type title:**
  String
* **param x:**
  Liste des tableaux pour les abscisses.
* **type x:**
  list of numpy.ndarray
* **param y:**
  Liste des tableaux pour les ordonnées.
* **type y:**
  list of numpy.ndarray
* **param x_label:**
  Nom de l’axe des abscisses.
* **type x_label:**
  String
* **param y_label:**
  Nom de l’axe des ordonnées.
* **type y_label:**
  String
* **param labels:**
  Liste des légendes pour chaque courbe.
* **type labels:**
  list of String
* **param colors:**
  Liste des couleurs pour chaque courbe.
* **type colors:**
  list of String
* **param styles:**
  Liste des styles de ligne pour chaque courbe. Le défaut est None.
* **type styles:**
  list of String, optionnel
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### plot_spectra()

#### Description

Produit un ensemble de sous-graphiques pour visualiser le spectre,
le bruit, les données centrées et l’ajustement du modèle.

* **param None:**
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### log_prior(theta)

#### Description

Calcule la probabilité a priori des paramètres selon des limites définies.

* **param theta:**
  Tableau des paramètres du modèle.
* **type theta:**
  numpy.ndarray
* **returns:**
  Logarithme de la probabilité a priori.
* **rtype:**
  float

#### log_likelihood(theta, wavelengths, fluxes, flux_errors)

#### Description

Calcule la vraisemblance logarithmique pour un ensemble de paramètres donnés.

* **param theta:**
  Tableau des paramètres du modèle.
* **type theta:**
  numpy.ndarray
* **param wavelengths:**
  Tableau des longueurs d’onde.
* **type wavelengths:**
  numpy.ndarray
* **param fluxes:**
  Tableau des flux observés.
* **type fluxes:**
  numpy.ndarray
* **param flux_errors:**
  Tableau des erreurs sur les flux.
* **type flux_errors:**
  numpy.ndarray
* **returns:**
  Logarithme de la vraisemblance.
* **rtype:**
  float

#### log_probability(theta, wavelengths, fluxes, flux_errors)

#### Description

Calcule la probabilité totale (prior + likelihood) pour un ensemble de paramètres donnés.

* **param theta:**
  Tableau des paramètres du modèle.
* **type theta:**
  numpy.ndarray
* **param wavelengths:**
  Tableau des longueurs d’onde.
* **type wavelengths:**
  numpy.ndarray
* **param fluxes:**
  Tableau des flux observés.
* **type fluxes:**
  numpy.ndarray
* **param flux_errors:**
  Tableau des erreurs sur les flux.
* **type flux_errors:**
  numpy.ndarray
* **returns:**
  Logarithme de la probabilité totale.
* **rtype:**
  float

#### parameters_initialisation(best_param)

#### Description

Initialise les positions des marcheurs pour le MCMC en ajoutant des variations
gaussiennes autour des meilleurs paramètres initiaux.

* **param best_param:**
  Paramètres initiaux optimaux obtenus par ajustement.
* **type best_param:**
  numpy.ndarray
* **returns:**
  Position initiale des marcheurs, nombre de marcheurs, et nombre de dimensions.
* **rtype:**
  tuple

#### run_mcmc(n_steps=10000)

#### Description

Effectue deux étapes de MCMC pour ajustement des paramètres et
retourne les résultats.

* **param n_steps:**
  Nombre d’étapes pour le MCMC. Le défaut est 10000.
* **type n_steps:**
  int, optionnel
* **returns:**
  Sampler MCMC et échantillons aplatis après burn-in.
* **rtype:**
  tuple

#### analyze_chains(sampler, labels=['A', 'sigma_g', 'z', 'a', 'b'])

#### Description

Analyse les chaînes MCMC : convergence, temps d’auto-corrélation,
et estimation des paramètres. Donne la valeur du redshift calculée.

* **param sampler:**
  Objet contenant les chaînes MCMC.
* **type sampler:**
  emcee.EnsembleSampler
* **param flat_samples:**
  Échantillons aplatis après burn-in.
* **type flat_samples:**
  numpy.ndarray
* **param labels:**
  Noms des paramètres, par défaut [« A », « sigma_g », « z », « a », « b »].
* **type labels:**
  list of String, optionnel
* **returns:**
  Cette méthode ne retourne rien.
* **rtype:**
  None

#### error()


list of weak references to the object (if defined)

## Module contents

# Reponse aux questions

## Valeur du χ²

La valeur de χ² obtenue est cohérente avec la valeur attendue. En effet, le χ² réduit calculé est proche de 1, ce qui indique que le modèle ajusté reproduit correctement les données, tout en respectant le degré de liberté statistique de l’analyse.

## Utilisation de la librairie *emcee* et test de convergence

1. **Critère de Gelman-Rubin (𝑅̂)**
   Afin d’assurer une convergence satisfaisante des chaînes de Markov, le critère de Gelman-Rubin 𝑅̂ doit être inférieur à 1,03. Dans ce contexte, nous avons déterminé qu’un minimum de 10 000 pas est nécessaire pour satisfaire cette condition pour chaque simulation.
2. **Autocorrélation**
   L’analyse de l’autocorrélation permet d’estimer la longueur de la corrélation dans les chaînes. On considère que deux points *i* et *j* ne sont plus corrélés lorsque leur autocorrélation devient inférieure à une valeur absolue de 0,1. Pour garantir une mesure conservatrice, nous retenons la valeur maximale de l’autocorrélation parmi les 50 combinaisons possibles (10 chaînes et 5 paramètres), ce qui conduit à une estimation de 300 pas pour atteindre l’indépendance statistique.

## Comparaison des contours de confiance entre MCMC et minimisation

L’approche MCMC produit des contours de confiance plus larges comparés à ceux obtenus par des méthodes de minimisation. Cela peut être attribué à l’influence des *priors* dans l’algorithme MCMC, qui contribuent à une exploration plus complète de l’espace des paramètres et incluent des incertitudes supplémentaires, contrairement à la minimisation qui se concentre sur le maximum de vraisemblance sans inclure explicitement ces contributions.

