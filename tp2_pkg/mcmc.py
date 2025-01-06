import numpy as np
import matplotlib.pyplot as plt
import emcee as emc
from getdist import MCSamples, plots
from tp2_pkg import minimisation as mini


import time
import codecarbon as cc
import logging

version = "1.0"

logging.getLogger("codecarbon").disabled = True

class MCMC:
    """
    Classe pour l'exécution d'une analyse MCMC sur un modèle de données. Cette classe inclut des méthodes pour 
    l'échantillonnage MCMC, la convergence des chaînes, l'autocorrélation, ainsi que la gestion du suivi du temps et 
    de l'empreinte carbone associée aux calculs.

    Attributes
    ----------
    data_fit : object
        Instance de la classe Minimisation contenant les données à ajuster et le modèle de base.
    psigma : numpy.ndarray
        Sigma initiale pour les propositions de nouveaux paramètres dans l'échantillonnage.
    theta_current : numpy.ndarray
        Valeurs actuelles des paramètres du modèle lors de l'échantillonnage.
    theta_new : numpy.ndarray
        Valeurs proposées des paramètres du modèle à chaque itération.
    sampler_theta : numpy.ndarray
        Chaîne des paramètres échantillonnés après le MCMC.
    sampler : emcee.EnsembleSampler
        L'échantillonneur MCMC utilisant l'algorithme de l'ensemble de chaînes de Markov (emcee).
    sampler_chain : numpy.ndarray
        Chaîne d'échantillons obtenus après avoir effectué un sous-échantillonnage de la chaîne MCMC.
    autocorr_func : numpy.ndarray
        Fonction d'autocorrélation calculée pour chaque chaîne et chaque paramètre.
    conv : numpy.ndarray
        Valeurs de convergence pour chaque paramètre.
    ndim : int
        Nombre de dimensions des paramètres du modèle.
    nwalkers : int
        Nombre de "walkers" (chaînes) utilisées dans l'échantillonnage MCMC.
    nech : int
        Nombre d'itérations (pas) dans l'échantillonnage MCMC.
    burn_in : int
        Période de "burn-in" des échantillons à ignorer pour l'analyse finale.
    pas_inde : numpy.ndarray
        Seuils d'indépendance pour chaque paramètre en fonction de l'autocorrélation.
    """
    
    def __init__(self, init_sigma, multi_chain=False):
        """
        Initialisation de la classe MCMC.        

        Description
        -----------
        Cette méthode initialise l'instance de la classe, charge les données, et prépare les configurations nécessaires 
        pour l'analyse MCMC.
        
        Parameters
        ----------
        init_sigma : numpy.ndarray
            Valeurs initiales de sigma pour les propositions de nouveaux paramètres.

        Return
        ------
        None
            Cette méthode ne retourne rien.
        """
        self.data_fit = None
        self.psigma = init_sigma
        self.multi_chain = multi_chain

        self.theta_current = None
        self.theta_new = None
        self.sampler_theta = None
        
        self.sampler = None
        self.sampler_chain = None
        self.autocorr_func = None
        self.conv = None
        
        self.ndim = None
        self.nwalkers = None
        self.nech = None
        self.burn_in = None
        self.pas_inde = None
        
        self.__compile_data()
        
    def __compile_data(self):
        """
        Charge et prépare les données nécessaires à l'analyse MCMC.
        
        Description
        -----------
        Cette méthode crée une instance de la classe Minimisation et charge les données à ajuster. Elle 
        initialise aussi les valeurs de paramètres `theta_new` à partir de l'ajustement initial.
        
        Parameters
        ----------
            None
            Cette méthode ne prend pas de paramètre.
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """
        self.data_fit = mini.Minimisation()
        #self.data_fit.Main()
        self.theta_new = self.data_fit.popt
        self.ndim = len(self.theta_new)
        
    def Main_mcmc(self):
        """
        Gère l'exécution principale de l'algorithme MCMC avec suivi des émissions de CO2.

        Description
        -----------
        Cette méthode exécute l'algorithme Metropolis-Hastings, soit en mode multi-chaînes 
        (`multi_chain=True`) soit en mode simple chaîne (`multi_chain=False`). Pendant l'exécution, 
        elle utilise le tracker d'émissions de CO2 pour mesurer l'empreinte carbone générée par 
        le calcul. À la fin, elle affiche les émissions estimées et génère les graphiques associés.

        Parameters
        ----------
            None
            Cette méthode ne prend pas de paramètre.

        Return
        ------
        None
            Cette méthode ne retourne rien.
        """
        tracker = cc.EmissionsTracker(save_to_file=False)
        
        if self.multi_chain:
            tracker.start()
            self.__multi_metropolis_hastings()
            emission = tracker.stop()
            print(f"\nCO2 generer pour notre MCMC : {emission} kg.\n")
            
        else:
            tracker.start()
            self.__metropolis_hastings()
            emission = tracker.stop()
            print(f"\nCO2 generer pour notre MCMC : {emission*10} kg.\n")

        self.PlotMCMC()
        
    def Main_emcee(self):
        """
        Gère l'exécution principale de l'algorithme emcee avec suivi des émissions de CO2.

        Description
        -----------
        Cette méthode exécute l'algorithme MCMC basé sur `emcee`. Elle suit les émissions de CO2 
        générées pendant l'exécution à l'aide d'un tracker. Après le calcul, elle effectue des 
        analyses supplémentaires, notamment la convergence, l'autocorrélation, la sélection des 
        échantillons, et la génération des distributions de paramètres, avant de superposer 
        les distributions.

        Parameters
        ----------
            None
            Cette méthode ne prend pas de paramètre.

        Return
        ------
        None
            Cette méthode ne retourne rien.
        """
        tracker = cc.EmissionsTracker(save_to_file=False)
        tracker.start()
        self.__mcmc_emcee()
        emission = tracker.stop()
        print(f"CO2 generer pour MCMC emcee : {emission} kg.")
        
        self.__convergence()
        self.__autocorr()
        self.PlotAutoCorr()
        self.__selection()
        self.GetDist()
        self.SuperposeDist()

    def __function_proposal(self):
        """
        Propose de nouveaux paramètres en fonction de la distribution normale autour des valeurs actuelles.
        
        Description
        -----------
        Cette méthode génère de nouvelles valeurs pour les paramètres `theta_new` en utilisant une distribution normale 
        centrée sur les valeurs actuelles des paramètres, avec un écart-type défini par `psigma`.
        
        Parameters
        ----------
            None
            Cette méthode ne prend pas de paramètre.
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """
        self.theta_new = np.random.normal(self.theta_new, self.psigma)
    
    def __update_theta(self):
        """
        Met à jour les valeurs des paramètres du modèle en fonction de la proposition.
        
        Description
        -----------
        Cette méthode copie les valeurs actuelles des paramètres dans `theta_current` et propose ensuite de nouvelles 
        valeurs pour les paramètres à l'aide de la méthode `__function_proposal`.
        
        Parameters
        ----------
            None
            Cette méthode ne prend pas de paramètre.
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """

        self.theta_current = np.copy(self.theta_new)
        self.__function_proposal()
        
    def __backup_theta(self):
        """
        Sauvegarde l'état actuel des paramètres du modèle.
        
        Description
        -----------
        Cette méthode crée une copie des paramètres actuels dans `theta_current` pour les restaurer en cas de rejet 
        de la proposition de nouveaux paramètres.
        
        Parameters
        ----------
            None
            Cette méthode ne prend pas de paramètre.
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """
        self.theta_new = np.copy(self.theta_current)
    
    def __log_prior(self, theta=None):
        """
        Calcule la probabilité a priori des paramètres du modèle.
        
        Description
        -----------
        Cette méthode évalue la probabilité a priori des paramètres `theta` en fonction de leurs limites. Si un ou 
        plusieurs paramètres sont en dehors des limites spécifiées, la probabilité est renvoyée comme `-np.inf`.
        
        Parameters
        ----------
        theta : numpy.ndarray, optional
            Les paramètres pour lesquels calculer la probabilité a priori. Si `None`, utilise `theta_new`.
        
        Return
        ------
        float
            La probabilité a priori des paramètres `theta`.
        """

        if theta is None:
            cdt_p0 = self.theta_new[0] > 0 and self.theta_new[0] < 0.015
            cdt_rp = self.theta_new[1] > 0 and self.theta_new[1] < 1500
            cdt_A = self.theta_new[2] > 0 and self.theta_new[2] < 0.03
            cdt_mu = self.theta_new[3] > 1450 and self.theta_new[3] < 1600
            cdt_sigma = self.theta_new[4] > 200 and self.theta_new[4] < 350

        else:
            cdt_p0 = theta[0] > 0 and theta[0] < 0.015
            cdt_rp = theta[1] > 0 and theta[1] < 1500
            cdt_A = theta[2] > 0 and theta[2] < 0.03
            cdt_mu = theta[3] > 1450 and theta[3] < 1600
            cdt_sigma = theta[4] > 200 and theta[4] < 350

        if cdt_p0 and cdt_rp and cdt_A and cdt_mu and cdt_sigma:
            return 0
        
        return -np.inf
    
    def __log_likelihood(self, theta=None):
        """
        Calcule la vraisemblance du modèle en fonction des données observées.
        
        Description
        -----------
        Cette méthode évalue la vraisemblance du modèle en utilisant le chi-carré entre les données observées et 
        le modèle avec les paramètres `theta`.
        
        Parameters
        ----------
        theta : numpy.ndarray, optional
            Les paramètres pour lesquels calculer la vraisemblance. Si `None`, utilise `theta_new`.
        
        Return
        ------
        float
            La vraisemblance du modèle avec les paramètres `theta`.
        """

        if theta is None:
            self.data_fit._model(self.data_fit.data['r'], *self.theta_new)
            
        else:
            self.data_fit._model(self.data_fit.data['r'], *theta)
            
        self.data_fit._chisq()
        return -0.5 * self.data_fit.chisq
    
    def __log_acceptance(self):
        """
        Calcule la probabilité d'acceptation des nouveaux paramètres proposés dans l'échantillonnage.
        
        Description
        -----------
        Cette méthode calcule la différence de log-vraisemblance entre les propositions actuelles et nouvelles de 
        paramètres. La probabilité d'acceptation est ensuite calculée en fonction de cette différence.
        
        Parameters
        ----------
            None
            Cette méthode ne prend pas de paramètre.
        
        Return
        ------
        float
            La log-probabilité d'acceptation des nouveaux paramètres.
        """
        log_prior_current = self.__log_prior()
        log_likelihood_current = self.__log_likelihood()
        
        self.__update_theta()
        log_prior_new = self.__log_prior()
            
        if log_prior_new == -np.inf:
            return -np.inf

        log_likelihood_new = self.__log_likelihood()        
        return log_prior_new + log_likelihood_new - log_prior_current - log_likelihood_current
    
    def __accept_or_reject(self, log_alpha):
        """
        Accepte ou rejette les nouveaux paramètres selon la probabilité d'acceptation.
        
        Description
        -----------
        Cette méthode accepte les nouveaux paramètres avec une probabilité égale à `exp(log_alpha)`. Sinon, les 
        paramètres actuels sont restaurés.
        
        Parameters
        ----------
        log_alpha : float
            La log-probabilité d'acceptation des nouveaux paramètres.
        
        Return
        ------
        numpy.ndarray
            Les paramètres soit acceptés soit restaurés.
        """

        if np.random.uniform() < np.exp(log_alpha):
            return self.theta_new
        
        self.__backup_theta()
        return self.theta_current
    
    def __metropolis_hastings(self, N=10000):
        """
        Effectue un échantillonnage MCMC en utilisant l'algorithme de Metropolis-Hastings.
         
        Description
        -----------
        Cette méthode exécute l'algorithme de Metropolis-Hastings pour échantillonner les paramètres du modèle. 
        Elle génère une chaîne d'échantillons des paramètres.
        
        Parameters
        ----------
        N : int, optional
            Le nombre d'itérations de l'échantillonnage MCMC. Par défaut, N=10000.
       
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """

        samples = []
        start_time = time.time()
        print('\n')
        
        for n in range(N):
            _time = 10*(time.time() - start_time)
            minutes = int(_time // 60)
            seconds = int(_time % 60)
            print(f"\rProgression : {int((n+1)/N*100)}% [{minutes:02d}:{seconds:02d}]", end='', flush=True)
            log_alpha = self.__log_acceptance()
            theta_current = self.__accept_or_reject(log_alpha)
            samples.append(theta_current)
            
        self.sampler_theta = np.array(samples)
    
    def __multi_metropolis_hastings(self, M=10, N=10000):
        """
        Effectue un échantillonnage MCMC sur plusieurs chaînes en utilisant l'algorithme de Metropolis-Hastings.
        
        Description
        -----------
        Cette méthode exécute l'algorithme de Metropolis-Hastings pour échantillonner les paramètres d'un modèle 
        sur plusieurs chaînes indépendantes. Chaque itération propose une nouvelle valeur des paramètres selon 
        une distribution de proposition et accepte ou rejette cette valeur en fonction de la probabilité d'acceptation 
        calculée. Les chaînes sont stockées dans un tableau multidimensionnel de taille (M, N, ndim), où `ndim` est 
        le nombre de dimensions du paramètre.
    
        Parameters
        ----------
        M : int, optional
            Le nombre de chaînes indépendantes à générer. Par défaut, M=10.
        N : int, optional
            Le nombre d'échantillons à générer par chaîne. Par défaut, N=10000.
                
        Return
        ------
        None
            Cette méthode ne retourne rien. Les chaînes générées sont enregistrées dans l'attribut `self.sampler_theta` sous forme d'un tableau NumPy.
        """
        chains = []
        start_time = time.time()
        print('\n')
        
        for m in range(M):
            self.theta_current = None
            self.theta_new = self.theta_new + self.psigma * np.random.randn(self.ndim)
            samples = []
    
            for n in range(N):
                _time = time.time() - start_time
                minutes = int(_time // 60)
                seconds = int(_time % 60)
                print(f"\rProgression : {int((m*N+n+1)/(M*N)*100)}% [{minutes:02d}:{seconds:02d}]", end='', flush=True)
                log_alpha = self.__log_acceptance()
                theta_current = self.__accept_or_reject(log_alpha)
                samples.append(theta_current)
                
            chains.append(samples)
        self.sampler_theta = np.array(chains)

    def PlotMCMC(self):
        """
        Affiche les évolutions des paramètres du modèle durant l'échantillonnage MCMC.
        
        Description
        -----------
        Cette méthode génère deux graphiques montrant l'évolution des paramètres `rho0` et `rp` au cours des 
        itérations de l'échantillonnage MCMC.
        
        Parameters
        ----------
            None
            Cette méthode ne prend pas de paramètre.
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """
        if self.multi_chain:
            nchains = self.sampler_theta.shape[0]
            
            for chain in range(nchains):
                rho0_samples = self.sampler_theta[chain, :, 0]
                plt.plot(rho0_samples, label=f"Walker {chain+1}")
            plt.xlabel('Pas du MCMC')
            plt.ylabel('$\\rho_0$')
            plt.title('Évolution de $\\rho_0$')
            plt.legend()
            plt.show()
    
            for chain in range(nchains):
                rp_samples = self.sampler_theta[chain, :, 1]
                rho0_samples = self.sampler_theta[chain, :, 0]
                plt.plot(rp_samples, rho0_samples, label=f"Walker {chain+1}")
            plt.xlabel('$r_p$')
            plt.ylabel('$\\rho_0$')
            plt.title('Évolution de $\\rho_0$ en fonction de $r_p$')
            plt.legend()
            plt.show()
            
        else:
            rho0_samples = self.sampler_theta[:, 0]
            plt.plot(rho0_samples)
            plt.xlabel('Pas du MCMC')
            plt.ylabel('$\\rho_0$')
            plt.title('Évolution de $\\rho_0$')
            plt.show()
    
            rp_samples = self.sampler_theta[:, 1]
            plt.plot(rp_samples, rho0_samples)
            plt.xlabel('$r_p$')
            plt.ylabel('$\\rho_0$')
            plt.title('Évolution de $\\rho_0$ en fonction de $r_p$')
            plt.show()
    
    def __log_probability(self, theta=None):
        """
        Calcule la probabilité totale du modèle, combinant a priori et vraisemblance.
        
        Description
        -----------
        Cette méthode calcule la probabilité totale en combinant la log-probabilité a priori et la log-vraisemblance 
        du modèle.
        
        Parameters
        ----------
        theta : numpy.ndarray, optional
            Les paramètres pour lesquels calculer la probabilité totale. Si `None`, utilise `theta_new`.        
        
        Return
        ------
        float
            La probabilité totale du modèle pour les paramètres `theta`.
        """
        lp = self.__log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.__log_likelihood(theta)
        le = np.log(np.sum(np.exp(ll)+np.exp(lp)))
        return ll + lp - le
    
    def __mcmc_emcee(self):
        """
        Effectue un échantillonnage MCMC en utilisant l'échantillonneur emcee.
        
        Description
        -----------
        Cette méthode utilise l'échantillonneur `EnsembleSampler` de la bibliothèque `emcee` pour effectuer l'échantillonnage 
        MCMC. Elle initialise les positions des "walkers" et lance l'échantillonnage.
        
        Parameters
        ----------
            None
            Cette méthode ne prend pas de paramètre.
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """
        self.ndim = len(self.theta_new)
        self.nwalkers = 10
        self.nech = 10000
        init_pos = self.theta_new + self.psigma * np.random.randn(self.nwalkers, self.ndim)
        sampler = emc.EnsembleSampler(self.nwalkers, self.ndim, self.__log_probability)
        sampler.run_mcmc(init_pos, self.nech, progress=True)
        self.sampler = sampler
        
        #plt.plot(self.sampler_chain[:,0,0])
        #plt.xlabel('Pas du MCMC')
        #plt.ylabel('$\\rho_0$')
        #plt.title('Évolution de $\\rho_0$')
        #plt.show()
        
    def __autocorr(self):
        """
        Calcule la fonction d'autocorrélation pour chaque "walker" et chaque paramètre.
        
        Description
        -----------
        Cette méthode calcule la fonction d'autocorrélation pour chaque "walker" dans la chaîne MCMC et pour chaque paramètre.
        
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """
        self.autocorr_func = np.zeros((self.nwalkers, self.ndim, self.nech))
        test_autocorr = self.sampler.get_chain()

        for walker in range(self.nwalkers):
            for param in range(self.ndim):
                x = test_autocorr[:, walker, param]
                autocorr = emc.autocorr.function_1d(x)
                self.autocorr_func[walker, param, :len(autocorr)] = autocorr
    
    def PlotAutoCorr(self):
        """
        Affiche les fonctions d'autocorrélation pour chaque paramètre en fonction du nombre de pas.
        
        Description
        -----------
        Cette méthode génère un graphique pour chaque paramètre montrant l'autocorrélation en fonction des pas d'échantillonnage.
        
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """

        pas = np.arange(self.nech)
        self.pas_inde = np.zeros((self.nwalkers, self.ndim))
        seuil = 0.1

        for param in range(self.ndim):
            for walker in range(self.nwalkers):
                autocorr = self.autocorr_func[walker, param, :]
                plt.plot(pas, autocorr, label=f"Walker {walker+1}")
                self.pas_inde[walker, param] = np.where(np.abs(autocorr) < seuil)[0][0]
            
            plt.axhline(seuil, color='red', linestyle='--', label="Seuil d'independance")
            plt.title(f"Paramètre {param+1}")
            plt.xlabel("Pas")
            plt.ylabel("Autocorrélation")
            plt.xscale('log')
            plt.legend()
            plt.show()
        
    def __convergence(self):
        """
        Calcule et affiche la convergence de chaque paramètre dans les chaînes MCMC.
        
        Description
        -----------
        Cette méthode évalue la convergence des chaînes MCMC en calculant le critère de Gelman-Rubin pour chaque paramètre.
        
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """

        test_conv = self.sampler.get_chain()
        N = test_conv.shape[0]
        M = test_conv.shape[1]
        mean_chain = np.mean(test_conv, axis=0)
        mean_all = np.mean(mean_chain, axis=0)
        B = 1 / (M-1) * np.sum((mean_chain-mean_all)**2, axis=0)
        W_ = 1 / (N-1) * np.sum((test_conv-mean_chain)**2, axis=0)
        W = 1 / M * np.sum(W_, axis=0)
        R1 = (N-1)/N * W
        R2 = (M+1)/M * B
        self.conv = (R1 + R2)/W

        print("\nConvergence sur les prametres :")
        print(f"     p0 : {self.conv[0]}")
        print(f"     rp : {self.conv[1]}")
        print(f"      A : {self.conv[2]}")
        print(f"     mu : {self.conv[3]}")
        print(f"  sigma : {self.conv[4]}")
        
        if np.all(self.conv < 1.03):
            print("Tout les parametres ont converges.")
            
        else:
            print("Un ou plusieurs parametres n'ont pas converges.")

    def __selection(self):
        """
        Effectue un sous-échantillonnage des chaînes MCMC en fonction de la période de "burn-in" et de l'indépendance.
        
        Description
        -----------
        Cette méthode supprime les premiers échantillons (période de burn-in) et applique un sous-échantillonnage 
        pour obtenir des échantillons indépendants.
        
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """

        self.burn_in = self.nech // 10
        self.sampler_chain = self.sampler.get_chain(discard=self.burn_in, thin=int(np.max(self.pas_inde)), flat=True)

    def GetDist(self):
        """
        Affiche un graphique des distributions des paramètres à partir des échantillons MCMC.
        
        Description
        -----------
        Cette méthode utilise la bibliothèque `getdist` pour générer un graphique des distributions des paramètres 
        obtenus après le sous-échantillonnage de la chaîne MCMC.
        
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        """

        plt.close('all') # Probleme de gestion entre matplotlib et getdist
        names = ['rho_0', 'rp', 'A', 'mu', 'sigma']
        labels = ['\\rho_0', 'r_p', 'A', '\\mu', '\\sigma']
        mcsamples = MCSamples(samples=self.sampler_chain, names=names, labels=labels)
        g = plots.get_subplot_plotter()
        g.triangle_plot(mcsamples, filled=True, legend_labels = ['MCMC (emcee)'])
        plt.show()
        
    def SuperposeDist(self):
        """
        Superpose les distributions des paramètres obtenues par MCMC et par minimisation du $\chi^2$.
        
        Description
        -----------
        Cette méthode utilise la bibliothèque `getdist` pour tracer un diagramme en triangle 
        des distributions marginales des paramètres. Les distributions obtenues par l'échantillonnage 
        MCMC (via `self.sampler_chain`) sont comparées à celles obtenues par un tirage simulé autour 
        des valeurs minimisant le $\chi^2$ (via `self.data_fit._tirage()`). Les deux distributions 
        sont superposées et affichées sur le même graphique avec des légendes explicites.
        
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien. Le graphique est affiché à l'écran.
        """
        plt.close('all') # Probleme de gestion entre matplotlib et getdist
        tirage = self.data_fit._tirage()
        names = ['rho_0', 'rp', 'A', 'mu', 'sigma']
        labels = ['\\rho_0', 'r_p', 'A', '\\mu', '\\sigma']
        sample_mcmc = MCSamples(samples=self.sampler_chain, names=names, labels=labels)
        sample_chisq = MCSamples(samples=tirage, names=names, labels=labels)
        g = plots.get_subplot_plotter()
        g.triangle_plot([sample_mcmc, sample_chisq], filled=True, legend_labels = ['MCMC (emcee)', r'Minimisation du $\chi^2$'])
        plt.show()