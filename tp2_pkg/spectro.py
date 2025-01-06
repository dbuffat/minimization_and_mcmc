import emcee
import math
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from getdist import plots, MCSamples
from matplotlib.gridspec import GridSpec
from emcee.autocorr import function_1d

version = "1.0"

class SpectroAnalysis:
    
    """
    Classe pour l'analyse spectroscopique d'un spectre de galaxie. Cette classe inclut des fonctionnalités pour 
    la lecture des données, le traitement de bruit, l'ajustement d'un modèle gaussien avec un continuum linéaire, 
    et l'analyse des chaînes MCMC.

    Attributes
    ----------
    filename : String
        Chemin vers le fichier contenant les données spectroscopiques.
    data : pandas.DataFrame
        Données spectroscopiques chargées depuis le fichier.
    noise_signal : pandas.DataFrame
        Sous-ensemble des données représentant la région du bruit.
    lin_fit_params : tuple
        Paramètres de l'ajustement linéaire sur le bruit (pente et intercept).
    data_centered : pandas.DataFrame
        Données après soustraction du modèle linéaire.
    noise_signal_centered : pandas.DataFrame
        Données de bruit centrées après soustraction du modèle linéaire.
    core_data : pandas.DataFrame
        Sous-ensemble des données représentant la région centrale contenant les pics OIII.
    fit_params : numpy.ndarray
        Paramètres du modèle gaussien ajusté aux données centrales.
    theta_origin : numpy.ndarray
        Meilleurs paramètres initiaux pour MCMC, issus de l'ajustement initial.
    theta_new : numpy.ndarray
        Meilleurs paramètres finaux obtenus après les analyses MCMC.
    thin : int
        Largeur de corrélation utilisée pour le sous-échantillonnage des chaînes MCMC.
    """
    
    def __init__(self):
        
        '''
        Description
        -----------
        Méthode d'initialisation de la classe.
        Charge le fichier de données, initialise les paramètres et lance l'analyse complète.
        
        Parameters
        ----------
        filename : String
            Chemin du fichier avec les données de spectrométrie.
                    
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        self.filename = 'tp2_pkg/f1245_888.dat'
        self.data = None
        self.noise_signal = None
        self.lin_fit_params = None
        self.data_centered = None
        self.noise_signal_centered = None
        self.core_data = None
        self.fit_params = None
        self.theta_origin = None
        self.theta_new = None
        self.thin = 70

    def run_analysis(self):
        '''
        Description
        -----------
        Coeur du programme, effectue l'analyse complète : lecture des données, ajustements, traitement du bruit,
        et analyse MCMC.
            
        Parameters
        ----------
            None
    
        Return
        ------
        numpy.ndarray
            Flux ajusté à partir du modèle gaussien.
        '''
        self.read_csv()
        self.noise_signal = self.selection(self.data, 6600)
        self.fit_linear_model()
        self.data_centered = self.data.copy()
        self.subtract_linear_model(self.data_centered)
        self.noise_signal_centered = self.noise_signal.copy()
        self.subtract_linear_model(self.noise_signal_centered)
        self.core_data = self.selection(self.data, 6350, 6600)
        fitted_flux = self.fit_OIII()
        self.plot_spectra()
        self.theta_origin = self.fit_params
        sampler = self.run_mcmc()
        self.analyze_chains(sampler)
        self.error()
        return fitted_flux
    
    def read_csv(self):
        
        '''
        Description
        -----------
        Lit les données spectroscopiques depuis un fichier CSV 
        et les stocke dans un DataFrame `self.data`.
            
        Parameters
        ----------
            None

        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        
        try:
            self.data = pd.read_csv(
                self.filename, delimiter=",", names=["Wavelength", "Flux"], skiprows=1
            )
        except FileNotFoundError:
            raise ValueError(f"File '{self.filename}' not found.")

    def selection(self, type_data, lambda_min, lambda_max=7274.57784357302):
        '''        
        Description
        -----------
        Sélectionne un sous-ensemble des données dans la plage donnée 
        de longueurs d'onde `[lambda_min, lambda_max]`.
            
        Parameters
        ----------
        type_data : pandas.DataFrame
            Ensemble de données spectroscopiques.
        lambda_min : float
            Longueur d'onde minimale pour la sélection.
        lambda_max : float, optionnel
            Longueur d'onde maximale pour la sélection, par défaut 7274.57784357302.
                
        Return
        ------
        pandas.DataFrame
            Données filtrées dans la plage donnée.
        '''
        return type_data[
            (type_data["Wavelength"] > lambda_min) & (type_data["Wavelength"] <= lambda_max)
        ]

    @staticmethod
    def linear_model(x, a, b):
        '''
        Description
        -----------
            Définit un modèle linéaire sous la forme `y = ax + b`.
    
        Parameters
        ----------
        x : numpy.ndarray
            Tableau des abscisses (longueurs d'onde).
        a : float
            Pente du modèle linéaire.
        b : float
            Ordonnée à l'origine du modèle linéaire.    
        
        Return
        ------
        numpy.ndarray
            Valeurs de `y` correspondant aux valeurs de `x`.
        '''
        return a * x + b

    def fit_linear_model(self):
        '''
        Description
        -----------
        Ajuste un modèle linéaire sur les données de bruit et stocke
        les paramètres ajustés dans `self.lin_fit_params`.
        
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        x_noise = self.noise_signal["Wavelength"].values
        y_noise = self.noise_signal["Flux"].values
        popt, _ = curve_fit(self.linear_model, x_noise, y_noise)
        self.lin_fit_params = popt
        print(f"Linear fit completed: a = {popt[0]}, b = {popt[1]}")

    def subtract_linear_model(self, dataset):
        '''
        Description
        -----------
        Soustrait le modèle linéaire défini par `self.lin_fit_params`
        des flux du dataset donné.
        
        Parameters
        ----------
        dataset : pandas.DataFrame
            Données auxquelles soustraire le modèle linéaire.   
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        a, b = self.lin_fit_params
        dataset["Flux"] -= (a * dataset["Wavelength"] + b)

    def calculate_std_dev(self):
        '''
        Description
        -----------
        Calcule l'écart type des flux dans les données de bruit centrées.
        
        Parameters
        ----------
            None
    
        Return
        ------
            float
                Écart type des flux.
        '''
        return np.std(self.noise_signal_centered["Flux"])

    def gaussian_model(self, wavelength, A, sigma_g, z, a, b):
        '''
        Description
        -----------
        Définit un modèle avec deux pics gaussiens correspondant aux 
        raies OIII (4959 et 5007 Å) sur un continuum linéaire.
        
        Parameters
        ----------
        wavelength : numpy.ndarray
            Tableau des longueurs d'onde.
        A : float
            Amplitude du premier pic gaussien.
        sigma_g : float
            Largeur à mi-hauteur des pics gaussiens.
        z : float
            Redshift.
        a : float
            Pente du continuum linéaire.
        b : float
            Ordonnée à l'origine du continuum linéaire.
    
        Return
        ------
        numpy.ndarray
            Valeurs du modèle pour les longueurs d'onde données.
        '''
        mu1 = 4959 * (1 + z)
        mu2 = 5007 * (1 + z)
        continuum = a * wavelength + b
        gaussian1 = (A / 3) * np.exp(-0.5 * ((wavelength - mu1) / sigma_g) ** 2)
        gaussian2 = A * np.exp(-0.5 * ((wavelength - mu2) / sigma_g) ** 2)
        return continuum + gaussian1 + gaussian2

    def fit_OIII(self):
        '''
        Description
        -----------
        Ajuste un modèle gaussien avec un continuum linéaire sur les pics OIII
        et stocke les paramètres ajustés dans `self.fit_params`.
    
        Parameters
        ----------
            None
    
        Return
        ------
        numpy.ndarray
            Flux ajusté à partir du modèle.
        '''
        wavelengths = self.core_data["Wavelength"]
        fluxes = self.core_data["Flux"]
        sigma = np.full_like(fluxes, self.calculate_std_dev())
        initial_params = [9.0e-16, 20, 0.3, *self.lin_fit_params]
        popt, _ = curve_fit(self.gaussian_model, wavelengths, fluxes, p0=initial_params, sigma=sigma)
        self.fit_params = popt
        print(popt)
        print(f"La valeur du redshift est : {popt[2]}")
        return self.gaussian_model(wavelengths, *popt)    

    def plot_data(self, title, x, y, x_label, y_label, labels, colors, styles=None):
        '''
        Description
        -----------
        Génère un graphique personnalisé avec les courbes données.
    
        Parameters
        ----------
        title : String
            Titre du graphique.
        x : list of numpy.ndarray
            Liste des tableaux pour les abscisses.
        y : list of numpy.ndarray
            Liste des tableaux pour les ordonnées.
        x_label : String
            Nom de l'axe des abscisses.
        y_label : String
            Nom de l'axe des ordonnées.
        labels : list of String
            Liste des légendes pour chaque courbe.
        colors : list of String
            Liste des couleurs pour chaque courbe.
        styles : list of String, optionnel
            Liste des styles de ligne pour chaque courbe. Le défaut est None.
    
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        plt.figure(figsize=(10, 6))
        for i in range(len(x)):
            style = styles[i] if styles else "-"
            plt.plot(x[i], y[i], style, label=labels[i], color=colors[i])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid()
        plt.show()

    def plot_spectra(self):
        '''
        Description
        -----------
        Produit un ensemble de sous-graphiques pour visualiser le spectre, 
        le bruit, les données centrées et l'ajustement du modèle.
        
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot 1: Galaxy Spectrum
        axs1 = fig.add_subplot(gs[0, 0])
        axs1.plot(self.data["Wavelength"], self.data["Flux"], color="purple")
        axs1.set_title("Spectre de la Galaxie")
        axs1.set_xlabel("Longueur d'onde (Å)")
        axs1.set_ylabel("Flux")
        
        # Plot 2: Noise Spectrum with Linear Fit
        axs2 = fig.add_subplot(gs[0, 1])
        axs2.plot(self.noise_signal["Wavelength"], self.noise_signal["Flux"], color="blue", label="Spectre de bruit")
        axs2.plot(self.noise_signal["Wavelength"], self.linear_model(self.noise_signal["Wavelength"], *self.lin_fit_params), color="green", linestyle="--", label="Fit linéaire")
        axs2.set_title("Spectre du bruit de la galaxie avec un fit linéaire")
        axs2.set_xlabel("Longueur d'onde (Å)")
        axs2.set_ylabel("Flux")
        axs2.legend()
        
        # Plot 3: Centered Galaxy Spectrum
        axs3 = fig.add_subplot(gs[1, 0])
        axs3.plot(self.data_centered["Wavelength"], self.data_centered["Flux"], color="purple")
        axs3.set_title("Spectre de la galaxie centré")
        axs3.set_xlabel("Longueur d'onde (Å)")
        axs3.set_ylabel("Flux")
        
        # Plot 4: Centered Noise Spectrum
        axs4 = fig.add_subplot(gs[1, 1])
        axs4.plot(self.noise_signal_centered["Wavelength"], self.noise_signal_centered["Flux"], color="blue")
        axs4.set_title("Spectre du bruit centré")
        axs4.set_xlabel("Longueur d'onde (Å)")
        axs4.set_ylabel("Flux")
        
        # Plot 5: Galaxy Spectrum with Gaussian Fit (Occupy Full Width)
        fitted_flux = self.fit_OIII()
        axs5 = fig.add_subplot(gs[2, :])  # This will span the entire last row
        axs5.plot(self.core_data["Wavelength"], self.core_data["Flux"], color="purple", label="Spectre de l'émission OIII")
        axs5.plot(self.core_data["Wavelength"], fitted_flux, color="orange", linestyle="--", label="Fit du modèle")
        axs5.set_title("Raies d'émission OIII avec fit d'une double gaussienne")
        axs5.set_xlabel("Longueur d'onde (Å)")
        axs5.set_ylabel("Flux")
        axs5.legend()
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
    def log_prior(self, theta):
        '''
        Description
        -----------
        Calcule la probabilité a priori des paramètres selon des limites définies.
    
        Parameters
        ----------
        theta : numpy.ndarray
            Tableau des paramètres du modèle.
    
        Return
        ------
        float
            Logarithme de la probabilité a priori.
        '''
        cdt_A = theta[0] > 0 and theta[0] < 19e-16
        cdt_sigma_g = theta[1] > 0 and theta[1] < 6
        cdt_z = theta[2] > 0.25 and theta[2] < 0.35
        cdt_a = theta[3] > -1 and theta[3] < 1
        cdt_b = theta[4] > -1 and theta[4] < 1

        if cdt_A and cdt_sigma_g and cdt_z and cdt_a and cdt_b:
            return 0.0
        return -np.inf

    def log_likelihood(self, theta, wavelengths, fluxes, flux_errors):
        '''
        Description
        -----------
        Calcule la vraisemblance logarithmique pour un ensemble de paramètres donnés.
    
        Parameters
        ----------
        theta : numpy.ndarray
            Tableau des paramètres du modèle.
        wavelengths : numpy.ndarray
            Tableau des longueurs d'onde.
        fluxes : numpy.ndarray
            Tableau des flux observés.
        flux_errors : numpy.ndarray
            Tableau des erreurs sur les flux.
    
        Return
        ------
        float
            Logarithme de la vraisemblance.
        '''
        model_flux = self.gaussian_model(wavelengths, *theta)
        return -0.5 * np.sum(((fluxes - model_flux) / flux_errors) ** 2)

    def log_probability(self, theta, wavelengths, fluxes, flux_errors):
        '''
        Description
        -----------
        Calcule la probabilité totale (prior + likelihood) pour un ensemble de paramètres donnés.
    
        Parameters
        ----------
        theta : numpy.ndarray
            Tableau des paramètres du modèle.
        wavelengths : numpy.ndarray
            Tableau des longueurs d'onde.
        fluxes : numpy.ndarray
            Tableau des flux observés.
        flux_errors : numpy.ndarray
            Tableau des erreurs sur les flux.
    
        Return
        ------
        float
            Logarithme de la probabilité totale.
        '''
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, wavelengths, fluxes, flux_errors)
    
    def parameters_initialisation(self, best_param):
        '''
        Description
        -----------
        Initialise les positions des marcheurs pour le MCMC en ajoutant des variations 
        gaussiennes autour des meilleurs paramètres initiaux.
    
        Parameters
        ----------
        best_param : numpy.ndarray
            Paramètres initiaux optimaux obtenus par ajustement.
    
        Return
        ------
        tuple
            Position initiale des marcheurs, nombre de marcheurs, et nombre de dimensions.
        '''
        initial_params = best_param
        
        ndim = len(initial_params)
        n_walkers = 10 * ndim
        initial_positions = np.zeros((n_walkers, ndim))
        
        for i in range(ndim):
            scale = math.floor(math.log10(abs(initial_params[i]))) if initial_params[i] != 0 else 0
            initial_positions[:, i] = initial_params[i] + 1*10**(scale-3) * np.random.randn(n_walkers)
        
        return initial_positions, n_walkers, ndim
    
    def run_mcmc(self, n_steps= 10000):
        '''
        Description
        -----------
        Effectue deux étapes de MCMC pour ajustement des paramètres et 
        retourne les résultats.
    
        Parameters
        ----------
        n_steps : int, optionnel
            Nombre d'étapes pour le MCMC. Le défaut est 10000.
    
        Return
        ------
        tuple
            Sampler MCMC et échantillons aplatis après burn-in.
        '''
        
        best_params = self.theta_origin
        initial_positions, n_walkers, ndim = self.parameters_initialisation(best_params)        
        
        sampler = emcee.EnsembleSampler(
            n_walkers, 
            ndim, 
            self.log_probability,
            args=(self.core_data["Wavelength"].values,
                  self.core_data["Flux"].values,
                  self.calculate_std_dev()
                  )
        )

        print("Running first MCMC...")
        sampler.run_mcmc(initial_positions, n_steps, progress=True)

        flat_samples = sampler.get_chain(discard=n_steps//5, thin=self.thin, flat=True)
        log_prob = sampler.get_log_prob(discard=n_steps//5, thin=self.thin, flat=True)
        best_idx = np.argmax(log_prob) #reviens à minimiser le chi carré
        best_params = flat_samples[best_idx]
        
        print("Meilleurs lot de paramètres du MCMC_1:", best_params)
        
        ###############
        # SECOND MCMC #
        ###############
        
        print("Running second MCMC...")
        
        initial_positions, _, _ = self.parameters_initialisation(best_params)  
        
        sampler.reset()
        
        sampler.run_mcmc(initial_positions, n_steps, progress=True)
        
        flat_samples = sampler.get_chain(discard=n_steps//5, thin=self.thin, flat=True)
        log_prob = sampler.get_log_prob(discard=n_steps//5, thin=self.thin, flat=True)
        best_idx = np.argmax(log_prob)
        best_params = flat_samples[best_idx]

        print("Best parameters from MCMC:", best_params)

        return sampler

    def analyze_chains(self, sampler, labels=["A", "sigma_g", "z", "a", "b"]):
        '''
        Description
        -----------
        Analyse les chaînes MCMC : convergence, temps d'auto-corrélation, 
        et estimation des paramètres. Donne la valeur du redshift calculée.
    
        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            Objet contenant les chaînes MCMC.
        flat_samples : numpy.ndarray
            Échantillons aplatis après burn-in.
        labels : list of String, optionnel
            Noms des paramètres, par défaut ["A", "sigma_g", "z", "a", "b"].
    
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        # Obtenir les chaînes brutes et leurs dimensions
        chains = sampler.get_chain()
        n_steps, n_walkers, ndim = chains.shape
    

        
        # Étude de la convergence via R-hat
        az_data = az.from_emcee(sampler)
        rhat_values = az.rhat(az_data)
        print("Diagnostic de convergence (R-hat) :")
        for i, label in enumerate(labels):
            var_name = f"var_{i}"
            print(f"R-hat pour {label} : {rhat_values[var_name].values}")
        
        # Étude des temps d'auto-corrélation
        try:
            autocorr_times = sampler.get_autocorr_time()
            print("\nTemps d'auto-corrélation estimés :")
            for i in range(ndim):
                print(f"  {labels[i]} : {autocorr_times[i]:.2f}")
        except Exception as e:
            print(f"\nTemps d'auto-corrélation non disponibles : chaîne trop courte : {e}" )
            autocorr_times = None
        
        fig, axs = plt.subplots(ndim, 1, figsize=(8, 2 * ndim), sharex=True)
        fig.suptitle("Auto-correlation for Parameters", fontsize=16)
        
        t_z = autocorr_times[2]
        T_z= int(t_z)
        flat_samples = sampler.get_chain(discard=0, thin=T_z, flat=True)
        
        # Tracé des distributions et des corrélations
        plt.close('all')
        mc_samples = MCSamples(samples=flat_samples, names=labels, labels=labels)
        g = plots.get_subplot_plotter()
        g.triangle_plot(mc_samples, filled=True)
        plt.show()
        
        # Tracer la fonction d'auto-corrélation pour chaque paramètre sur échelle logarithmique
        fig, axs = plt.subplots(ndim, 1, figsize=(8, 2 * ndim), sharex=True)
        fig.suptitle("Fonction d'Auto-corrélation pour chaque paramètre", fontsize=16)
        
        for i in range(ndim):
            # Calculer la fonction d'auto-corrélation pour le paramètre i
            chains_1d = chains[:, :, i].flatten()
            autocorr = function_1d(chains_1d)
            
            # Tracer la fonction d'auto-corrélation en fonction du pas du MCMC
            axs[i].plot(np.arange(len(autocorr)), autocorr, label=f"Paramètre {labels[i]}")
            axs[i].set_title(f"Auto-corrélation pour {labels[i]}")
            axs[i].set_ylabel("ACF")
        
        axs[-1].set_xlabel("Lag (log scale)")
        axs[0].set_xscale('log')  # Appliquer l'échelle logarithmique à l'axe des x
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
        # Calcul du redshift z 
        print("=" * 56)             
        z_samples = flat_samples[:, 2]
        z_mean = np.mean(z_samples)
        z_std = np.std(z_samples)
        print(f"Redshift éstimé : {z_mean:.6f} +/- {z_std:.6f}")    
        print("=" * 56)
        
        self.z_mean = z_mean
        self.z_std = z_std
        
    def error(self):
        """
        Calcule l'erreur  des données par rapport à une valeur théorique donnée.

        Description
        -----------
        Cette méthode calcule l'erreur des données en comparant l'écart-type (`z_std`) 
        et la moyenne (`z_mean`) des valeurs observées avec une valeur de référence (`z_th`) et son 
        incertitude (`z_err_th`). Le résultat est imprimé avec une comparaison à la valeur théorique.

        Parameters
        ----------
            None
            Cette méthode ne prend pas de paramètre.

        Return
        ------
        None
            Cette méthode ne retourne rien mais imprime l'erreur calculée.
        """
        z_th = 0.296340
        
        err = np.abs(z_th - self.z_mean)
        
        print(f"L'erreur est : {err} par rapport à la valeur théorique : {z_th}")