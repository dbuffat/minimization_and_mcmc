import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from getdist import MCSamples, plots

version = "1.0"

class Minimisation:
    
    def __init__(self):
        '''
        Descrition
        ----------
            Initialise tout les attributs sur None et lance les calculs.
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        self.data = None
        self.cov = None
        self.inv_cov = None
        self.model = None
        self.popt = None
        self.pcov = None
        self.chisq = None
        self.chisq_red = None
        self.nof = None
        
        self.__compile_data()
        
    def __compile_data(self):
        '''
        Description
        -----------
            Methode prive qui lance les calculs preliminaires.
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        self.__load_data()
        self.__covariance()
        self.__inv_covariance()
        self.__fit()
        self._chisq()
        self._chisq_red()
        
    def Main(self):
        '''
        Description
        -----------
            Methode qui realise tout les print et l'affichage des graphiques.
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        #self.PrintData()
        self.PrintFit()
        self.PrintChisq()
        self.GetDist()
        
    def __load_data(self):
        '''
        Description
        -----------
            Methode prive. Charge les donnees .npz.
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        self.data = np.load('tp2_pkg/mydata_cluster.npz')

        
    def PrintData(self):
        '''
        Description
        -----------
            Trace les donnees.
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        plt.plot(self.data['r'], self.data['y'])
        plt.xscale('log')
        plt.show()

    def __bruit_blanc(self, n=300):
        '''
        Description
        -----------
            Methode privee. Genere un bruit blanc gaussien.

        Parameters
        ----------
            n : Integer, optional : the default is 300.
        
        Retourne
        --------
            Un array numpy de taille n.
        '''
        return np.random.normal(0, 1e-3, size=n)

    def __bruit_colore(self, n=300):
        '''
        Description
        -----------
            Methode privee . 
            Genere un bruit colore a partir d'un bruit blanc gaussien et de la PSD issue des donnees.'
            
        Parameters
        ----------
            n : Integer, optional : the default is 300.
            
        Retourne
        --------
            Un array numpy de taille n
        '''
        bb = np.fft.fft(self.__bruit_blanc(n))
        psd_sqrt = np.sqrt(self.data['psd'])
        bc = bb * psd_sqrt
        return np.real(np.fft.ifft(bc))

    def __covariance(self):
        '''
        Description
        -----------
            Methode prive. Evalue la matrice de covariance.
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        b = self.__bruit_colore()
        s = np.zeros((b.size, b.size))

        for _ in range(5000):
            b = self.__bruit_colore()
            s += np.outer(b, b)
        self.cov = s / 5000
        
        #plt.matshow(self.cov, cmap='viridis')
        #plt.colorbar()
        #plt.show()
    
    def __inv_covariance(self):
        '''
        Description
        -----------
            Methode privee. Calcul la matrice inverse de la matrice de covariance.
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        self.inv_cov = np.linalg.inv(self.cov)
    
    def _model(self, r, p0, rp, A, mu, sigma):
        '''
        Description
        -----------
            Methode protegee.
            Definition d'un modele mathematiques pour nos donnees.'

        Parameters
        ----------
        r : Numpy array
            Ensemble des r a appliquer sur ce modele.
        p0 : Float
            Amplitude de la fonction densite.
        rp : Float
            Rayon caracteristique de l'amas de galaxie etudie.
        A : Float
            Amplitude de la fonction gaussienne.
        mu : Float
            Valeur centree de la fonction gaussienne.
        sigma : Float
            Ecart-type de la fonction gaussienne.

        Retourne
        --------
            Un array numpy.
            Points du modele evaluee en r.
        '''
        a = 1.1
        b = 5.5
        c = 0.31
        g = A * np.exp(- (r-mu)**2 / sigma**2)
        rho = p0 / ((r/rp)**c * (1 + (r/rp)**a)**((b-c)/a))
        self.model = rho + g
        return self.model
    
    def __fit(self, p0=1, rp=1000, A=0.02, mu=1500, sigma=200):
        '''
        Description
        -----------
            Methode prive. Realise un fit des donnees a partir du modele.
            
        Parameters
        ----------
        p0 : Float, optional : the default is 1.
            Amplitude de la fonction densite.
        rp : Float, optional : the default is 1000.
            Rayon caracteristique de l'amas de galaxie etudie.
        A : Float, optional : the default is 0.02.
            Amplitude de la fonction gaussienne.
        mu : Float, optional : the default is 1500.
            Valeur centree de la fonction gaussienne.
        sigma : Float, optional : the default is 200.
            Ecart-type de la fonction gaussienne. 
                   
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        init_parameters = np.array([p0, rp, A, mu, sigma])
        x = self.data['r']
        y = self.data['y']
            
        self.popt, self.pcov = sp.optimize.curve_fit(self._model, x, y, p0 = init_parameters, bounds=(0, [np.inf, np.inf, 20, 1700, 2000]), sigma=self.cov)

        print("Valeurs obtenues par un fit :")
        print(f"     p0 = {self.popt[0]}")
        print(f"     rp = {self.popt[1]}")
        print(f"      A = {self.popt[2]}")
        print(f"     mu = {self.popt[3]}")
        print(f"  sigma = {self.popt[4]}")

        self._model(x, *self.popt)
        self.nof = len(y) - len(self.popt)
    
    def PrintFit(self):
        '''
        Description
        -----------
            Trace les donnees et le fit sur le meme graphique.
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        x = self.data['r']
        y = self.data['y']

        plt.plot(x, y)
        plt.plot(x, self.model)
        plt.xscale('log')
        plt.show()
    
    def _chisq(self):
        '''
        Description
        -----------
            Methode protegee. Calcul le $\\chi^2$.
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        y = self.data['y'] 
        residus = y - self.model
        self.chisq = residus.T @ self.inv_cov @ residus
        
    def _chisq_red(self):
        '''
        Description
        -----------
            Methode protegee. Calcul le $\\chi^2_{reduit}$.
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        self.chisq_red = self.chisq / self.nof
    
    def PrintChisq(self):
        '''
        Description
        -----------
            Affiche les valeurs du $\\chi^2$ et du $\\chi^2_{reduit}$
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        print(f"Chi2 = {self.chisq}")
        print(f"Chi2_reduit = {self.chisq_red}")
        
    def _tirage(self):
        '''
        Description
        -----------
            Methode protegee.
            On realise 1000 tirages de lot des 5 parametres du probleme.
            Chaque tirage est realise sur une loi normale, centree sur les valeurs du fit et d'ecart type la matrice de covariance calculer pendant le fit.'
        
        Parameters
        ----------
            None
        
        Retourne
        --------
            Un array numpy 
        '''
        return np.random.multivariate_normal(self.popt, self.pcov, size=1000)
        
    def GetDist(self):
        '''
        Description
        -----------
            Trace les courbes de confiance associee a nos ajustements.
            
        Parameters
        ----------
            None
        
        Return
        ------
        None
            Cette méthode ne retourne rien.
        '''
        plt.close('all') # Probleme de gestion entre matplotlib et getdist

        tirage = self._tirage()
        names = ['rho_0', 'rp', 'A', 'mu', 'sigma']
        labels = ['\\rho_0', 'r_p', 'A', '\\mu', '\\sigma']
        mcsamples = MCSamples(samples=tirage, names=names, labels=labels)
        g = plots.get_subplot_plotter()
        g.triangle_plot(mcsamples, filled=True, legend_labels = [r'Minimisation du $\chi^2$'])
        plt.show()
