import os
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class TACKineticModel():
    """
    Attributes:

        comp_names : [Lumen, Liver, blood, Urine, Metabolites]
            names of pharmacokinetic compartments

        n_comps : 5
            number of compartments in model

        rates : [ABCB1, ABCG2, ABCC2, gamma, SLCO2B1, biliary_Cl, bact_metabolism, delta, renal_Cl]
            kinetic rates of SASP movement through model

            ABCG2 : BCRP efflux
                Lumen -> Liver
            ABCC2 : MRP2 efflux
                Lumen -> Liver
            gamma : absorption 
                Lumen -> Liver
            SLCO2B1 : OATP2B1 influx
                Lumen -> Liver
            biliary_Cl : biliary clearance 
                Liver -> Lumen
            bact_metabolism : bacterial intestinal metabolism of SASP into metabolites(5-ASA, SP)
                Lumen -> out
            delta : first pass distribution
                Liver -> blood
            renal_Cl : renal clearance
                blood -> Urine 

        exp_blood_conc : [0, 10.04, 25.1, 46.7017, 55.7377, 50.9373, 39.0776, 29.9719, 19.3113, 14.7933, 9.4986, 5.6161, 3.2858, 1.6064]
            blood concentrations of SASP from Azadkhan et al. (micromolar)

        init_comp_conc : [0.00879, 0.0, 0.0, 0.0, 0.0]
            inital amounts of SASP in each compartment (moles)
        
        times : [0, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48]
            times of measurement from Wong et al. 
    """

    def __init__(self):
        # Initialize attributes
        self.n_comps = 0
        self.rate_names = ['ABCB1',
                           'ABCC2', 
                           'alpha', 
                           'CYP3A4', 
                           'CYP3A5',
                           'clearance']
        
        self.rates = np.zeros(len(self.rate_names))

        self.exp_blood_conc = np.array([6, 20.6, 18.1 ,11, 8.6, 7.5, 6]) # ng/mL
        self.exp_blood_conc_std = np.array([1.3, 8.3, 4.7, 2.4, 2, 1.6, 1]) # ng/mL
        self.init_comp_conc = np.array([5.643e6, 0.0]) # ng
        self.times = np.array([0, 2, 4, 6, 8, 10, 12]) # hours
#         self.font = {'family' : 'normal',
#                     'weight' : 'bold',
#                     'size'   : 22}
#         matplotlib.rc('font', **self.font)
        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({'font.family':'Times New Roman'})
        plt.rcParams.update({'axes.linewidth': 1})

    """
    PLOTTING METHODS
    """

    # Plot data from Wong et al. 
    def plot_exp_data(self):
        plt.plot(self.times, self.exp_blood_conc, 'o', c='k', label='Wong et al.', lw=10)
        plt.errorbar(self.times, self.exp_blood_conc, yerr=self.exp_blood_conc_std, c='k', capsize=3 )
        plt.ylabel('$\it{TAC Blood Concentration}$ [\u00B5M]')
        plt.xlabel('$\it{Time}$ [h]')  
        plt.legend(loc='upper right')
        plt.show()

    # Plot rates
    def plot_rates(self):
        x=[i for i in range(len(self.rates))]
        plt.bar(x, height=self.rates, color='k')
        plt.xticks(x, self.rate_names, rotation=45)
        for x, y in zip(x,self.rates):
            plt.text(x-.3,y+.005, str(round(y,3)))
        plt.show()


    # Plot model
    def plot_model(self, title=''):
        # Add axes object if not provided
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        # Create smooth curve
        x, curve = self._get_curve(self.model(self.rates, comp_no=1))
        ax.plot(x, curve, label='optimized\nmodel', ls='dashed', color='k')
        ax.plot(self.times, self.exp_blood_conc, 'o', c='k', label='Wong et al.')
        ax.errorbar(self.times, self.exp_blood_conc, yerr=self.exp_blood_conc_std, ls='None', c='k', capsize=3)
        ax.set_ylabel('$\it{TAC Blood Concentration}$ [ng/mL]')
        ax.set_xlabel('$\it{Time}$ [h]')  
        ax.legend(bbox_to_anchor=(1,1))
        ax.set_title(title)

    # Get smooth curve
    def _get_curve(self, model):
        """
        model: directly call self.model()
        """
        
        xy_spline = make_interp_spline(self.times, model)
        x = np.linspace(self.times.min(), self.times.max(), 500)
        curve = xy_spline(x)

        return [x, curve]
    
    # Plot knockouts
    def _plot_knockout(self, knockout_rates, name):
        fig, ax = plt.subplots(figsize=(3.35,2))

        x, curve = self._get_curve(self.model(self.rates, comp_no=1))
        ax.plot(x, curve, label='model', color='k')
        ax.plot(self.times, self.exp_blood_conc, 'o', c='k', label='Wong et al.')
        ax.errorbar(self.times, self.exp_blood_conc, yerr=self.exp_blood_conc_std, ls='None', capsize=3, c='k',)
        x, knockout_curve = self._get_curve(self.model(knockout_rates, comp_no=1))
        ax.plot(x, knockout_curve, label='knockout', color='k', ls='dashed')
        ax.set_ylabel('$\it{TAC Blood}$\n$\it{Concentration}$ \n[ng/mL]')
        ax.set_xlabel('$\it{Time}$ [h]')  
#         ax.legend(bbox_to_anchor=(1,1))
        fig.set_size_inches(3.35, 2)
        fig.tight_layout()
        plt.show()


        # Peak conc. fold change
        max_wt = curve.max()
        max_ko = knockout_curve.max()
        fold_change = np.abs(max_wt - max_ko) / max_wt
        print(f'PEAK CONC. FOLD CHANGE: {fold_change}')

        # AUC fold change
        def reimann_sum(x, curve):
            return np.sum([curve[i]*(x[i]-x[i-1]) for i in range(len(curve))])
    
        # Integrate curve 
        auc_wt = reimann_sum(x, curve)
        auc_mut = reimann_sum(x, knockout_curve)

        # Report fold change
        fold_change = np.abs(auc_wt - auc_mut) / auc_wt
        print(f'AUC FOLD CHANGE: {fold_change}')

    # Plot bact model
    def _plot_bact_model(self, ax, means, stds, bact, condition, title, plot_model=True):

        # Parameters
        if condition == 'aerobic':
            color = 'red'
        else:
            color = 'blue'
                
        # Plot optimized model
        if plot_model:
            x, curve = self._get_curve(self.model(self.rates, comp_no=1))
            ax.plot(x, curve, color='k')#, label='model')
        else:
            x, curve = self._get_curve(self.model(self.rates, comp_no=1))
            ax.plot(x, curve, color='k')
        opt_max = curve.max()

        # Plot bact model
        spline = make_interp_spline(self.times, means)
        x = np.linspace(self.times.min(), self.times.max(), 500)
        curve = spline(x)
        bact_max = curve.max()
        ax.plot(x, curve, color=color, ls='dashed')#, label=condition)
        ax.errorbar(self.times, means, yerr=stds, color=color, ls='None', ecolor=color, capsize=3)
        if title:
            ax.set_title(f'{bact} {condition}')
            
        print(f'PEAK CONC. FOLD CHANGE: {np.abs(bact_max - opt_max) / opt_max} for BACTERIA: {bact} and CONDITION: {condition}')

    
    """
    COMPARTMENTAL MODEL
    """

    # System of differential equations that describe compartmental model
    def model(self, p, comp_no=1):
        """
        p :  parameters in shape [ABCB1, ABCC2, Alpha, CYP3A4, metabolism, clearance]
        comp_no: indice of compartment to calculate concentration over time 
            0 : Lumen
            1 : Blood
        """

        def odes(y, t, p):
            """
            y: moles of SASP in each compartment
            """

            # Lumen DiffEq
            dldt = (p[0]+p[1])*y[1] - (p[2]+p[3]+p[4])*y[0]

            # Blood DiffEq
            dbdt = p[2]*y[0] - (p[0]+p[1]+p[5])*y[1]

            return [dldt, dbdt]

        # Return the solutions for one compartment 
        def get_comp_conc(y, comp_no):
            # Get concentration
            comp_conc = np.empty(len(y))
            for i, conc_t in enumerate(y):
                comp_conc[i] = conc_t[comp_no]
            return comp_conc

        # Get numerical solutions to ODEs
        sol = odeint(odes, t=self.times, y0=self.init_comp_conc, args=tuple([p]))

        return get_comp_conc(sol, comp_no)

    """
    OPTIMIZATION
    """

    # Objective function
    def _obj(self, p):
        return np.sum((self.model(p, comp_no=1) - self.exp_blood_conc)**2)

    # Optimization powered by scipy.optimize.minimize
    def optimize(self, init_rates):
        res = minimize(self._obj, x0=init_rates, bounds = [(0, np.inf) for i in range(len(init_rates))])
        self.rates = res.x
        self.blood_conc = self.model(self.rates, comp_no=1)
        self.lumen_conc = self.model(self.rates, comp_no=0)

        self.plot_rates()

    """
    VALIDATION
    """

    #Knockout method
    def knockout(self, rates=[0], name=None):
        """
        rates: array of rate indices to knockout, indices can be found in self.rates_names
        """

        # Make temp rates
        knockout_rates = [r for r in self.rates]
        for r in rates:
            knockout_rates[r] = 0
            
           
        if name == None:
            name = ' '.join(self.rate_names[i] for i in rates) + ' knockout'

        self._plot_knockout(knockout_rates, name)

    """
    BACTERIA INFLUENCE
    """

    # Model bacterial conditions

    def model_bact(self, title: bool=False):

        # Parameters from Wang et al. 

        # Experimental Values
        mrp2_Ecoli_minus = [-4.50333, .946]
        mrp2_Ecoli_plus = [0.921969391, .813]
        mrp2_Bif_minus = [-6.30333333, 1.819]
        mrp2_Bif_plus = [-.008977557, .280]

        cyp3a4_Ecoli_minus = [4.30756439, 0.151]
        cyp3a4_Ecoli_plus = [1.682, .507]
        cyp3a4_Bif_minus = [3.90118617, .210]
        cyp3a4_Bif_plus = [0.949, .493]


        # Distributions 
        cp3a4_Ecoli_minus_dist = np.random.normal(cyp3a4_Ecoli_minus[0], cyp3a4_Ecoli_minus[1], size=(1000))
        cp3a4_Ecoli_plus_dist = np.random.normal(cyp3a4_Ecoli_plus[0], cyp3a4_Ecoli_plus[1], size=(1000))
        cp3a4_Bif_minus_dist = np.random.normal(cyp3a4_Bif_minus[0], cyp3a4_Bif_minus[1], size=(1000))
        cp3a4_Bif_plus_dist = np.random.normal(cyp3a4_Bif_plus[0], cyp3a4_Bif_plus[1], size=(1000))
        mrp2_Ecoli_minus_dist = np.random.normal(mrp2_Ecoli_minus[0], mrp2_Ecoli_minus[1], size=(1000))
        mrp2_Ecoli_plus_dist = np.random.normal(mrp2_Ecoli_plus[0], mrp2_Ecoli_plus[1], size=(1000))
        mrp2_Bif_minus_dist = np.random.normal(mrp2_Bif_minus[0], mrp2_Bif_minus[1], size=(1000))
        mrp2_Bif_plus_dist = np.random.normal(mrp2_Bif_plus[0], mrp2_Bif_plus[1], size=(1000))
   

        fig, axs = plt.subplots(nrows=2, figsize=(3.35,4), layout='constrained')

        means, stds, new_params = self._simulate(cp3a4_Ecoli_minus_dist, mrp2_Ecoli_minus_dist)
        ax = axs[0]
        self._plot_bact_model(ax, means, stds, bact = 'E. coli Nissle 1917',condition = 'aerobic', title=title)
        print(f'Simulated EColi. aerobic:\nMRP2\n\t\tmean: {new_params[1].mean()}\n\t\tstd: {new_params[1].std()}')
        print(f'\nCYP3A4\n\t\tmean: {new_params[3].mean()}\n\t\tstd: {new_params[3].std()}')
        
        means, stds, new_params = self._simulate(cp3a4_Ecoli_plus_dist, mrp2_Ecoli_plus_dist)
        ax = axs[0]
        self._plot_bact_model(ax, means, stds, bact = 'E. coli Nissle 1917', condition = 'anaerobic', title=title, plot_model=False)
        print(f'Simulated EColi. aerobic:\nMRP2\n\t\tmean: {new_params[1].mean()}\n\t\tstd: {new_params[1].std()}')
        print(f'\nCYP3A4\n\t\tmean: {new_params[3].mean()}\n\t\tstd: {new_params[3].std()}')
        
        means, stds, new_params = self._simulate(cp3a4_Bif_minus_dist, mrp2_Bif_minus_dist)
        ax = axs[1]
        self._plot_bact_model(ax, means, stds, bact = 'Bifidobacterium adolescentis', condition = 'aerobic', title=title, plot_model=False)
        ax.plot([0], [0], color='b', ls='dashed')#, label='anaerobic')
        print(f'Simulated EColi. aerobic:\nMRP2\n\t\tmean: {new_params[1].mean()}\n\t\tstd: {new_params[1].std()}')
        print(f'\nCYP3A4\n\t\tmean: {new_params[3].mean()}\n\t\tstd: {new_params[3].std()}')
        
        means, stds, new_params = self._simulate(cp3a4_Bif_plus_dist, mrp2_Bif_plus_dist)
        ax = axs[1]
        self._plot_bact_model(ax, means, stds, bact = 'Bifidobacterium adolescentis', condition = 'anaerobic', title=title, plot_model=False)
        print(f'Simulated EColi. aerobic:\nMRP2\n\t\tmean: {new_params[1].mean()}\n\t\tstd: {new_params[1].std()}')
        print(f'\nCYP3A4\n\t\tmean: {new_params[3].mean()}\n\t\tstd: {new_params[3].std()}')

        for ax in axs.flatten():
            ax.plot(self.times, self.exp_blood_conc, 'o', c='k')#, label = 'Wong et al.')
            ax.errorbar(self.times, self.exp_blood_conc, yerr=self.exp_blood_conc_std, ls='None', c='k', capsize=3)
            ax.set_xlabel('$\it{Time}$ [h]')
            ax.set_ylabel('$\it{TAC Blood}$\n$\it{Concentration}$ \n[ng/mL]')
            ax.label_outer()
        
        axs[0].legend(handles=[], labels=[], title='n = 1000')
        fig.set_size_inches(3.5, 4)
        fig.tight_layout()


    def _simulate(self, cyp3a4, mrp2):
        timeseries = np.empty((len(cyp3a4), len(self.times)))
        exp_parameters = np.empty((len(self.rates), len(cyp3a4)))
        for i, (cyp3a4_exp, mrp2_exp) in enumerate(zip(cyp3a4, mrp2)):
            exp_para = [r for r in self.rates]
            exp_para[1] *= 2**mrp2_exp
            exp_para[3] *= 2**cyp3a4_exp
            exp_parameters[:,i] = exp_para
            timeseries[i] = self.model(exp_para, 1)

        means = np.mean(timeseries, axis=0)
        stds = np.std(timeseries, axis=0)

        return [means, stds, exp_parameters]



















