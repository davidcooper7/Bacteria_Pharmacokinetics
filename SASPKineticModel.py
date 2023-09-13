import os
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np

class SASPKineticModel():
    """
    Attributes:

        comp_names : [Lumen, Liver, Plasma, Urine, Metabolites]
            names of pharmacokinetic compartments

        rates : [ABCG2, ABCC2, gamma, SLCO2B1, biliary_Cl, bact_metabolism, delta, renal_Cl]
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
                Liver -> Plasma
            renal_Cl : renal clearance
                Plasma -> Urine 

        exp_plasma_conc : [0, 10.04, 25.1, 46.7017, 55.7377, 50.9373, 39.0776, 29.9719, 19.3113, 14.7933, 9.4986, 5.6161, 3.2858, 1.6064]
            plasma concentrations of SASP from Azadkhan et al. (micromolar)

        init_comp_conc : [0.00879, 0.0, 0.0, 0.0, 0.0]
            inital amounts of SASP in each compartment (moles)
        
        times : [0, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48]
            times of measurement from Azadkhan et al. 
    """

    def __init__(self):
        # Initialize attributes
        self.rate_names = ['ABCG2', 
                           'ABCC2', 
                           'gamma', 
                           'SLCO2B1', 
                           'biliary\nclearance', 
                           'bacterial\nmetabolism', 
                            'delta',
                            'renal\nclearance']
        self.rates = np.zeros(len(self.rate_names))
        self.exp_plasma_conc =np.array([0, 
                                        10.04, 
                                        25.1, 
                                        46.7017, 
                                        55.7377, 
                                        50.9373, 
                                        39.0776, 
                                        29.9719, 
                                        19.3113, 
                                        14.7933, 
                                        9.4986, 
                                        5.6161, 
                                        3.2858, 
                                        1.6064]) 
        self.init_comp_conc = np.array([0.00879, 0.0, 0.0, 0.0, 0.0]) 
        self.times = np.array([0, 1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48]) 


    """
    PLOTTING METHODS
    """

    # Plot data from Azadkhan et al. 
    def plot_exp_data(self):
        plt.plot(self.times, self.exp_plasma_conc, 'o', c='k', label='Azadkhan et al.')
        plt.ylabel('SASP Plasma Concentration (\u00B5M)')
        plt.xlabel('Time (h)')  
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
    def plot_model(self, comp_no=2, title=''):
        """

        comp_no: compartment number to graph
        """

        # Add axes object if not provided
        fig, ax = plt.subplots()

        # Create smooth curve
        x, curve = self._get_curve(self.model(self.rates, comp_no=comp_no))
        ax.plot(x, curve, label='optimized\nmodel', ls='dashed', color='k')
        if comp_no == 0:
            met_x, met_curve = self._get_curve(self.model(self.rates, comp_no=4))
            ax.plot(met_x, met_curve, ls='dashdot', label='5-ASA', color='k')
            ax.set_ylabel('Lumen Concentration (\u00B5M)')

        elif comp_no == 2:
            ax.plot(self.times, self.exp_plasma_conc, 'o', c='k', label='Azadkhan et al.')
            ax.set_ylabel('SASP Plasma Concentration (\u00B5M)')
        elif comp_no == 3:
            ax.set_ylabel('micromoles of SASP')
            perc = curve[-1]/((10**6)*self.init_comp_conc[0])*100
            print(f'Percent of SASP excreted via urine: {round(perc,3)}%')
        ax.set_xlabel('Time (h)')  
        ax.legend(bbox_to_anchor=(1,1))
        ax.set_title(title)
        plt.show()

    # Plot knockouts
    def _plot_knockout(self, knockout_rates, name,):
        fig, ax = plt.subplots()

        x, curve = self._get_curve(self.model(self.rates, comp_no=2))
        ax.plot(x, curve, label='optimized\nmodel', color='k')
        ax.plot(self.times, self.exp_plasma_conc, 'o', c='k', label='Azadkhan et al.')
        x, knockout_curve = self._get_curve(self.model(knockout_rates, comp_no=2))
        ax.plot(x, knockout_curve, label=name, color='k', ls='dashed')
        ax.set_ylabel('SASP Plasma Concentration (\u00B5M)')
        ax.set_xlabel('Time (h)')  
        ax.legend(bbox_to_anchor=(1,1))
        plt.show()


        # Peak conc. fold change
        max_c = curve.max()
        max_ind = np.where(curve == max_c)[0][0]
        max_wt = knockout_curve[max_ind]
        fold_change = np.abs(max_c - max_wt) / max_c
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

    # Get smooth curve
    def _get_curve(self, model):
        """
        model: directly call self.model()
        """
        
        xy_spline = make_interp_spline(self.times, model)
        x = np.linspace(self.times.min(), self.times.max(), 500)
        curve = xy_spline(x)

        return [x, curve]

    # Plot bact model
    def _plot_bact_model(self, ax, means, stds, bact, condition, comp_no, title):

        # Parameters
        if condition == 'aerobic':
            color = 'red'
        else:
            color = 'blue'
                
        # Plot optimized model
        x, curve = self._get_curve(self.model(self.rates, comp_no=comp_no))
        ax.plot(x, curve, color='k', label='model')

        # Plot bact model
        spline = make_interp_spline(self.times, means)
        x = np.linspace(self.times.min(), self.times.max(), 500)
        curve = spline(x)
        ax.plot(x, curve, color=color, ls='dashed', label=condition)
        ax.errorbar(self.times, means, yerr=stds, color=color, ls='None', ecolor=color, capsize=3)
        if title:
            ax.set_title(f'{bact} {condition}')

    """
    COMPARTMENTAL MODEL
    """

    # System of differential equations that describe compartmental model
    def model(self, p, comp_no=2):
        """
        p :  parameters in shape [ABCG2, ABCC2, gamma, SLCO2B1, biliary_Cl, bact_metabolism, delta, renal_Cl]
        comp_no: indice of compartment to calculate concentration over time 
            0 : Lumen
            1 : Liver
            2 : Plasma
            3 : Urine 
            4 : Metabolites
            5 : Bile
        """

        def odes(y, t, p):
            """
            y: moles of SASP in each compartment
            """

            # Lumen DiffEq
            dldt = (p[0]+p[1]+p[4])*y[1] - (p[2]+p[3]+p[5])*y[0]
            
            # Liver DiffEq
            drdt = (p[2]+p[3])*y[0] - (p[0]+p[1]+p[4]+p[6])*y[1]

            # Plasma DiffEq
            dpdt = p[6]*y[1] - p[7]*y[2]

            # Urine DiffEq
            dudt = p[7]*y[2]

            # Metabolites DiffEq
            dmdt = p[5]*y[0]

            return [dldt, drdt, dpdt, dudt, dmdt]

        # Return the solutions for one compartment 
        def get_comp_conc(y, comp_no):
            # Physiological Volumes for compartments (L)
            if comp_no == 0 or comp_no == 5:
                volume = 0.105 + .013 # small intestine, large intestine
            elif comp_no == 2:
                volume = 5 * 0.60 # Blood, plasma 
            else:
                volume = 1

            # Get concentration
            comp_conc = np.empty(len(y))
            print(y)
            for i, conc_t in enumerate(y):
                comp_conc[i] = conc_t[comp_no]*((10**6)/volume) # convert to micromolar
            return comp_conc

        # Get numerical solutions to ODEs
        sol = odeint(odes, t=self.times, y0=self.init_comp_conc, args=tuple([p]))

        return get_comp_conc(sol, comp_no)
        """"""

    """
    OPTIMIZATION
    """

    # Objective function
    def _obj(self, p):
        return np.sum((self.model(p, comp_no=2) - self.exp_plasma_conc)**2)

    # Optimization powered by scipy.optimize.minimize
    def optimize(self, init_rates):
        res = minimize(self._obj, x0=init_rates, bounds = [(0, np.inf) for i in range(len(init_rates))])
        self.rates = res.x
        self.plasma_conc = self.model(self.rates, comp_no=2)
        self.lumen_conc = self.model(self.rates, comp_no=0)

        self.plot_rates()

    """
    VALIDATION
    """

    #Knockout method
    def knockout(self, rates=[0]):
        """
        rates: array of rate indices to knockout, indices can be found in self.rates_names
        """

        # Make temp rates
        knockout_rates = [r for r in self.rates]
        for r in rates:
            knockout_rates[r] = 0
        
        name = ' '.join(self.rate_names[i] for i in rates) + ' knockout'

        self._plot_knockout(knockout_rates, name)

    """
    BACTERIA INFLUENCE
    """

    # Model bacterial conditions

    def model_bact(self, comp_no: int=2, title: bool=False):

        # Parameters from Wang et al. 

        # Experimental Values
        mrp2_Ecoli_minus = [-4.50333, .946]
        mrp2_Ecoli_plus = [0.921969391, .813]
        mrp2_Bif_minus = [-6.30333333, 1.819]
        mrp2_Bif_plus = [-.008977557, .280]

        bcrp_Ecoli_minus = [-6.0466666,1.705]
        bcrp_Ecoli_plus = [-5.235153323, 1.420]
        bcrp_Bif_minus = [-7.47, 1.635]
        bcrp_Bif_plus = [-1.97470, 1.603]

        oatpb_Ecoli_minus = [-.426666, .922]
        oatpb_Ecoli_plus = [-2.514, 0.442]
        oatpb_Bif_minus = [-1.2966666, .120]
        oatpb_Bif_plus = [-2.3945, 2.005]

        # Distributions 
        bcrp_Ecoli_minus_dist = np.random.normal(bcrp_Ecoli_minus[0], bcrp_Ecoli_minus[1], size=(1000))
        bcrp_Ecoli_plus_dist = np.random.normal(bcrp_Ecoli_plus[0], bcrp_Ecoli_plus[1], size=(1000))
        bcrp_Bif_minus_dist = np.random.normal(bcrp_Bif_minus[0], bcrp_Bif_minus[1], size=(1000))
        bcrp_Bif_plus_dist = np.random.normal(bcrp_Bif_plus[0], bcrp_Bif_plus[1], size=(1000))
        mrp2_Ecoli_minus_dist = np.random.normal(mrp2_Ecoli_minus[0], mrp2_Ecoli_minus[1], size=(1000))
        mrp2_Ecoli_plus_dist = np.random.normal(mrp2_Ecoli_plus[0], mrp2_Ecoli_plus[1], size=(1000))
        mrp2_Bif_minus_dist = np.random.normal(mrp2_Bif_minus[0], mrp2_Bif_minus[1], size=(1000))
        mrp2_Bif_plus_dist = np.random.normal(mrp2_Bif_plus[0], mrp2_Bif_plus[1], size=(1000))
        oatpb_Ecoli_minus_dist = np.random.normal(oatpb_Ecoli_minus[0], oatpb_Ecoli_minus[1], size=(1000))
        oatpb_Ecoli_plus_dist = np.random.normal(oatpb_Ecoli_plus[0], oatpb_Ecoli_plus[1], size=(1000))
        oatpb_Bif_minus_dist = np.random.normal(oatpb_Bif_minus[0], oatpb_Bif_minus[1], size=(1000))
        oatpb_Bif_plus_dist = np.random.normal(oatpb_Bif_plus[0], oatpb_Bif_plus[1], size=(1000))     

        fig, axs = plt.subplots(2,2, figsize=(12,6), layout='constrained')

        means, stds = self._simulate(bcrp_Ecoli_minus_dist, mrp2_Ecoli_minus_dist, oatpb_Ecoli_minus_dist, comp_no)
        ax = axs[0,0]
        self._plot_bact_model(ax, means, stds, bact = 'E. coli Nissle 1917',condition = 'aerobic', comp_no=comp_no, title=title)

        means, stds = self._simulate(bcrp_Ecoli_plus_dist, mrp2_Ecoli_plus_dist, oatpb_Ecoli_plus_dist, comp_no)
        ax = axs[1,0]
        self._plot_bact_model(ax, means, stds, bact = 'E. coli Nissle 1917', condition = 'anaerobic', comp_no=comp_no, title=title)

        means, stds = self._simulate(bcrp_Bif_minus_dist, mrp2_Bif_minus_dist, oatpb_Bif_minus_dist, comp_no)
        ax = axs[0,1]
        self._plot_bact_model(ax, means, stds, bact = 'Bifidobacterium adolescentis', condition = 'aerobic', comp_no=comp_no, title=title)
        ax.plot([0], [0], color='b', ls='dashed', label='anaerobic')

        means, stds = self._simulate(bcrp_Bif_plus_dist, mrp2_Bif_plus_dist, oatpb_Bif_plus_dist, comp_no)
        ax = axs[1,1]
        self._plot_bact_model(ax, means, stds, bact = 'Bifidobacterium adolescentis', condition = 'anaerobic', comp_no=comp_no, title=title)

        for ax in axs.flatten():
            if comp_no == 2:
                ax.plot(self.times, self.exp_plasma_conc, 'o', c='k', label = 'Azadkhan et al.')
            if comp_no == 4:
                ax.set_ylabel('5-ASA Plasma Concentration (\u00B5M)')
            else:
                ax.set_ylabel('SASP Plasma Concentration (\u00B5M)')
            ax.set_xlabel('Time (h)')
        
        axs[0,1].legend(loc='upper right')

    def _simulate(self, bcrp, mrp2, oatpb, comp_no):
        timeseries = np.empty((len(bcrp), len(self.times)))
        for i, (bcrp_exp, mrp2_exp, oatpb_exp) in enumerate(zip(bcrp, mrp2, oatpb)):
            exp_para = [r for r in self.rates]
            exp_para[0] *= 2**bcrp_exp
            exp_para[1] *= 2**mrp2_exp
            exp_para[3] *= 2**oatpb_exp
            timeseries[i] = self.model(exp_para, comp_no)

        means = np.mean(timeseries, axis=0)
        stds = np.std(timeseries, axis=0)

        return [means, stds]













