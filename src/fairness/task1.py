from IPython.display import display, Math

from src.model.sfm import SFM
from src.metric.stats import *
from src.metric.queries import sample_ctf, get_u_n, get_conditioned_u
from src.metric.counterfactual import CTFTerm

def _prob(sample, val):
    n = sample.numel()
    return 0 if n==0 else (sample==val).sum().item() / n

class FairnessCookbook:
    def __init__(self, sfm: SFM, x0_val, x1_val, u=None, n=10000, verbose=True):
        self.u, _ = get_u_n(sfm, u, n)
        self.sfm = sfm 
        self.x0 = {sfm.X: x0_val}
        self.x1 = {sfm.X: x1_val}
        self.verbose = verbose

    def _sampleY_dox1wx0(self, x=None):
        condition = x if x else self.x0
        Wx0 = {"nested": CTFTerm(self.sfm.W, self.x0)} if self.sfm.W is not None else {}
        Y_dox1wx0 = CTFTerm(self.sfm.Y, {**self.x1, **Wx0})
        return sample_ctf(self.sfm, Y_dox1wx0, condition, u=self.u)

    def _sampleY_dox0wx1(self, x=None):
        condition = x if x else self.x0
        Wx1 = {"nested": CTFTerm(self.sfm.W, self.x1)} if self.sfm.W is not None else {}
        Y_dox0wx1 = CTFTerm(self.sfm.Y, {**self.x0, **Wx1})
        return sample_ctf(self.sfm, Y_dox0wx1, condition, u=self.u)
    
    def _sampleY_dox0(self, x=None):
        condition = x if x else self.x0
        return sample_ctf(self.sfm, CTFTerm(self.sfm.Y,self.x0), condition, u=self.u)
    
    def _sampleY_dox1(self, x=None):
        condition = x if x else self.x0
        return sample_ctf(self.sfm, CTFTerm(self.sfm.Y,self.x1), condition, u=self.u)
    
    def x_specific_TE_DE_IE(self, y_val=1, x=None):
        x = x if x else self.x0
        Y = self.sfm.Y

        pY_dox1wx0 = _prob(self._sampleY_dox1wx0(x)[Y], y_val)
        pY_dox0wx1 = _prob(self._sampleY_dox0wx1(x)[Y], y_val)
        pY_dox0 = _prob(self._sampleY_dox0(x)[Y], y_val)
        pY_dox1 = _prob(self._sampleY_dox1(x)[Y], y_val)

        te = pY_dox1-pY_dox0
        print(f'xTE_x0x1(Y={y_val} | X={x[self.sfm.X]}) = {te:.4f}')

        xde_x0x1 = pY_dox1wx0 - pY_dox0
        xde_x1x0 = pY_dox0wx1 - pY_dox1
        xde_sym = 0.5 * (xde_x0x1 - xde_x1x0)
        print(f'xDE^sym(Y={y_val} | X={x[self.sfm.X]}) = {xde_sym:.4f}')

        xie_x0x1 = pY_dox0wx1 - pY_dox0
        xie_x1x0 = pY_dox1wx0 - pY_dox1
        xie_sym = 0.5 * (xie_x0x1 - xie_x1x0)
        print(f'xIE^sym(Y={y_val} | X={x[self.sfm.X]}) = {xie_sym:.4f}')
        return {'TE':te, 'DE':xde_sym, 'DEx0x1':xde_x0x1, 'DEx1x0':xde_x1x0, 'IE':xie_sym, 'IEx0x1':xie_x0x1, 'IEx1x0':xie_x1x0}

    def x_se(self, y_val=1, x=None, x_=None):
        x = x if x else self.x1
        x_ = x_ if x_ else self.x0 
        Y = self.sfm.Y
        pY_dox_condx_ = _prob(sample_ctf(self.sfm, CTFTerm(Y, x), x_, u=self.u)[Y], y_val)
        pY_dox_condx = _prob(sample_ctf(self.sfm, CTFTerm(Y, x), x, u=self.u)[Y], y_val)
        se = pY_dox_condx_ - pY_dox_condx
        print(f'xSE_{x},{x_}(Y={y_val}) = {se:.4f}')
        return se

    def x_specific_effects(self, y_val=1):
        sym = self.x_specific_TE_DE_IE(y_val, self.x0)
        se = self.x_se(y_val, self.x1, self.x0)
        tv = sym['TE'] - se 
        print(f'TV_{self.x0},{self.x1}(Y={y_val}) = {tv:.4f}')


    def fairness_cookbook(self, bn={}):
        Y = self.sfm.Y
        # get 95% confidence interval on expected values
        mx1wx0, hx1wx0, _, _ = confidence_interval(self._sampleY_dox1wx0()[Y])
        mx0, hx0, _, _ = confidence_interval(self._sampleY_dox0()[Y])
        # x-DE_{x0,x1}(y|x)
        de_x0x1, hde_x0x1, _, _ = diff_from_margins(mx1wx0, hx1wx0, mx0, hx0)


        mx0wx1, hx0wx1, _, _ = confidence_interval(self._sampleY_dox0wx1()[Y])
        mx1, hx1, _, _ = confidence_interval(self._sampleY_dox1()[Y])

        # x-DE_{x0,x1}(y|x)
        de_x0x1, hde_x0x1, _, _ = diff_from_margins(mx1wx0, hx1wx0, mx0, hx0)
        # x-DE_{x1,x0}(y|x)
        de_x1x0, hde_x1x0, _, _ = diff_from_margins(mx0wx1, hx0wx1, mx1, hx1)
        # x-DE^sym
        de_sym, hde_sym, _, _ = diff_from_margins(de_x0x1, hde_x0x1, de_x1x0, hde_x1x0, 0.5)
        hde0 = (abs(de_sym) <= abs(hde_sym)).item()
        print(f'xDE^sym = {de_sym:.4f}\u00B1{hde_sym:.4f}\t hypothesis "no direct effect" {'ACCEPTED' if hde0 else 'REJECTED'}')
        print(f'\t ---> {'no ' if hde0 else ''}evidence of disperate TREATMENT.')
        

        # x-IE_{x0,x1}(y|x)
        ie_x0x1, hie_x0x1, _, _ = diff_from_margins(mx0wx1, hx0wx1, mx0, hx0)
        # x-IE_{x1,x0}(y|x)
        ie_x1x0, hie_x1x0, _, _ = diff_from_margins(mx1wx0, hx1wx0, mx1, hx1)
        # x-IE^sym
        ie_sym, hie_sym, _, _ = diff_from_margins(ie_x0x1, hie_x0x1, ie_x1x0, hie_x1x0, 0.5)
        hie0 = (abs(ie_sym) <= abs(hie_sym)).item()
        print(f'xIE^sym = {ie_sym:.4f}\u00B1{hie_sym:.4f}\t hypothesis "no indirect effect" {'ACCEPTED' if hie0 else 'REJECTED'}')

        # x-SE_{x1,x0}
        mx1_condx0, hx1_condx0, _, _ = confidence_interval(self._sampleY_dox1(self.x0)[Y])
        mx1_condx1, hx1_condx1, _, _ = confidence_interval(self._sampleY_dox1(self.x1)[Y])
        se_x1x0, hse_x1x0, _, _ = diff_from_margins(mx1_condx0, hx1_condx0, mx1_condx1, hx1_condx1)
        hse0 = (abs(se_x1x0) <= abs(hse_x1x0)).item()
        print(f'xSE_x1x0 = {se_x1x0:.4f}\u00B1{hse_x1x0:.4f}\t hypothesis "no spurious effect" {'ACCEPTED' if hse0 else 'REJECTED'}')
        print(f'\t ---> {'no ' if hie0 and hse0 else ''}evidence of disperate IMPACT.')
        print()

def exp_se(model, U, X, Y, x_val, y_val=1):
    _x = {X:x_val}
    sampleY_condx = sample_ctf(model, CTFTerm([Y]), conditions=_x, u=U)
    sampleY_dox = sample_ctf(model, CTFTerm([Y], do_vals=_x), u=U)
    pY_condx = _prob(sampleY_condx[Y], y_val)
    pY_dox = _prob(sampleY_dox[Y], y_val)
    return pY_condx - pY_dox

def x_se(model, U, X, Y, x0, x1, y_val=1):
    _x0 = {X:x0}
    _x1 = {X:x1}
    sampleY_dox0_condx1 = sample_ctf(model, CTFTerm([Y], do_vals=_x0), conditions=_x1, u=U)
    sampleY_dox0_condx0 = sample_ctf(model, CTFTerm([Y], do_vals=_x0), conditions=_x0, u=U)
    pY_dox0_condx1 = _prob(sampleY_dox0_condx1[Y], y_val)
    pY_dox0_condx0 = _prob(sampleY_dox0_condx0[Y], y_val)
    return pY_dox0_condx1 - pY_dox0_condx0

def fairness_cookbook(model, X, Z, W, Y, x0=0, x1=1, y_val=1, effect_type='nat', get_probs=True, u=None, n=10000):
    U, n = get_u_n(model, u, n)
    X_x0 = {X:x0}
    X_x1 = {X:x1}

    condition = {} if effect_type=='nat' else X_x0 if effect_type=='ctf' else None # could add support for z-specific, etc.

    Wx0 = {"nested": CTFTerm(W, X_x0)} if W is not None else {}
    Y_dox1wx0 = CTFTerm([Y], {**X_x1, **Wx0})
    sampleY_dox1wx0 = sample_ctf(model, Y_dox1wx0, condition, u=U)

    Wx1 = {"nested": CTFTerm(W, X_x1)} if W is not None else {}
    Y_dox0wx1 = CTFTerm([Y], {**X_x0, **Wx1})
    sampleY_dox0wx1 = sample_ctf(model, Y_dox0wx1, condition, u=U)

    sampleY_dox0 = sample_ctf(model, CTFTerm([Y], X_x0), condition, u=U)
    sampleY_dox1 = sample_ctf(model, CTFTerm([Y], X_x1), condition, u=U)

    if get_probs:
        pY_dox1wx0 = _prob(sampleY_dox1wx0[Y], y_val)
        pY_dox0wx1 = _prob(sampleY_dox0wx1[Y], y_val)
        pY_dox0 = _prob(sampleY_dox0[Y], y_val)
        pY_dox1 = _prob(sampleY_dox1[Y], y_val)

        te = pY_dox1-pY_dox0

        de_x0x1 = pY_dox1wx0 - pY_dox0
        de_x1x0 = pY_dox0wx1 - pY_dox1
        de_sym = 0.5 * (de_x0x1 - de_x1x0)

        ie_x0x1 = pY_dox0wx1 - pY_dox0
        ie_x1x0 = pY_dox1wx0 - pY_dox1
        ie_sym = 0.5 * (ie_x0x1 - ie_x1x0)

        ret = {'TE':te, 'DE':de_sym, 'DEx0x1':de_x0x1, 'DEx1x0':de_x1x0, 'IE':ie_sym, 'IEx0x1':ie_x0x1, 'IEx1x0':ie_x1x0}
        if effect_type=='nat':
            ret['exp-SEx0'] = exp_se(model, U=U, X=X, Y=Y, x_val=x0, y_val=y_val)
            ret['exp-SEx1'] = exp_se(model, U=U, X=X, Y=Y, x_val=x1, y_val=y_val)
        elif effect_type=='ctf':
            ret['SEx0x1'] = x_se(model, U=U, X=X, Y=Y, x0=x0, x1=x1, y_val=y_val)
            ret['SEx1x0'] = x_se(model, U=U, X=X, Y=Y, x0=x1, x1=x0, y_val=y_val)
        return ret


