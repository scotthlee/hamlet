import numpy as np
import pandas as pd

from scipy.stats import binom
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def threshold(probs, cutpoint=.5):
    """Quick function for thresholding probabilities."""
    return np.array(probs >= cutpoint).astype(np.uint8)


def brier_score(targets, guesses):
    """Calculates Brier score."""
    n_classes = len(np.unique(targets))
    assert n_classes > 1
    if n_classes == 2:
        bs = np.sum((guesses - targets)**2) / targets.shape[0]
    else:
        y = onehot_matrix(targets)
        row_diffs = np.diff((guesses, y), axis=0)[0]
        squared_diffs = row_diffs ** 2
        row_sums = np.sum(squared_diffs, axis=1)
        bs = row_sums.mean()
    return bs


def matthews_correlation(y, y_, undef_val=0):
    """Calculates Matthews Correlation Coefficient."""
    infos = youdens_j(y, y_) * youdens_j(y_, y)
    marks = markedness(y, y_) * markedness(y_, y)
    prod = infos * marks
    return prod ** (1/4) if prod != 0 else undef_val


def f1_score(y, y_, undef_val=0):
    """Alternative call for f_score()."""
    return f_score(y, y_, b=1, undef_val=undef_val)


def f_score(y, y_, b=1, undef_val=0):
    """Calculates F-score."""
    se = sensitivity(y, y_)
    pv = positive_predictive_value(y, y_)
    if se + pv != 0:
        return (1 + b**2) * (se * pv) / ((b**2 * pv) + se)
    else:
        return undef_val


def sensitivity(y, y_, undef_val=0.0):
    """Calculates sensitivity, or recall."""
    tp = np.sum((y ==1) & (y_ == 1))
    Np = y.sum()
    return (tp / Np) if Np != 0 else undef_val


def specificity(y, y_, undef_val=0.0):
    """Calculates specificity, or 1 - FPR."""
    tn = np.sum((y == 0) & (y_ == 0))
    Nn = np.sum(y == 0)
    return (tn / Nn) if Nn != 0 else undef_val


def positive_predictive_value(y, y_, undef_val=0.0):
    """Calculates positive predictive value, or precision."""
    tp = np.sum((y == 1) & (y_ == 1))
    pNp = y_.sum()
    return tp / pNp if pNp != 0 else undef_val


def negative_predictive_value(y, y_, undef_val=0.0):
    """Calculates negative predictive value."""
    tn = np.sum((y == 0) & (y_ == 0))
    pNn = np.sum(y_ == 0)
    return tn / pNn if pNn != 0 else undef_val


def markedness(y, y_):
    """Calculates markedness, or PPV + NPV - 1."""
    ppv = positive_predictive_value(y, y_)
    npv = negative_predictive_value(y, y_)
    return ppv + npv - 1


def youdens_j(y, y_, a=1, b=1):
    """Calculates Youden's J index from two binary vectors."""
    c = a + b
    a = a / c * 2
    b = b / c * 2
    sens = np.sum((y == 1) & (y_ == 1)) / y.sum()
    spec = np.sum((y == 0) & (y_ == 0)) / (len(y) - y.sum())
    return a*sens + b*spec - 1


def sesp_to_obs(se, sp, p, N=1000):
    """Returns simulated target-prediction pairs from sensitivity,
    specificity, prevalence, and total N.
    """
    pairs = [[0, 0], [1, 0], [0, 1], [1, 1]]
    D, C, A, B = sesp_to_counts(se, sp, p, N)
    obs = []
    for i, count in enumerate([A, B, C, D]):
        if count > 0:
            obs += [count * [pairs[i]]]
    obs = pd.DataFrame(np.concatenate(obs, axis=0),
                       columns=['y', 'yhat'])
    return obs


def sesp_to_counts(se, sp, p, N):
    """Calculates the counts in a 2x2 contingency table as a function of
    sensitivity, specificity, and prevalence.
    """
    Np = int(round(p * N))
    Nn = int(N - Np)
    tp = int(round(se * Np))
    fp = int(round((1 - sp) * Nn))
    tn = int(round(sp * Nn))
    fn = int(round((1 - se) * Np))
    return tp, fp, tn, fn


def sesp_to_ppv(se, sp, p):
    """Calculates PPV as a function of sensitivity, specificity, and
    prevalence.
    """
    return (se * p) / ((se * p) + ((1 - sp) * (1 - p)))


def sesp_to_npv(se, sp, p):
    """Calculates NPV as a function of sensitivity, specificity, and
    prevalence.
    """
    pn = 1 - p
    return (sp * pn) / ((sp * pn) + ((1 - se) * p))


def sesp_to_mcc(se, sp, p):
    """Calculates Matthews Correlation Coefficient as a function of
    sensitivity, specificity, and prevalence.
    """
    return


def mcnemar_test(targets, guesses, cc=True):
    """Calculates McNemar's chi-squared statistic."""
    cm = confusion_matrix(targets, guesses)
    b = int(cm[0, 1])
    c = int(cm[1, 0])
    if cc:
        stat = (abs(b - c) - 1)**2 / (b + c)
    else:
        stat = (b - c)**2 / (b + c)
    p = 1 - chi2(df=1).cdf(stat)
    outmat = np.array([b, c, stat, p]).reshape(-1, 1)
    out = pd.DataFrame(outmat.transpose(),
                       columns=['b', 'c', 'stat', 'pval'])
    return out


def brier_score(targets, guesses):
    """Calculates Brier score for binary and multiclass problems."""
    n_classes = len(np.unique(targets))
    if n_classes == 2:
        guesses = guesses.flatten()
        bs = np.sum((guesses - targets)**2) / targets.shape[0]
    else:
        y = onehot_matrix(targets)
        row_diffs = np.diff((guesses, y), axis=0)[0]
        squared_diffs = row_diffs ** 2
        row_sums = np.sum(squared_diffs, axis=1)
        bs = row_sums.mean()
    return bs


def spec_at_sens(y, y_, sens=0.7, return_df=True, round=2, pct=True):
    """Calculates maximum specifity that achieves the required level of 
    specificity.
    """
    fprs, tprs, cuts = roc_curve(y, y_, drop_intermediate=False)
    nearest = np.min(np.where(tprs >= sens)[0])
    out = [tprs[nearest], 1 - fprs[nearest], cuts[nearest]]
    if pct:
        out[0] *= 100
        out[1] *= 100
    if return_df:
        out = pd.DataFrame(out).transpose().round(round)
        out.columns = ['sens', 'spec', 'cutpoint']
    return out
    

def clf_metrics(y, y_,
                cutpoint=0.5,
                p_adj=None,
                average='weighted',
                mod_name=None,
                round=4,
                round_pval=False,
                mcnemar=False,
                argmax_axis=1):
    # Converting pd.Series to np.array
    stype = type(pd.Series([0]))
    if type(y_) == stype:
        y_ = y_.values
    if type(y) == stype:
        y = y.values

    # Figuring out if the guesses are classes or probabilities
    if np.any([0 < p < 1 for p in y_.flatten()]):
        preds_are_probs = True
    else:
        preds_are_probs = False

    # Optional exit for doing averages with multiclass/label inputs
    if len(np.unique(y)) > 2:
        # Getting binary metrics for each set of results
        codes = np.unique(y)

        # Softmaxing the probabilities if it hasn't already been done
        if np.sum(y_[0]) > 1:
            y_ = np.array([np.exp(p) / np.sum(np.exp(p)) for p in y_])

        # Argmaxing for when we have probabilities
        if preds_are_probs:
            auc = roc_auc_score(y, y_,
                                average=average,
                                multi_class='ovr')
            brier = brier_score(y, y_)
            y_ = np.argmax(y_, axis=argmax_axis)

        # Making lists of the binary predictions (OVR)
        y = [np.array([doc == code for doc in y], dtype=np.uint8)
             for code in codes]
        y_ = [np.array([doc == code for doc in y_], dtype=np.uint8)
              for code in codes]

        # Getting the stats for each set of binary predictions
        stats = [clf_metrics(y[i], y_[i], round=16) for i in range(len(y))]
        stats = pd.concat(stats, axis=0)
        stats.fillna(0, inplace=True)
        cols = stats.columns.values

        # Calculating the averaged metrics
        if average == 'weighted':
            weighted = np.average(stats,
                                  weights=stats.true_prev,
                                  axis=0)
            out = pd.DataFrame(weighted).transpose()
            out.columns = cols
        elif average == 'macro':
            out = pd.DataFrame(stats.mean()).transpose()
        elif average == 'micro':
            out = clf_metrics(np.concatenate(y),
                              np.concatenate(y_))

        # Adding AUC and AP for when we have probabilities
        if preds_are_probs:
            out.auc = auc
            out.brier = brier

        # Rounding things off
        out = out.round(round)
        count_cols = [
            'tp', 'fp', 'tn', 'fn', 'true_prev',
            'pred_prev', 'prev_diff'
        ]
        out[count_cols] = out[count_cols].round()

        if mod_name is not None:
            out['model'] = mod_name

        return out

    # Thresholding the probabilities, if provided
    if preds_are_probs:
        auc = roc_auc_score(y, y_)
        brier = brier_score(y, y_)
        ap = average_precision_score(y, y_)
        y_ = threshold(y_, cutpoint)
    else:
        brier = np.round(brier_score(y, y_), round)

    # Doing sens and spec first
    sens = np.round(sensitivity(y, y_), round)
    spec = np.round(specificity(y, y_), round)

    # Optionally making a reweighted sample
    if p_adj is not None:
        reweighted = sesp_to_obs(sens, spec, p_adj, y.shape[0])
        y, y_ = reweighted['y'].values, reweighted['yhat'].values
    else:
        p_adj = y.sum() / y.shape[0]

    # Calculating the main binary metrics
    ppv = positive_predictive_value(y, y_)
    npv = negative_predictive_value(y, y_)
    mcc = matthews_correlation(y, y_)
    f1 = f1_score(y, y_)
    j = sens + spec - 1

    # Getting the counts
    p = y.sum() / y.shape[0]
    if p_adj is not None:
        p = p_adj

    tp, fp, tn, fn = sesp_to_counts(sens, spec, p, y.shape[0])

    # Rolling everything so far into a dataframe
    outmat = np.array(
        [tp, fp, tn, fn, sens, spec, ppv, npv, j, f1, mcc,
         brier]).reshape(-1, 1)
    out = pd.DataFrame(outmat.transpose(),
                       columns=['tp', 'fp', 'tn',
                                'fn', 'sens', 'spec', 'ppv',
                                'npv', 'j', 'f1', 'mcc', 'brier'])

    # Optionally tacking on stats from the raw probabilities
    if preds_are_probs:
        out['auc'] = auc
        out['ap'] = ap
    else:
        out['auc'] = 0.0
        out['ap'] = 0.0

    # Calculating some additional measures based on positive calls
    true_prev = int(np.sum(y == 1))
    pred_prev = int(np.sum(y_ == 1))
    abs_diff = pred_prev - true_prev
    rel_diff = abs_diff / true_prev
    if mcnemar:
        pval = mcnemar_test(y, y_).pval[0]
        if round_pval:
            pval = np.round(pval, round)
    count_outmat = np.array([true_prev, pred_prev, abs_diff,
                             rel_diff]).reshape(-1, 1)
    count_out = pd.DataFrame(
        count_outmat.transpose(),
        columns=['true_prev', 'pred_prev', 'prev_diff', 'rel_prev_diff'])
    out = pd.concat([out, count_out], axis=1)

    # Optionally dropping the mcnemar p-val
    if mcnemar:
        out['mcnemar'] = pval

    # Tacking on the model name
    if mod_name is not None:
        out['model'] = mod_name
    
    # And findally rounding 
    float_cols = ['sens', 'spec', 'ppv',
                  'npv', 'j', 'f1', 'mcc',
                  'brier', 'rel_prev_diff']
    count_cols = ['tp', 'fp', 'tn',
                  'fn', 'true_prev', 'pred_prev',
                  'prev_diff']
    out[float_cols] = out[float_cols].round(round)
    out[count_cols] = out[count_cols].astype('int64')

    return out
