import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set_context('poster')
sns.set_style('ticks')

matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.labelsize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['axes.titlesize'] = 9
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['savefig.dpi'] = 600

import pandas as pd
import numpy as np

from statsmodels.api import Logit
from joblib import Parallel, delayed
from joblib import load

import eval_measures as ems

X_cols = np.array([u'Intercept', u'journal_same[T.True]',
       u'C(gender, levels=GENDERS)[T.F]', u'C(gender, levels=GENDERS)[T.M]',
       u'mesh_sim', u'lang_sim', u'pubtype_sim',
       u'np.log10(auth_prev_papers + 1)', u'np.log10(sink_prev_ncites + 1)',
       u'np.log10(source_ncites)', u'np.log10(source_n_authors)',
       u'np.log10(source_n_mesh + 1)'])

y_cols = np.array([u'is_self_cite[False]', u'is_self_cite[True]'])

X_col_dict = dict((k,i) for i, k in enumerate(X_cols))
y_col_dict = dict((k,i) for i, k in enumerate(y_cols))


feature_dict = {
    "Intercept": ["Intercept"],
    "JournalSim": [u'journal_same[T.True]'],
    "Gender": [u'C(gender, levels=GENDERS)[T.F]', u'C(gender, levels=GENDERS)[T.M]'],
    "MeSHSim": [u'mesh_sim'],
    "LanguageSim": [u'lang_sim'],
    "PubTypeSim": [u'pubtype_sim'],
    "AuthorAge": [u'np.log10(auth_prev_papers + 1)'],
    "PriorCites": [u'np.log10(sink_prev_ncites + 1)'],
    "SourceRefs": [u'np.log10(source_ncites)'],
    "SourceAuthors": [u'np.log10(source_n_authors)'],
    "SourceMeSH": [u'np.log10(source_n_mesh + 1)']
}
feature_colors = dict(zip(feature_dict.keys(), sns.color_palette("cubehelix", len(feature_dict))))
def get_col_ids(col_list, col_dict):
  col_ids = []
  for c in col_list:
    col_ids.append(col_dict[c])
  return np.array(col_ids)

def plot_prc(prc, ax, color="k", label="PRC"):
  precision, recall = prc
  ax.plot(recall, precision,marker="None", linestyle="-", color=color, label=label)

def get_all_eval_measures(res, endog, include_prc=False):
  pred_table = res.pred_table()
  predict = res.predict()
  measures = {}
  measures["precision"] = ems.precision(pred_table)
  measures["recall"] = ems.recall(pred_table)
  measures["accuracy"] = ems.accuracy(pred_table)
  measures["f_score"] = ems.fscore_measure(pred_table)
  measures["rmse"] = ems.rmse(predict, endog)
  measures["mae"] = ems.mae(predict, endog)
  measures["auc"] = ems.auc(predict, endog)
  measures["llf"] = res.llf
  measures["aic"] = res.aic
  measures["bic"] = res.bic
  measures["prsquared"] = res.prsquared
  measures["df_model"] = res.df_model
  tn, fn, fp, tp = map(float, pred_table.flatten()) # WRT to 1 as positive label
  measures["tn"] = tn
  measures["fn"] = fn
  measures["fp"] = fp
  measures["tp"] = tp
  print "In eval measures function."
  if include_prc:
    ## Include the precision recall values
    prc = ems.prc(predict, endog, float_precision=3)
    measures["prc"] = prc
  return measures

def model_fit(y,X, X_cols, y_col, feature_key="Gender", testing=False, include_prc=False):
  if testing:
    # If testing the just print X and y columns
    print X_cols, y_col
    return feature_key, ({"llf": 0.1}, "TEMP SUMMARY")
  ## Not testing. Fit the models and return the measures
  print feature_key, X.shape, X_cols, y.shape, y_col
  X = pd.DataFrame(X,columns=X_cols)
  y = pd.Series(y,name=y_col)
  print "Created dataframes."
  model = Logit(y,X)
  res = model.fit()
  measures = get_all_eval_measures(res, model.endog, include_prc=include_prc)
  return feature_key, (measures, res.summary2())

X = load("tmp/X", mmap_mode='r')
y = load("tmp/y", mmap_mode='r')

train_items = list(feature_dict.iteritems())
n_jobs = len(train_items)


results_full = []
TOP_FEATURES = ['Intercept']
y_feature_col = "is_self_cite[True]"
nrows, ncols = (len(feature_dict)-len(TOP_FEATURES) + 1)/5, 5
stage_id = 0
include_prc = True
if include_prc:
  fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3, nrows*3))
  print "Created figure with %s rows and %s columns" % (nrows, ncols)
  ax = ax.flatten()
## Run the forward selection process
while True:
  x_feature_cols = reduce(lambda x,y: x+y, map(lambda k: feature_dict[k], TOP_FEATURES))
  ## Generate feature ids to train on
  train_items = list((k,v) for k,v in feature_dict.iteritems() if k not in TOP_FEATURES)
  n_jobs = len(train_items)
  if n_jobs < 1:
    ## If no more jobs to train then exit.
    print "Finished with forward selection process.\nEnding."
    break
  ## Fit the models
  try:
    results = Parallel(n_jobs=n_jobs, verbose=10,
      temp_folder="./tmp")(delayed(model_fit)(y[:, y_col_dict[y_feature_col]],
        X[:, get_col_ids(x_feature_cols + v, X_col_dict)],
        x_feature_cols + v,
        y_feature_col, feature_key=k, testing=False,
        include_prc=include_prc) for k,v in train_items)
  except SystemError:
    ## Fall back to sequential training
    print "Falling back to sequential training"
    results = []
    for k,v in train_items:
      r = model_fit(y[:, y_col_dict[y_feature_col]],
        X[:, get_col_ids(x_feature_cols + v, X_col_dict)],
        x_feature_cols + v,
        y_feature_col, feature_key=k, testing=False,
        include_prc=include_prc)
      results.append(r)
  ## Sort the results based on log likelihood
  results = sorted(results, key=lambda x: x[1][0]["llf"], reverse=True)
  ## Assign top feature from the current stage
  tf = results[0][0]
  TOP_FEATURES.append(tf)
  print "Top feature in Stage %s: %s" % (stage_id+1, tf)
  ## Print all the models from the current stage
  for k,(measures,summary) in results:
    ## Remove PRC from the dictionary
    prc = measures.pop("prc", None)
    ## Plot PRC curves
    if include_prc and prc is not None:
      plot_prc(prc, ax=ax[stage_id], color=feature_colors[k], label=k)
    print k, measures
    print summary
  ## Add plot properties - line for opposite diagonal
  if include_prc:
    ax[stage_id].plot(np.arange(0,1.01,0.01), 1-np.arange(0,1.01,0.01), "--k", linewidth=1)
    ax[stage_id].set_title("Stage %s: %s" % (stage_id+1, tf))
    ax[stage_id].set_xlabel("Recall")
    ax[stage_id].set_ylabel("Precision")
  ## Append the results to the results_full list
  results_full.append(results)
  ## Print features by log likelihood for each stage
  for i, (r, tf) in enumerate(zip(results_full, TOP_FEATURES[1:])):
    print "\nStage %s: %s\n" % (i + 1, tf)
    for k,(measures,summary) in r:
      print "%s: %.4f" % (k, measures["llf"],),
  print "\nTop feature in Stage %s: %s" % (stage_id+1, tf)
  ## Update stage id
  stage_id += 1

## Add plot legend and save plot
if include_prc:
  lgd = fig.legend(*ax[0].get_legend_handles_labels(), loc = 'upper center', bbox_to_anchor=(0.5,1.1), ncol=5, frameon=True, fancybox=True, prop={"size": 9})
  fig.tight_layout()
  plt.savefig("PRC_model.v1.pdf", bbox_inches="tight", bbox_extra_artists=[lgd])

## Measures plot for each stage for each type of measures
measures_labels = ["llf", "aic", "bic", "prsquared",
    "precision", "recall", "f_score", "auc"]
stage_colors = sns.color_palette("cubehelix", len(results_full))
fig, ax = plt.subplots(2,4,sharex=True, figsize=(4*3, 2*3))
ax = ax.flatten()
xticklabels = TOP_FEATURES[1:]
xticks_dict = dict((k,i) for i,k in enumerate(xticklabels))
for i,m in enumerate(measures_labels):
  for j,r in enumerate(results_full):
    features, values = zip(*map(lambda (k,(measures,summary)): (xticks_dict[k],measures[m]), r))
    ax[i].plot(features, values, color=stage_colors[j], linestyle="--", marker="o", label="Stage %s" % (j+1))
  ax[i].set_ylabel(m)
  ax[i].set_xticks(range(len(xticklabels)))
  ax[i].set_xticklabels(xticklabels, rotation=90)

lgd = fig.legend(*ax[0].get_legend_handles_labels(), loc = 'upper center', bbox_to_anchor=(0.5,1.1), ncol=5, frameon=True, fancybox=True, prop={"size": 9})
fig.tight_layout()
plt.savefig("Measures_model.v1.pdf", bbox_inches="tight", bbox_extra_artists=[lgd])


## Create table for each measure
pd.options.display.width=500
measures_labels = ["llf", "aic", "bic", "prsquared", "df_model",
    "precision", "recall", "f_score", "auc", "mae", "rmse", "tp", "fp", "tn", "fn"]
with open("Full_measures_coeffs_model.v1.txt", "wb+") as fp:
  for j,(r,tf) in enumerate(zip(results_full, TOP_FEATURES[:-1])):
    print >> fp, "Stage %s: %s" % (j+1,tf)
    header = ["Features"] + measures_labels
    rows = map(lambda (k,(measures,summary)): ([k] 
      + [measures[m] for m in measures_labels]), r)
    df_measures = pd.DataFrame(rows, columns=header)
    print >> fp, df_measures
  print >> fp, "***"*50
  for j,(r,tf) in enumerate(zip(results_full, TOP_FEATURES[:-1])):
    print >> fp, "##"*10 + ("Stage %s: %s" % (j+1,tf)) + "##"*10
    for k,(m,summary) in r:
      print >> fp, "Feature: %s" % k
      print >> fp, summary
  print >> fp, "***"*50


## Plot coefficients of Gender
gender_coeffs = []
gender_ci_025 = []
gender_ci_975 = []
index_names = []
for i, (r,tf) in enumerate(zip(results_full, TOP_FEATURES)):
  print "Stage %s: %s" % (i, tf)
  res_dict = dict(r)
  k = "Gender"
  if k in res_dict:
    measures, summary = res_dict["Gender"]
  else:
    measures, summary = res_dict.values()[0]
  print summary.tables[1].ix[["C(gender, levels=GENDERS)[T.F]", "C(gender, levels=GENDERS)[T.M]"], "Coef."]
  print summary.tables[1].ix[["C(gender, levels=GENDERS)[T.F]", "C(gender, levels=GENDERS)[T.M]"], "[0.025"]
  print summary.tables[1].ix[["C(gender, levels=GENDERS)[T.F]", "C(gender, levels=GENDERS)[T.M]"], "0.975]"]
  index_names.append("Stage %s: %s" % (i, tf))
  gender_coeffs.append(summary.tables[1].ix[["C(gender, levels=GENDERS)[T.F]", "C(gender, levels=GENDERS)[T.M]"], "Coef."])
  gender_ci_025.append(summary.tables[1].ix[["C(gender, levels=GENDERS)[T.F]", "C(gender, levels=GENDERS)[T.M]"], "[0.025"])
  gender_ci_975.append(summary.tables[1].ix[["C(gender, levels=GENDERS)[T.F]", "C(gender, levels=GENDERS)[T.M]"], "0.975]"])

## Create dataframes
df_gender_coeffs = pd.DataFrame(gender_coeffs, index=index_names)
df_gender_ci_025 = pd.DataFrame(gender_ci_025, index=index_names)
df_gender_ci_975 = pd.DataFrame(gender_ci_975, index=index_names)

## Calculate odds ratio

df_gender_coeffs[["Odds Ratio Female", "Odds Ratio Male"]] = np.exp(
    df_gender_coeffs[["C(gender, levels=GENDERS)[T.F]",
      "C(gender, levels=GENDERS)[T.M]"]],)
df_gender_ci_025[["Odds Ratio Female", "Odds Ratio Male"]] = np.exp(
    df_gender_ci_025[["C(gender, levels=GENDERS)[T.F]",
      "C(gender, levels=GENDERS)[T.M]"]],)
df_gender_ci_975[["Odds Ratio Female", "Odds Ratio Male"]] = np.exp(
    df_gender_ci_975[["C(gender, levels=GENDERS)[T.F]",
      "C(gender, levels=GENDERS)[T.M]"]],)

df_gender_coeffs["Odds Ratio Female/Male"] = df_gender_coeffs["Odds Ratio Female"] / df_gender_coeffs["Odds Ratio Male"]


## Plot the coefficients
xticks = df_gender_coeffs.index.values
x = np.arange(xticks.shape[0])
fig, ax = plt.subplots(3,1, sharex=True)

ax[0].errorbar(x, df_gender_coeffs["C(gender, levels=GENDERS)[T.F]"],
    yerr=[df_gender_coeffs["C(gender, levels=GENDERS)[T.F]"] - df_gender_ci_025["C(gender, levels=GENDERS)[T.F]"], 
         df_gender_ci_975["C(gender, levels=GENDERS)[T.F]"] - df_gender_coeffs["C(gender, levels=GENDERS)[T.F]"]],
  fmt="-ro", label="Female", elinewidth=2)
ax[0].errorbar(x, df_gender_coeffs["C(gender, levels=GENDERS)[T.M]"],
    yerr=[df_gender_coeffs["C(gender, levels=GENDERS)[T.M]"] - df_gender_ci_025["C(gender, levels=GENDERS)[T.M]"],
         df_gender_ci_975["C(gender, levels=GENDERS)[T.M]"] - df_gender_coeffs["C(gender, levels=GENDERS)[T.M]"]],
  fmt="-bs", label="Male", elinewidth=2)
ax[0].axhline(y=0.0, color="k", linestyle="--", lw=1)
ax[0].set_ylabel("Coefficient")
ax[0].legend(loc="upper right")

ax[1].errorbar(x, df_gender_coeffs["Odds Ratio Female"],
    yerr=[df_gender_coeffs["Odds Ratio Female"] - df_gender_ci_025["Odds Ratio Female"],
         df_gender_ci_975["Odds Ratio Female"] - df_gender_coeffs["Odds Ratio Female"]],
  fmt="-ro", label="Female", elinewidth=2)
ax[1].errorbar(x, df_gender_coeffs["Odds Ratio Male"],
    yerr=[df_gender_coeffs["Odds Ratio Male"] - df_gender_ci_025["Odds Ratio Male"],
         df_gender_ci_975["Odds Ratio Male"] - df_gender_coeffs["Odds Ratio Male"]],
  fmt="-bs", label="Male", elinewidth=2)
ax[1].axhline(y=1.0, color="k", linestyle="--", lw=1)
ax[1].set_ylabel("Odds Ratio (O.R.)")
ax[1].legend(loc="upper right")

bar_vals = (1./df_gender_coeffs["Odds Ratio Female/Male"]) - 1
ax[2].bar(x - 0.25, bar_vals, width=0.5, color=(bar_vals >= 0.0).map({True: "b", False: "r"}))
ax[2].axhline(y=0.0, color="k", linestyle="--", lw=1)
ax[2].set_ylabel("O.R. M/F - 1")
ax[2].set_xticks(x)
ax[2].set_xticklabels(xticks, rotation=90)
plt.savefig("Gender_coeffs_model.v1.pdf", bbox_inches="tight")



## Plot empirical distributions

def logit(p):
  return np.log(p/(1-p))

def get_empirical_logit(y,X, X_col, y_col, conditioning_vals, 
    conditioning_var="Gender", 
    conditioning_names = ["F", "M", "U"],
    bins=None, testing=False, clip=(-3,3)):
  if testing:
    # If testing the just print X and y columns
    print X_col, y_col
    return conditioning_var, ({"llf": 0.1}, "TEMP SUMMARY")
  ## Not testing. Fit the models and return the measures
  print conditioning_var, X.shape, X_col, y.shape, y_col
  df = pd.DataFrame(X,columns=X_col)
  df[y_col] = y
  df[conditioning_var] = 0
  print df.shape
  print conditioning_vals.shape
  for i,v in enumerate(conditioning_names[:-1]):
    t = conditioning_vals[:,i]
    print v, t.shape
    df.ix[t == 1, conditioning_var] = v
  df.ix[df[conditioning_var] == 0, conditioning_var] = conditioning_names[-1]
  print X_col, conditioning_var, y_col
  t = df.pivot_table(index=pd.cut(df[X_col[0]], bins=bins),
      columns=conditioning_var, values=y_col,
      aggfunc=[np.mean, len])
  #t["logit"] = logit(t["mean"]).clip(*clip)
  t = t.set_index(bins[1:])
  return t

colors = ["r", "b", "k"]
conditioning_names = ["F", "M", "U"]
cm = plt.cm.get_cmap('Greys')
plt.clf()
plt.close("all")
fig, ax = plt.subplots(2,5,figsize=(15,6))
ax = ax.flatten()
plot_id = 0
for k in TOP_FEATURES:
  if k == "Gender" or k == "Intercept":
    continue
  if k in ["AuthorAge", "PriorCites", "SourceMeSH"]:
    bins = np.array([-0.1] + range(10) + range(10,100,10) + range(100,1000,100))
    bins = np.log10(bins + 1)
  if k in ["SourceRefs", "SourceAuthors"]:
    bins = np.array(range(1,10) + range(10,100,10) + range(100,1000,100))
    bins = np.log10(bins)
  if k in ["PubTypeSim", "MeSHSim", "JournalSim", "LanguageSim"]:
    bins = np.arange(-0.01,1.01,0.01)
  print k
  t = get_empirical_logit(y[:, y_col_dict[y_feature_col]],
    X[:, get_col_ids(feature_dict[k], X_col_dict)],
    feature_dict[k],
    y_feature_col,
    X[:, get_col_ids(feature_dict["Gender"], X_col_dict)],
    conditioning_var="Gender",
    conditioning_names = conditioning_names,
    bins=bins,
    testing=False)
  t_logit = logit(t["mean"]).clip(-5,5) 
  lgd_items = []
  for j,c in enumerate(conditioning_names):
    #scp = ax[plot_id].scatter(t_logit.index, t_logit[c], c=t["len"][c], cmap=cm, s=200, norm=matplotlib.colors.LogNorm(), edgecolors='none', alpha=0.7)
    scp = ax[plot_id].scatter(t_logit.index, t_logit[c], c=colors[j], s=20*(np.log10(t["len"][c])), edgecolors='none', alpha=0.3)
    line, = ax[plot_id].plot(t_logit[t_logit[c].notnull()].index, t_logit[t_logit[c].notnull()][c], marker="None", linestyle="-", color=colors[j], label=c, lw=3, alpha=0.7)
    ax[plot_id].set_xlabel(feature_dict[k][0])
    ax[plot_id].set_ylabel("$logit(p)$")
    ax[plot_id].set_title(k)
    lgd_items.append(line)
  plot_id += 1

#ax.legend(handles=lgd_items, loc="upper center", ncol=3)
lgd = fig.legend(lgd_items, ["Female", "Male", "Unknown"], loc = 'upper center', bbox_to_anchor=(0.5,1.1), ncol=3, frameon=True, fancybox=True, prop={"size": 16})
fig.tight_layout()
#cax,kw = matplotlib.colorbar.make_axes([k for k in ax])
#fcb = fig.colorbar(scp, ax=ax.ravel().tolist())
#fcb = fig.colorbar(scp, cax=cax, **kw)
#fcb.ax.set_ylabel('Frequency')
plt.savefig("Logit_model.v1.pdf", bbox_inches="tight", bbox_extra_artists=[lgd])
