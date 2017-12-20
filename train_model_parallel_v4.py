## Add MKL config to safeguard against conflict of MKL's and joblib's parallelization
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

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
from joblib import load, dump

import eval_measures as ems

MODEL_VERSION = "v4"
MODEL_SUFFIX = "%s.last_author" % MODEL_VERSION

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", default="v4", type=str,
    help="Model version number")
parser.add_argument("-t", "--model-type", default="last_author", type=str,
    help="Model type [last_author | first_author] or any other string")
args = parser.parse_args()
if args.version:
  MODEL_VERSION = args.version
  print "Model version number: %s" % MODEL_VERSION
if args.model_type:
  MODEL_SUFFIX = "%s.%s" % (MODEL_VERSION, args.model_type)
  print "Model suffix: %s" % MODEL_SUFFIX


from feature_data import feature_dict

def plot_prc(prc, ax, color="k", label="PRC"):
  precision, recall = prc
  ax.plot(recall, precision,marker="None", linestyle="-", color=color, label=label)

def get_all_eval_measures(predict, endog, include_prc=False):
  measures = {}
  pred_table = ems.cm(predict, endog)
  measures["precision"] = ems.precision(pred_table)
  measures["recall"] = ems.recall(pred_table)
  measures["accuracy"] = ems.accuracy(pred_table)
  measures["f_score"] = ems.fscore_measure(pred_table)
  measures["rmse"] = ems.rmse(predict, endog)
  measures["mae"] = ems.mae(predict, endog)
  measures["auc"] = ems.auc(predict, endog)
  tn, fp, fn, tp = map(float, pred_table.flatten()) # WRT to 1 as positive label
  measures["tn"] = tn
  measures["fn"] = fn
  measures["fp"] = fp
  measures["tp"] = tp
  measures["tpr"] = tp * 1. / (tp + fn)
  measures["fpr"] = fp * 1. / (fp + tn)
  print "In eval measures function."
  if include_prc:
    print "Generating PRC AND ROC"
    ## Include the precision recall values
    prc = ems.prc(predict, endog, float_precision=3)
    measures["prc"] = prc
    roc = ems.roc(predict, endog, float_precision=3)
    measures["roc"] = roc
  return measures


def model_fit(store_path, X_df_path, y_df_path,
    feature_key="Gender", X_cols = [], testing=False, include_prc=False):
  if testing:
    # If testing the just print X and y columns
    print store_path, X_df_path, y_df_path, feature_key, X_cols
    return feature_key, ({"llf": 0.1}, "TEMP SUMMARY")
  ## Not testing. Fit the models and return the measures
  print store_path, X_df_path, y_df_path, feature_key, X_cols
  X = pd.read_hdf(store_path, key=X_df_path, columns=X_cols)
  y = pd.read_hdf(store_path, key=y_df_path)
  print "Created dataframes, feature_key=%s" % feature_key
  print "X.shape = %s, y.shape = %s" % (X.shape, y.shape)
  model = Logit(y,X)
  res = model.fit()
  predict = res.predict()
  measures = get_all_eval_measures(predict, model.endog, include_prc=include_prc)
  measures["llf"] = res.llf
  measures["aic"] = res.aic
  measures["bic"] = res.bic
  measures["prsquared"] = res.prsquared
  measures["df_model"] = res.df_model
  return feature_key, (measures, res.summary2())


store_path, X_df_path, y_df_path = "out/Model.%s.h5" % MODEL_SUFFIX, "X", "y"
train_items = list(feature_dict.iteritems())
n_jobs = len(train_items)
results_full = []
TOP_FEATURES = ['Intercept']
y_feature_col = "is_self_cite[True]"
nrows, ncols = ((len(feature_dict)-len(TOP_FEATURES))/5) + 1, 5
stage_id = 0
include_prc = True
feature_colors = dict(zip(feature_dict.keys(),
  sns.color_palette("cubehelix", len(feature_dict)))
  )
if include_prc:
  fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3, nrows*3))
  fig1, ax1 = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(ncols*3, nrows*3))
  print "Created figure with %s rows and %s columns" % (nrows, ncols)
  ax = ax.flatten()
  ax1 = ax1.flatten()
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
      temp_folder="./tmp")(delayed(model_fit)(store_path, X_df_path, y_df_path,
        feature_key=k,
        X_cols = x_feature_cols + v,
        testing=False,
        include_prc=include_prc) for k,v in train_items)
  except (SystemError, MemoryError) as e:
    ## Fall back to sequential training
    print "Falling back to sequential training"
    results = []
    for k,v in train_items:
      r = model_fit(store_path, X_df_path, y_df_path,
        feature_key=k,
        X_cols = x_feature_cols + v,
        testing=False,
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
    roc = measures.pop("roc", None)
    ## Plot PRC curves
    if include_prc and prc is not None:
      plot_prc(prc, ax=ax[stage_id], color=feature_colors[k], label=k)
      ax[stage_id].plot([measures["recall"]], [measures["precision"]],
          marker="o", color=feature_colors[k], ms=20)
      plot_prc(roc, ax=ax1[stage_id], color=feature_colors[k], label=k)
      ax1[stage_id].plot([measures["fpr"]], [measures["tpr"]],
          marker="o", color=feature_colors[k], ms=20)
    print k, measures
    print summary
  ## Add plot properties - line for opposite diagonal
  if include_prc:
    ax[stage_id].plot([0,1], [1,0], "--k", linewidth=1)
    ax[stage_id].set_title("Stage %s: %s" % (stage_id+1, tf))
    ax[stage_id].set_xlabel("Recall")
    ax[stage_id].set_ylabel("Precision")
    ax1[stage_id].plot([0,1], [0,1], "--k", linewidth=1)
    ax1[stage_id].set_title("Stage %s: %s" % (stage_id+1, tf))
    ax1[stage_id].set_xlabel("False Positive Rate")
    ax1[stage_id].set_ylabel("True Positive Rate")
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
  fig.savefig("PRC_model.%s.pdf" % MODEL_SUFFIX, bbox_inches="tight", bbox_extra_artists=[lgd])
  
  lgd = fig1.legend(*ax1[0].get_legend_handles_labels(), loc = 'upper center', bbox_to_anchor=(0.5,1.1), ncol=5, frameon=True, fancybox=True, prop={"size": 9})
  fig1.tight_layout()
  fig1.savefig("ROC_model.%s.pdf" % MODEL_SUFFIX, bbox_inches="tight", bbox_extra_artists=[lgd])


## Measures plot for each stage for each type of measures
measures_labels = ["llf", "aic", "bic", "prsquared",
    "precision", "recall", "f_score", "auc"]
stage_colors = sns.color_palette("husl", len(results_full))
plt.clf()
plt.close("all")
fig, ax = plt.subplots(2,4,sharex=True, figsize=(4*3, 2*3))
ax = ax.flatten()
xticklabels = TOP_FEATURES[1:]
xticks_dict = dict((k,i) for i,k in enumerate(xticklabels))
for i,m in enumerate(measures_labels):
  for j,r in enumerate(results_full):
    features, values = zip(*map(lambda (k,(measures,summary)): (xticks_dict[k],measures[m]), r))
    ax[i].plot(features, values, color=stage_colors[j], linestyle="--", marker="x", label="Stage %s" % (j+1), linewidth=1)
  ax[i].set_ylabel(m)
  ax[i].set_xticks(range(len(xticklabels)))
  ax[i].set_xticklabels(xticklabels, rotation=90)

lgd = fig.legend(*ax[0].get_legend_handles_labels(), loc = 'upper center', bbox_to_anchor=(0.5,1.1), ncol=5, frameon=True, fancybox=True, prop={"size": 9})
fig.tight_layout()
plt.savefig("Measures_model.%s.pdf" % MODEL_SUFFIX, bbox_inches="tight", bbox_extra_artists=[lgd])


## Generate intercept model
dump(results_full, "results_all_model/%s/results_all_model.%s.pkl" % (MODEL_VERSION, MODEL_SUFFIX))
x_feature_cols = feature_dict["Intercept"]
results_intercept = model_fit(store_path, X_df_path, y_df_path, feature_key="Intercept", X_cols = x_feature_cols, testing=False, include_prc=False)
dump([[results_intercept]] + results_full, "results_all_model/%s/results_all_model_intercept.%s.pkl" % (MODEL_VERSION, MODEL_SUFFIX))
# DONE


## Create table for each measure
pd.options.display.width=500
measures_labels = ["llf", "aic", "bic", "prsquared", "df_model",
    "precision", "recall", "f_score", "auc", "mae", "rmse", "tp", "fp", "tn", "fn"]
with open("Full_measures_coeffs_model.%s.txt" % MODEL_SUFFIX, "wb+") as fp:
  for j,(r,tf) in enumerate(zip([[results_intercept]] + results_full, TOP_FEATURES)):
    print >> fp, "Stage %s: %s" % (j+1,tf)
    header = ["Features"] + measures_labels
    rows = map(lambda (k,(measures,summary)): ([k] 
      + [measures[m] for m in measures_labels]), r)
    df_measures = pd.DataFrame(rows, columns=header)
    print >> fp, df_measures
  print >> fp, "***"*50
  for j,(r,tf) in enumerate(zip([[results_intercept]] + results_full, TOP_FEATURES)):
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
plt.savefig("Gender_coeffs_model.%s.pdf" % MODEL_SUFFIX, bbox_inches="tight")
