import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

default_format = 'png'

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.formula.api import logit
from statsmodels.api import Logit
import patsy

import eval_measures as ems
import model_functions as mf


print "MFH Reloaded1"
genders = ['-', 'M', 'F'] # Ordering is important as that will decide which is used as reference

def plot_prc(prc, prc_filename="PRC.pdf"):
  plt.close("all")
  plt.clf()
  precision, recall = prc
  plt.plot(recall, precision, "-r")
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title("Average Precision: %.3f, Average Recall: %.3f" % (precision.mean(), recall.mean()))
  plt.savefig(prc_filename, bbox_inches="tight")

def fit_model(df, formula, title="Full", fp=None, filename="Model", save=False):
  """
  Function to fit model, collect stats and save predictions and model.
  df: dataframe
  formula: formula
  title: title of model (Default: "Full")
  fp: File pointer (Default: None)
  filename: Model and data file prefix ("Model")
  save: Weather to save predictions, model or both or none ["Both", "Data", "Model", False] (Default: False)
  """
  if df.shape[0] < 10:
    print "Too less instances. Skipping. Make sure you have atleast 10 instances."
    return None, None
  print "Modelling Model[%s] with instances %s" % (title, df.shape[0])
  print "Using formula:\n %s" % (formula)
  print "Generating patsy matrices"
  y, X = patsy.dmatrices(formula, df, return_type="dataframe")
  print "Initializing model"
  model = Logit(y,X)
  print "Fitting model"
  res = model.fit()
  print title, "\n", res.summary2()
  print "Confusion Matrix:", res.pred_table()
  precision = ems.precision(res.pred_table())
  recall = ems.recall(res.pred_table())
  accuracy = ems.accuracy(res.pred_table())
  f_score = ems.fscore_measure(res.pred_table())
  rmse = ems.rmse(res.predict(), model.endog)
  mae = ems.mae(res.predict(), model.endog)
  auc = ems.auc(res.predict(), model.endog)
  prc = ems.prc(res.predict(), model.endog)
  prc_filename = "%s.pdf" % filename
  plot_prc(prc, prc_filename)
  evaluation_metrics = "[Model Measures]: Confusion Matrix: %s\nRMSE: %s\tMAE: %s\tAUC: %s\nPrecision: %s\tRecall: %s\tAccuracy: %s\tF1-Score: %s\nPRC:\n%s" % (res.pred_table(), rmse, mae, auc, precision, recall, accuracy, f_score, prc_filename)
  print evaluation_metrics
  print "[save=%s]" % save, "" if save else "Not", "Saving Model to %s" % filename
  if fp is not None:
    print >> fp, "Modelling Model[%s] with instances %s" % (title, df.shape[0])
    print >> fp, "Using formula:\n %s" % (formula)
    print >> fp, title, "\n", res.summary2()
    print >> fp, evaluation_metrics
    print >> fp, "[save=%s]" % save, "" if save else "Not", "Saving Model to %s" % filename
  model_save, data_save = False, False
  if save == "Both":
    model_save, data_save = True, True
  if save == "Model" or model_save:
    model_file = "%s.pkl" % filename
    res.save(model_file, remove_data=True) # Save model
  if save == "Data" or data_save:
    data_file = "%s.data.txt" % filename # Include predictions
    print "df.index", df.index
    save_data(df[["from_id", "is_self_cite"]], res.predict(), filename=data_file)
  print "Done Saving"
  return model, res

def save_data(df_data, predictions, filename="Model.data.txt"):
  # NOTE This df should be the df_data and not the original df
  df_data = df_data.copy()
  df_data["pred"] = predictions
  df_data.to_csv(filename, sep='\t')
  print "Saved data predictions to %s with shape %s" % (filename, df_data.shape)

# To get the formula for creating patsy dmatricies while prediction
def get_formula(formula, title, solo_term='total_authors'):
  if title == 'Solo' and 'total_authors' in formula:
    formula += '- %s' % solo_term
  if title != 'Solo' and 'total_authors' in formula:
    formula += '+ mf.score_k(total_authors, 3) + mf.score_k(total_authors, 4) + mf.score_k(total_authors, 5)'
    if title != 'Middle':
      formula += '+ mf.score_k(total_authors, 2) '
  return "%s" % formula

def run_exp(df, formula, logfile='model_res.temp.txt',\
    title_pos = zip(['First', 'Last', 'Middle'], [1, -1, 2]),\
    include_solo=False, solo_term='total_authors',\
    path_prefix="all_data/models/FULL.temp", log_title="Full", save=False):
  if include_solo:
    #title_pos.append(("Solo", 0))
    title_pos = [("Solo", 0)] + title_pos
  with open(logfile, "wb+") as fp:
    print "=="*10, "\n", log_title, "\n", "=="*10
    print >> fp, "=="*10, "\n", log_title, "\n", "=="*10
    base_formula = formula
    for title, pos in title_pos:
      formula = get_formula(base_formula, title, solo_term=solo_term)
      print "formula=", formula
      filename = "%s.%s" % (path_prefix, title)
      #fit_model(df[df.au_pos_nice == pos], formula, title, fp, filename=filename, save=save)
      fit_model(df, formula, title, fp, filename=filename, save=save)


def run_features(df, formula_list, formula_features,\
    base_formula = "%s ~ mf.score_ref_k(from_yr, 2003) + %s",\
    response = "is_self_cite",\
    title_pos=zip(["First", "Last", "Middle"], [1,-1,2]),\
    logfile="all_data/IterativeModels.txt", path_prefix="all_data/models/Iterative",\
    iterative=False, include_solo = False, start_index=0):
  print "Using Formula List: ", formula_list
  t_formula = ""
  prefix = ""
  # start_index only should be set in the case when iterative is True
  if not iterative and start_index > 0:
    raise Exception("start_index only should be set in the case when iterative is True")
  print "USING START INDEX: %s, FOLLOWING FEATURES WILL BE IN ALL MODELS: %s" % (start_index, formula_list[:start_index])
  for i,k in enumerate(formula_list[:start_index]):
    tmpl = "%s + %s"
    if i == 0:
      tmpl = "%s %s"
    t_formula = tmpl % (t_formula, formula_features[k])
    prefix = "%s_%s" % (prefix, k)
  for i, k in enumerate(formula_list[start_index:]):
    if iterative:
      tmpl = "%s + %s"
      if i + start_index == 0:
        tmpl = "%s %s"
      t_formula = tmpl % (t_formula, formula_features[k])
      formula = base_formula % (response, t_formula)
      prefix = "%s_%s" % (prefix, k)
    else:
      formula = base_formula % (response, formula_features[k])
      prefix = k
    print "Processing for formula of %s: %s" % (prefix, formula)
    log_filename = "%s.%s.txt" % (logfile, prefix)
    print "Using logfile: ", log_filename
    run_exp(df, formula, title_pos=title_pos, logfile=log_filename, path_prefix = "%s_%s" % (path_prefix, prefix), include_solo=include_solo, log_title=prefix)
