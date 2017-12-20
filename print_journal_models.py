# coding: utf-8

from joblib import load
import pandas as pd
columns = ["llf", "aic", "bic", "prsquared", "df_model",
        "precision", "recall", "f_score", "auc", "mae", "rmse", "tp", "fp", "tn", "fn", "tpr", "fpr"]
results_journals = load("results_all_model/v4/results_all_model.v4.first_author.journal.pkl")
with open("Journal_models.v4.first_author.txt", "wb+") as fp:
    measures = pd.DataFrame(columns=columns)
    for j,(m,s) in results_journals:
        print >> fp, "%s\t%s\t%s" % ("*"*50, j, "*"*50)
        measures.ix[j,columns] = [m[k] for k in columns]
        print >> fp, s
    print >> fp, "%s\t%s\t%s" % ("*"*50, "All Measures", "*"*50)
    print >> fp, measures.to_string()
    
results_journals = load("results_all_model/v4/results_all_model.v4.last_author.journal.pkl")
with open("Journal_models.v4.last_author.txt", "wb+") as fp:
    measures = pd.DataFrame(columns=columns)
    for j,(m,s) in results_journals:
        print >> fp, "%s\t%s\t%s" % ("*"*50, j, "*"*50)
        measures.ix[j,columns] = [m[k] for k in columns]
        print >> fp, s
    print >> fp, "%s\t%s\t%s" % ("*"*50, "All Measures", "*"*50)
    print >> fp, measures.to_string()
    
