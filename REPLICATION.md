# Steps for replicating the analysis

## Data required
1. File containing PMIDs, year of publication, MeSH terms for the Publication, MEDLINE journal code, PubType, Publication language
2. Author-ity dataset which can be used to map the authors for each PMID
3. Citation file with PMID of the paper and PMID of the reference
4. Novelty scores for each PMID using the Novelty dataset
5. Gender and ethnicity for each author
6. Age in number of prior papers for each author in each year obtained by using 1. and 2.
7. Journal similarity based on IMPLICIT data for each pair of paper and reference
8. Cumulative citations of a paper in each year
9. Join the above mentioned datasets to create a dataset which has the following columns

| Column header         | Description                                                                                           | Data type     |
|---------------------- |-----------------------------------------------------------------------------------------------        |-----------    |
| source_id             | PubMed ID of article                                                                                  | [int]         |
| source_year           | year of publication of the article                                                                    | [yyyy]        |
| source_j              | journal of the article                                                                                | [string]      |
| source_n_mesh         | number of MeSH terms in the article                                                                   | [int]         |
| source_n_mesh_ex      | number of MeSH terms in the article (expanded using the MeSH hierarchy)                               | [int]         |
| source_is_eng         | if the article is in English                                                                          | [0 or 1]      |
| source_country        | country of affiliation of the article inferred using MapAffil                                         | [string]      |
| source_is_journal     | if the article is a journal article                                                                   | [0 or 1]      |
| source_is_review      | if the article is a review article                                                                    | [0 or 1]      |
| source_is_case_rep    | if the article is a case report                                                                       | [0 or 1]      |
| source_is_let_ed_com  | if the article is a letter, editorial, or comment                                                     | [0 or 1]      |
| source_T_novelty      | time novelty of the article using novelty interface                                                   | [int]         |
| source_V_novelty      | volume novelty of the article using novelty interface                                                 | [int]         |
| source_PT_novelty     | pairwise time novelty of the article using novelty interface                                          | [int]         |
| source_PV_novelty     | pairwise volume novelty of the article using novelty interface                                        | [int]         |
| source_ncites         | number of references listed in the article                                                            | [int]         |
| source_n_authors      | number of authors listed in the article                                                               | [int]         |
| sink_id               | PubMed ID of reference                                                                                | [int]         |
| sink_year             | year of publication of the reference                                                                  | [yyyy]        |
| sink_j                | journal of the reference                                                                              | [string]      |
| sink_n_mesh           | number of MeSH terms in the reference                                                                 | [int]         |
| sink_n_mesh_ex        | number of MeSH terms in the reference (expanded using the MeSH hierarchy)                             | [int]         |
| sink_is_eng           | if the reference is in English                                                                        | [0 or 1]      |
| sink_is_journal       | if the reference is a journal reference                                                               | [0 or 1]      |
| sink_is_review        | if the reference is a review reference                                                                | [0 or 1]      |
| sink_is_case_rep      | if the reference is a case report                                                                     | [0 or 1]      |
| sink_is_let_ed_com    | if the reference is a letter, editorial, or comment                                                   | [0 or 1]      |
| sink_T_novelty        | time novelty of the reference using novelty interface                                                 | [int]         |
| sink_V_novelty        | volume novelty of the reference using novelty interface                                               | [int]         |
| sink_PT_novelty       | pairwise time novelty of the reference using novelty interface                                        | [int]         |
| sink_PV_novelty       | pairwise volume novelty of the reference using novelty interface                                      | [int]         |
| sink_n_authors        | number of authors listed in the reference                                                             | [int]         |
| year_span             | difference in publication years between reference and the article                                     | [int]         |
| journal_same          | if the journal of publication is same for reference and the article                                   | [0 or 1]      |
| mesh_sim              | cosine similarity between MeSH terms on reference and the article                                     | [float]       |
| title_sim             | cosine similarity between title terms on reference and the article                                    | [float]       |
| lang_sim              | cosine similarity between languages of publication for reference and the article                      | [float]       |
| affiliation_sim       | cosine similarity between affiliation terms on reference and the article                              | [float]       |
| pubtype_sim           | cosine similarity between publication types for reference and the article                             | [float]       |
| cite_sim              | jaccard similarity between citations for reference and the article                                    | [float]       |
| author_sim            | jaccard similarity between author for reference and the article                                       | [float]       |
| gender_sim            | cosine similarity between gender distributions for reference and the article                          | [float]       |
| eth_sim               | cosine similarity between ethnicity distributions for reference and the article                       | [float]       |
| n_common_authors      | number of common authors between the papers                                                           | [int]         |
| auid                  | author id from Author-ity                                                                             | [string]      |
| gender                | gender of the author based on Genni-Ethnea                                                            | [M, F, -]     |
| eth1                  | primary ethnicity of the author based on Genni-Ethnea                                                 | [string]      |
| eth2                  | secondary ethnicity of the author based on Genni-Ethnea                                               | [string]      |
| pos                   | author position on the byline                                                                         | [int]         |
| pos_nice              | author position on the byline, to represent Solo (0), First (1), Middle (2), Last (-1) author         | [int]         |
| sink_last_ncites      | number of citations of the reference in the last year                                                 | [int]         |
| sink_prev_ncites      | cumulative number of citations of the reference before the publication of the article                 | [int]         |
| auth_last_npapers     | number of papers of the author in the last year                                                       | [int]         |
| auth_prev_papers      | cumulative number of papers of the author before the publication of the article                       | [int]         |
| jj_sim                | journal similarity from the IMPLICIT database                                                         | [float]       |
| is_self_cite          | if the reference is a self-citation for the author                                                    | [0 or 1]      |

## Suggested libraries for the analysis

* Python 2
* Numpy, Scipy, pandas, matplotlib, seaborn
* Apache spark
* Statsmodels

## Data prep steps

Some of the above steps were done using our local copy of the dataset and can be found in the following files: 
1. [Prepare Data.ipynb](Prepare Data.ipynb) - Basic data preparation described above, relies on data collected from our inhouse databases, the DB queries are provided in [DATA_PREPROCESS.md](DATA_PREPROCESS.md)
2. [Model Prepare Data - First Author.ipynb](Model Prepare Data - First Author.ipynb) - Prepare data for doing the modelling for first author. Similar analysis for last author is provided in [Model Prepare Data - Last Author.ipynb](Model Prepare Data - Last Author.ipynb) and for middle author in [Model Prepare Data - Middle Author.ipynb](Model Prepare Data - Middle Author.ipynb). Finally, process for doing analysis on the PMC pair in provided in [Model Prepare Data - PMC Pair.ipynb](Model Prepare Data - PMC Pair.ipynb)
3. [train_model_parallel_v4.py](train_model_parallel_v4.py) - Do forward model selection on the data using. 
4. [Empirical versus Fit-First Author.ipynb](Empirical versus Fit-First Author.ipynb) - Do empirical versus fit plots for first author. For last author [Empirical versus Fit-Last Author.ipynb](Empirical versus Fit-Last Author.ipynb)
5. [Model Plotting-First Last.ipynb](Model Plotting-First Last.ipynb) - Plot model parameters for first and last author models.
6. [Journal Models First Author.ipynb](Journal Models First Author.ipynb) - Journal models for first author. For last author - [Journal Models Last Author.ipynb](Journal Models Last Author.ipynb)
7. [PMC Pair data analysis.ipynb](PMC Pair data analysis.ipynb) - Full analysis for PMC pair data
