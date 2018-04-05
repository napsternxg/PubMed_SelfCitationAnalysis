# Self-Citation analysis in PubMed

```
export DB_USER_NAME="<username>"
export DB_HOST="<host>"
export EXPERTISE_DB="<expertise_db_name>"

```


## Collect data

### Article level data
```
mysql -u "${DB_USERNAME}" -p PUBMED2015 -h "${DB_HOST}" -e "SELECT a.PMID, a.year, a.journal, mesh, Mesh_counts, Exploded_Mesh_counts, title_tokenized, languages, c.mapaffil_author, pub_types, TFirstP, VolFirstP, Pair_TFirstP, Pair_VolFirstP,d.cited, d.Ncited, d.Ncitedby FROM Articles as a LEFT JOIN novelty.novelty_scores as b ON a.PMID = b.PMID LEFT JOIN MapAffil.Country as c ON a.PMID = c.PMID LEFT JOIN citation.cite_list as d ON a.PMID = d.PMID" > data/FullArticlesData.txt
```

### Author level data

```
mysql -u "${DB_USERNAME}" -p PUBMED2010 -h "${DB_HOST}" -e "SELECT au_id, au_ids, Ethnea, Genni FROM au_clst_all AS a LEFT JOIN ethnea.authority AS b ON a.au_id = b.auid;" > data/FullAuthorData.txt
```

### Author level Ethnea
```
mysql -u "${DB_USERNAME}" -p ethnea -h "${DB_HOST}" -e "SELECT auid, Ethnea, Genni, SexMac, SSNgender FROM authority;" > data/AuthorityEthnea.txt
```

### Author first last years
```
mysql -u "${DB_USERNAME}" -p PUBMED2010 -h "${DB_HOST}" -e "SELECT au_id, first_year, last_year FROM au_clst_all;" > data/AuthorityFirstLastYears.txt
```

### Journal Journal similarity
```
mysql -u "${DB_USERNAME}" -p IMPLICIT -h "${DB_HOST}" -e "SELECT T1,T2,score FROM JJ;" > data/jj_sim.txt
```

## Expertise data

```
mysql -u "${DB_USERNAME}" -p "${EXPERTISE_DB}" -h "${DB_HOST}" -e "SELECT PMID, auid, match_len, match_prop, overall_coverage_len, overall_coverage_prop FROM pmid_auid_expertise_scores;" > data/AuthorExpertise.txt
```

## PMC Pairs

```
mysql -u "${DB_USERNAME}" -p citation -e "SELECT * FROM pmc_pair" > pmc_pair.txt
```

### Pubtypes

https://www.nlm.nih.gov/mesh/pubtypes.html

Considered:

 * Journal Article
 * Review
 * Case Reports
 * Letters, Comments, Editorials
 * Others

### Training Data
Folder: `out/training_data_full`
Description: Training data of all papers published between [2002,2005]
Header format:
```
['source_id', 'source_year', 'source_j', 'source_n_mesh', 'source_n_mesh_ex', 'source_is_eng', 'source_country', 'source_is_journal', 'source_is_review', 'source_is_case_rep', 'source_is_let_ed_com', 'source_T_novelty', 'source_V_novelty', 'source_PT_novelty', 'source_PV_novelty', 'source_ncites', 'source_n_authors', 'sink_id', 'sink_year', 'sink_j', 'sink_n_mesh', 'sink_n_mesh_ex', 'sink_is_eng', 'sink_is_journal', 'sink_is_review', 'sink_is_case_rep', 'sink_is_let_ed_com', 'sink_T_novelty', 'sink_V_novelty', 'sink_PT_novelty', 'sink_PV_novelty', 'sink_n_authors', 'year_span', 'journal_same', 'mesh_sim', 'title_sim', 'lang_sim', 'affiliation_sim', 'pubtype_sim', 'cite_sim', 'author_sim', 'gender_sim', 'eth_sim', 'n_common_authors', 'auid', 'gender', 'eth1', 'eth2', 'pos', 'pos_nice', 'sink_last_ncites', 'sink_prev_ncites', 'auth_last_npapers', 'auth_prev_papers', 'jj_sim', 'is_self_cite']
```

### Column descriptions

| Column header         | Description                                                                                     | Data type   |
|---------------------- |-----------------------------------------------------------------------------------------------  |-----------  |
| source_id             | PubMed ID of article                                                                            | [int]       |
| source_year           | year of publication of the article                                                              | [yyyy]      |
| source_j              | journal of the article                                                                          | [string]    |
| source_n_mesh         | number of MeSH terms in the article                                                             | [int]       |
| source_n_mesh_ex      | number of MeSH terms in the article (expanded using the MeSH hierarchy)                         | [int]       |
| source_is_eng         | if the article is in English                                                                    | [0 or 1]    |
| source_country        | country of affiliation of the article inferred using MapAffil                                   | [string]    |
| source_is_journal     | if the article is a journal article                                                             | [0 or 1]    |
| source_is_review      | if the article is a review article                                                              | [0 or 1]    |
| source_is_case_rep    | if the article is a case report                                                                 | [0 or 1]    |
| source_is_let_ed_com  | if the article is a letter, editorial, or comment                                               | [0 or 1]    |
| source_T_novelty      | time novelty of the article using novelty interface                                             | [int]       |
| source_V_novelty      | volume novelty of the article using novelty interface                                           | [int]       |
| source_PT_novelty     | pairwise time novelty of the article using novelty interface                                    | [int]       |
| source_PV_novelty     | pairwise volume novelty of the article using novelty interface                                  | [int]       |
| source_ncites         | number of references listed in the article                                                      | [int]       |
| source_n_authors      | number of authors listed in the article                                                         | [int]       |
| sink_id               | PubMed ID of reference                                                                          | [int]       |
| sink_year             | year of publication of the reference                                                            | [yyyy]      |
| sink_j                | journal of the reference                                                                        | [string]    |
| sink_n_mesh           | number of MeSH terms in the reference                                                           | [int]       |
| sink_n_mesh_ex        | number of MeSH terms in the reference (expanded using the MeSH hierarchy)                       | [int]       |
| sink_is_eng           | if the reference is in English                                                                  | [0 or 1]    |
| sink_is_journal       | if the reference is a journal reference                                                         | [0 or 1]    |
| sink_is_review        | if the reference is a review reference                                                          | [0 or 1]    |
| sink_is_case_rep      | if the reference is a case report                                                               | [0 or 1]    |
| sink_is_let_ed_com    | if the reference is a letter, editorial, or comment                                             | [0 or 1]    |
| sink_T_novelty        | time novelty of the reference using novelty interface                                           | [int]       |
| sink_V_novelty        | volume novelty of the reference using novelty interface                                         | [int]       |
| sink_PT_novelty       | pairwise time novelty of the reference using novelty interface                                  | [int]       |
| sink_PV_novelty       | pairwise volume novelty of the reference using novelty interface                                | [int]       |
| sink_n_authors        | number of authors listed in the reference                                                       | [int]       |
| year_span             | difference in publication years between reference and the article                               | [int]       |
| journal_same          | if the journal of publication is same for reference and the article                             | [0 or 1]    |
| mesh_sim              | cosine similarity between MeSH terms on reference and the article                               | [float]     |
| title_sim             | cosine similarity between title terms on reference and the article                              | [float]     |
| lang_sim              | cosine similarity between languages of publication for reference and the article                | [float]     |
| affiliation_sim       | cosine similarity between affiliation terms on reference and the article                        | [float]     |
| pubtype_sim           | cosine similarity between publication types for reference and the article                       | [float]     |
| cite_sim              | jaccard similarity between citations for reference and the article                              | [float]     |
| author_sim            | jaccard similarity between author for reference and the article                                 | [float]     |
| gender_sim            | cosine similarity between gender distributions for reference and the article                    | [float]     |
| eth_sim               | cosine similarity between ethnicity distributions for reference and the article                 | [float]     |
| n_common_authors      | number of common authors between the papers                                                     | [int]       |
| auid                  | author id from Author-ity                                                                       | [string]    |
| gender                | gender of the author based on Genni-Ethnea                                                      | [M, F, -]   |
| eth1                  | primary ethnicity of the author based on Genni-Ethnea                                           | [string]    |
| eth2                  | secondary ethnicity of the author based on Genni-Ethnea                                         | [string]    |
| pos                   | author position on the byline                                                                   | [int]       |
| pos_nice              | author position on the byline, to represent Solo (0), First (1), Middle (2), Last (-1) author   | [int]       |
| sink_last_ncites      | number of citations of the reference in the last year                                           | [int]       |
| sink_prev_ncites      | cumulative number of citations of the reference before the publication of the article           | [int]       |
| auth_last_npapers     | number of papers of the author in the last year                                                 | [int]       |
| auth_prev_papers      | cumulative number of papers of the author before the publication of the article                 | [int]       |
| jj_sim                | journal similarity from the IMPLICIT database                                                   | [float]     |
| is_self_cite          | if the reference is a self-citation for the author                                              | [0 or 1]    |

