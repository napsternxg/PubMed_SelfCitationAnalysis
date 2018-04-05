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
### Column

## Expertise data

```
mysql -u "${DB_USERNAME}" -p "${EXPERTISE_DB}" -h "${DB_HOST}" -e "SELECT PMID, auid, match_len, match_prop, overall_coverage_len, overall_coverage_prop FROM pmid_auid_expertise_scores;" > data/AuthorExpertise.txt
```

## PMC Pairs

```
mysql -u "${DB_USERNAME}" -p citation -e "SELECT * FROM pmc_pair" > pmc_pair.txt
```
