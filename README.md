# Self Citation analysis on PubMed data

Code for reproducing the experiments, figures, and tables presented in the paper **Mishra, S., Fegley, B. D., Diesner, J., & Torvik, V. I. (2018). Self-Citation is the Hallmark of Productive Authors, of Any Gender. PLoS One.**


## Data

Our experiments relied on data from multiple sources including properitery data from [Thompson Rueter's (now Clarivate Analytics) Web of Science collection of MEDLINE citations](https://clarivate.com/products/web-of-science/databases/). Author's interested in reproducing our experiments should personally request from Clarivate Analytics for this data. However, we do make a similar but open dataset based on citations from PubMed Central which can be utilized to get similar results to those reported in our analysis. Furthermore, we have also freely shared our datasets which can be used along with the citation datasets from Clarivate Analytics, to re-create the datased used in our experiments. These datasets are listed below. If you wish to use any of those datasets please make sure you cite both the dataset as well as the paper introducing the dataset. 

* MEDLINE 2015 baseline: https://www.nlm.nih.gov/bsd/licensee/2015_stats/baseline_doc.html
* Citation data from PubMed Central (original paper includes additional citations from Web of Science)
* Author-ity 2009 dataset: 
    - Dataset citation: Torvik, Vetle I.; Smalheiser, Neil R. (2018): Author-ity 2009 - PubMed author name disambiguated dataset. University of Illinois at Urbana-Champaign. https://doi.org/10.13012/B2IDB-4222651_V1
    - Paper citation: Torvik, V. I., & Smalheiser, N. R. (2009). Author name disambiguation in MEDLINE. ACM Transactions on Knowledge Discovery from Data, 3(3), 1–29. https://doi.org/10.1145/1552303.1552304
    - Paper citation: Torvik, V. I., Weeber, M., Swanson, D. R., & Smalheiser, N. R. (2004). A probabilistic similarity metric for Medline records: A model for author name disambiguation. Journal of the American Society for Information Science and Technology, 56(2), 140–158. https://doi.org/10.1002/asi.20105
* Genni 2.0 + Ethnea for identifying author gender and ethnicity:
    - Dataset citation: Torvik, Vetle (2018): Genni + Ethnea for the Author-ity 2009 dataset. University of Illinois at Urbana-Champaign. https://doi.org/10.13012/B2IDB-9087546_V1
    - Paper citation: Smith, B. N., Singh, M., & Torvik, V. I. (2013). A search engine approach to estimating temporal changes in gender orientation of first names. In Proceedings of the 13th ACM/IEEE-CS joint conference on Digital libraries - JCDL ’13. ACM Press. https://doi.org/10.1145/2467696.2467720
    - Paper citation: Torvik VI, Agarwal S. Ethnea -- an instance-based ethnicity classifier based on geo-coded author names in a large-scale bibliographic database. International Symposium on Science of Science March 22-23, 2016 - Library of Congress, Washington DC, USA. http://hdl.handle.net/2142/88927
* MapAffil for identifying article country of affiliation:
    - Dataset citation: Torvik, Vetle I. (2018): MapAffil 2016 dataset -- PubMed author affiliations mapped to cities and their geocodes worldwide. University of Illinois at Urbana-Champaign. https://doi.org/10.13012/B2IDB-4354331_V1
    - Paper citation: Torvik VI. MapAffil: A Bibliographic Tool for Mapping Author Affiliation Strings to Cities and Their Geocodes Worldwide. D-Lib magazine : the magazine of the Digital Library Forum. 2015;21(11-12):10.1045/november2015-torvik. 
* IMPLICIT journal similarity:
    - Dataset citation: Torvik, Vetle (2018): Author-implicit journal, MeSH, title-word, and affiliation-word pairs based on Author-ity 2009. University of Illinois at Urbana-Champaign. https://doi.org/10.13012/B2IDB-4742014_V1
* Novelty dataset for identify article level novelty:
    - Dataset citation: Mishra, Shubhanshu; Torvik, Vetle I. (2018): Conceptual novelty scores for PubMed articles. University of Illinois at Urbana-Champaign. https://doi.org/10.13012/B2IDB-5060298_V1
    - Paper citation: Mishra, S., & Torvik, V. I. (2016). Quantifying Conceptual Novelty in the Biomedical Literature. D-Lib Magazine, 22(9/10). https://doi.org/10.1045/september2016-mishra 
    - Code: https://github.com/napsternxg/Novelty
* Expertise dataset for identifying author expertise on articles: 
* Source code provided at: https://github.com/napsternxg/PubMed_SelfCitationAnalysis

**Note: The dataset is based on a snapshot of PubMed (which includes Medline and PubMed-not-Medline records) taken in the first week of October, 2016.**
Check here for information to get PubMed/MEDLINE, and NLMs data Terms and Conditions:
https://www.nlm.nih.gov/databases/download/pubmed_medline.html

Additional data related updates can be found at: http://abel.ischool.illinois.edu


## Acknowledgments

This work was made possible in part with funding to VIT from [NIH grant P01AG039347](https://projectreporter.nih.gov/project_info_description.cfm?aid=8475017&icde=18058490) and [NSF grant 1348742](http://www.nsf.gov/awardsearch/showAward?AWD_ID=1348742). The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

## License

Self-citation analysis data based on PubMed Central subset (2002-2005) by Shubhanshu Mishra, Brent D. Fegley, Jana Diesner, and Vetle Torvik is licensed under a Creative Commons Attribution 4.0 International License.
Permissions beyond the scope of this license may be available at https://github.com/napsternxg/PubMed_SelfCitationAnalysis.

