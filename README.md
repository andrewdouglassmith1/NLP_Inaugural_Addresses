# Analyzing Inaugural Addresses with NLP
Check out my blog post (TBU) .

**Description**

Utilizing topic modelling and dimensionality reduction, I analyze topic trends through time and cluster presidents. 

**Data**

* [OSF](https://osf.io/r56zb/?pid=6xdkn) , includes all addresses excluding President Biden
* Pasted President Biden's address into a text file

**File Contents**

* `Data/`
    - `InauguralTexts/` contains text files of all addresses
    - `PresidentInfo/Pres_Details.csv ` contains further details on the presidents such as, political party, start date, end date
    - `Visualization` contains images utilized in my presentation and blog posts
* `Code/` 
    - `01_Data_Cleaning.ipynb` contains all data aggregation and cleaning necessary for this project
    - `02_EDA_Analysis.ipynb` contains  all initial EDA (i.e. word counts, words per sentence, "I" vs. "We")
    - `03_NLP_Analysis_Aggregate_Tokenized.ipynb` contains initial topic modelling (LDA/NMF) and clustering at the address level.
    - `04_NLP_Sentence&Par_Tokenized.ipynb` contains NMF topic modelling at both the sentence and paragraph level.  Topic vectors were averaged for each president.  These results are the bulk of my presentation and blog post.
    - `05_CorEx.ipynb` contains topic modelling with CorEx.  Seeding initial topics with CorEx did not product results as well as NMF topic modelling at the paragraph level and so was not discussed in my presentation or blog post.
    - `06_Sentiment_Analysis.ipynb` contains sentiment analysis utilizing nltk's VADER 
    - `Classes_Functions.py` contains functions used throughout notebooks 01-06







 



