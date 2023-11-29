# Lexicons (feature_engine/features/lexicons/)
This is a folder for lexicon-based features. Some of the lexicons are proprietary, and therefore they are kept hidden on the .gitignore. 

List of lexicons in this folder:
- `liwc_lexicons/`: lexicons associated with the Linguistic Inquiry and Word Count (LIWC)
- `liwc_lexicons_small_test/`: smaller set of lexicons from LIWC for testing purposes
- `other_lexicons/' contains the following:
	- `positive-words.txt`: a list from Hu and Liu, KDD-2004, [downloadable here](http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar)
	- `nltk_english_stopwords.txt`: stopwords from [NLTK](https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords)
	- `hedge_words`: a list of hedge words from https://web.stanford.edu/~jurafsky/pubs/ranganath2013.pdf (C.9, pg. 9)
	- `first_person`: a list of first-person pronouns
- `certainty.txt`: a list of words associated with the extent to which someone expresses doubt, versus certainty. From [Rocklage et al. (2023)](https://journals.sagepub.com/doi/pdf/10.1177/00222437221134802).
- `dale_chall.txt`: List of easy words according to Dale-Chall, sourced from [this link](https://countwordsworth.com/download/DaleChallEasyWordList.txt).
- `function_words`: a list of "function words" (from https://web.stanford.edu/~jurafsky/pubs/ranganath2013.pdf)
- `question_words`: a list of words associated with questions