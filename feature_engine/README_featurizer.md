# Featurizer Documentation

The featurizer takes text conversations and transforms them into the following representations:

- A table of `chat-level features`, which generates a unique conversation feature for each chat (aka utterance, or message);
- A table of `conversation-level features`, which generates aggregations of features within each chat at the conversation level.

To set up and run the featurizer from scratch, you should do the following.

# Run separate iPython Scripts
Some features are computationally inefficient to run every time, so the featurizer performs some processing upfront. Before getting started, you should separately run the following iPython notebooks.

## Run ONCE (regardless of number of datasets)
- `features/preprocess_lexicons.ipynb` --> generates `features/lexicons_dict.pkl`

## Run once _per dataset_ (generates dataset-specific pre-processing / embeddings)
The following needs to be run upon initializing the directory:
- `features/process_sent_vectors.ipynb` --> generates `embeddings/*`

The following does not have to be run upon initialization (the outputs are already saved); however, as new datasets are added, this script needs to be re-run for each new dataset.
- `features/positivity_bert_analysis.ipynb` --> generates `sentiment_bert/*`

# Run the main featurizer [Do this every time you want to refresh/generate new features.]

In the terminal, run `python3 featurize.py`.
