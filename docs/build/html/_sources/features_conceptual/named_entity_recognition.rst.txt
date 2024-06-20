.. _named_entity_recognition:

Named Entity Recognition
========================

High-Level Intuition
*********************
This feature detects whether a user is talking about (or to) someone else in a conversation.

Citation
*********
N/A

Implementation Basics 
**********************
In conversations, named entities often matter: angry statements mean something different when they are directed at no one in particular, versus when they are directed at someone (e.g., who is being blamed for something). 
This feature uses a named entity recognizer (https://spacy.io/api/entityrecognizer) to identify whether someone is talking about (or to) someone else in a conversation. 

Implementation Notes/Caveats 
*****************************
Users should pass in a training file and a threshold for confidence in the FeatureBuilder constructor. The parameter names are as follows:

1) 'ner_training_df': This parameter expects a pandas DataFrame that contains the training data for named entity recognition.

.. list-table:: Named Entity Training Examples
   :widths: 50 25
   :header-rows: 1

   * - sentence_to_train
     - name_to_train
   * - Helena’s idea sounds great!
     - Helena
   * - I agree with Emily, what does everyone else think?
     - Emily
   * - I think we can also work with Shruti’s idea.
     - Shruti
   * - Maybe we should also ask Amy about this
     - Amy

The feature will not run without a provided training file. The file should contain ¼ of the quantity of named entities you expect to see as examples. For example, in a dataset with 100 named entities, the training file should provide 25 examples. 

2) 'ner_cutoff': This integer parameter specifies the threshold for confidence score for each prediction.

Each predicted named entity is associated with a confidence score that evaluates the probability of prediction of each entity. Users can pass in a cutoff value for the confidence scores. If this value is not provided, the default value is 0.9. 

The model was tested on a dataset of 100 sentences with 50 unique names. Here are the following evaluation metrics:

Precision: 0.9855072464
Recall: 0.68

Interpreting the Feature 
*************************
This feature will output the number of named entities in a message, the named entity, and its confidence scores. This is an example output format:

.. list-table:: Output File
   :widths: 40 20 20 40
   :header-rows: 1

   * - message
     - expected_value
     - num_named_entity
     - named_entities
   * - Helena's idea sounds great!
     - Helena
     - 1
     - ((Helena, 1.0))
   * - Sounds great, Emily
     - Emily
     - 1
     - ((Emily, 0.95))
   * - See you next week, Shruti!
     - Shruti
     - 1
     - ((Shruti, 0.9992))
   * - Priya, did you see Amy today?
     - Priya, Amy
     - 2
     - ((Priya, 0.99954), (Amy, 0.9123))

Related Features 
*****************
N/A
