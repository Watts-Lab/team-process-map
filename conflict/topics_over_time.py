from bertopic import BERTopic
import ssl
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from umap import UMAP
from matplotlib.lines import Line2D
import plotly
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords

# Disable SSL verification - because it is annoying
ssl._create_default_https_context = ssl._create_unverified_context

def fit_model_guided(documents,seed_topic_list):
    '''
    Fit the model based on the message column. Each message is a document 

    @param
    documents: The list of documents/messages we want to fit the model on
    seed_topic_list: The list of topics based on human intuition
    '''

    #umap model initialized to random_state 2 to eliminate stochasticity 
    umap_model = UMAP(n_neighbors=15, n_components=5,min_dist=0.0, metric='cosine', random_state=2)
    model = BERTopic(top_n_words=30,umap_model=umap_model,seed_topic_list=seed_topic_list)
    model.fit_transform(documents)
    return model

def fit_model_unguided(documents):
    '''
    Fit the model based on the message column. Each message is a document 

    @param
    documents: The list of documents/messages we want to fit the model on
    '''

    #umap model initialized to random_state 2 to eliminate stochasticity 
    umap_model = UMAP(n_neighbors=15, n_components=5,min_dist=0.0, metric='cosine', random_state=42)
    model = BERTopic(top_n_words=30,umap_model=umap_model)
    model.fit_transform(documents)
    return model


def get_top_topics(topic_model,n):
    '''
    Get the top n topics from the model
    @param  
    model: The BERTopic model
    n:the number of topics we want (This is excluding topic -1)
    '''
    return_list = []
    for i in range(0, n): 
        topic_list = topic_model.get_topic(i)
        return_list.append(topic_list)  
    return return_list  


def reduce_chunks(num_rows, max_num_chunks):
    '''
    Reduce the number of chunks

    @param
    num_rows: The total number of rows in the dataset
    max_num_chunks: The number of chunks to be created
    '''

    if (num_rows < max_num_chunks * 2):
        max_num_chunks = int(num_rows / 2)
    if max_num_chunks < 1:
        return 1
    else:
        return max_num_chunks
     
def assign_chunk_nums(chat_data, num_chunks):
    '''
    Assign chunk numbers to the chats within each conversation

    @param
    chat_data: The chat dataframe
    max_num_chunks: The number of chunks to be created
    '''

    # Calculate the total number of rows per conversation
    conversation_lengths = chat_data.groupby('conversation_num').size()

    chunks = conversation_lengths.apply(lambda x: reduce_chunks(x, num_chunks))

    # Calculate the chunk size based on the total number of conversations
    chunk_size = np.ceil(conversation_lengths / chunks) 
    
    for i, group in chat_data.groupby('conversation_num'): # for each group
        chunk_num = 0
        counter = 0

        for chat_id in group.index.values: # iterate over the index values
            chat_data.at[chat_id, 'chunk'] = int(chunk_num)

            counter += 1

            #if counter = 1 for the last row of a group (implies last chunk has one element), and the chunk num > 0, then just make the last one - 1
            if counter == 1 and chunk_num > 0 and chat_id == group.index.values[-1]:
                chat_data.at[chat_id, 'chunk'] = int(chunk_num - 1)

            if counter == chunk_size[i] and chunk_num < chunks[i] - 1: # assign any extras to the last chunk
                chunk_num += 1
                counter = 0    

    return(chat_data)

def create_chunks(df,num_chunks):
    '''
    Divides the conversation into n equal chunks of time and adds a label for each chunk
    If there are no timestamps, the conversation is divided based on equal number of rows
    @param: 
    df : The conversation dataframe
    num_chunks: the desired number of chunks
    '''

    #check if there are timestamps
    if 'timestamp' in df.columns:

        # Convert timestamp column to DateTime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Calculate the total duration of the conversation
        total_duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
        # total_duration = int(df['duration'][0])

        # Calculate the duration of each chunk
        chunk_duration = total_duration / num_chunks

        if chunk_duration == 0:
            chunk_duration = 1

        # Add a new column for chunk number
        df['chunk'] = -1 

        # Assign the chunk number for each row
        for index, row in df.iterrows():

            #get the timestamp 
            timestamp = row['timestamp']

            #calculate the chunk number
            chunk_number = int(((timestamp - df['timestamp'].min())).total_seconds() / chunk_duration)

            #restrict the range of the chunks from 0 to num_chunks - 1
            if chunk_number >= num_chunks:
                df.at[index, 'chunk'] = num_chunks - 1
            else:
                df.at[index, 'chunk'] = chunk_number
    
    #chunk into parts based on the number of messages
    else:
        assign_chunk_nums(df, num_chunks)


def extract_embeddings(model,df):

    '''
    Extract the SBERT Embeddings from the model

    @param
    model: The BERTopic model
    df: The conversation dataframe
    '''

    embeddings_matrix = model._extract_embeddings(df['message'].tolist())
    return embeddings_matrix

def get_rep_docs(model,topic_num):

    '''
    Get the representative docs for a topic and convert it to a df

    @param
    model: The BERTopic model
    topic_num: the number of topics we desire to extract
    '''
    data = model.get_representative_docs(topic_num)
    df = pd.DataFrame(data,columns=['rep_docs'])
    return df


def create_bert_vectors(df,on_column):
    '''
    Vectorize the documents for the topic

    @param
    df: The conversation dataframe
    on_column: the column for which we want to create text vectors.
    '''

    docs = df[on_column].tolist()

    #check that the user is trying to vectorize text data
    if type(docs) is str:
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    else:
        print("You cannot vectorize data which is not text!")



def get_embeddings_top_topics(top_topics,model):
    '''
    Get the embeddings for the top topics across the conversation

    @param

    top_topics: The top n topics produced by the BERTopic model
    model: the BERTopic model
    '''
    
    embeddings_for_top_topics = []
    
    #for each topic
    for j in range(0,len(top_topics)):
        
        #get the representative documents 
        rep_docs_df = get_rep_docs(model,j)

        #create embeddings of the representative documents for each topic
        topic_embeddings = create_bert_vectors(rep_docs_df,'rep_docs')

        embeddings_for_top_topics.append(topic_embeddings) 
        
    return embeddings_for_top_topics


def get_embeddings_per_chunk(df,num_chunks):
    '''
    Get the embeddings for each chunk

    @param
    df: the conversation dataframe
    num_chunks: the number of chunks
    '''

    chunk_embeddings_list = []

    for i in range(0,num_chunks):
        convo = df[df['chunk'] == i]

        #if the df is empty, i.e there is a pause, create a pseudo 384-dimension embeddings with all values as 0
        if convo.empty:            
            #create n arrays of length 384 (number of dimensions)
            chunk_embeddings = np.zeros((1, 384))

        else:
            chunk_embeddings = create_bert_vectors(convo,'message')
        chunk_embeddings_list.append(chunk_embeddings)
    
    return chunk_embeddings_list

def get_embeddings_per_convo(df):

    '''
    Get the embeddings for each chunk

    @param
    df: the conversation dataframe
    '''

    convo_max_embeddings_list = []
    min_chat = df['conversation_num'].min()
    max_chat = df['conversation_num'].max() + 1

    for i in range(min_chat,max_chat):
        convo = df[df['conversation_num'] == i]

        #if the df is empty, i.e there is a pause, create a pseudo 384-dimension embeddings with all values as 0
        if convo.empty:            
            #create n arrays of length 384 (number of dimensions)
            chunk_embeddings = np.zeros((1, 384))

        else:
            chunk_embeddings = create_bert_vectors(convo,'message')

        convo_max_embeddings_list.append(chunk_embeddings)
    
    return convo_max_embeddings_list

def get_similarity(embeddings_per_chunk,embeddings_top_topics):

    '''
    Calculate the cosine similarity betwee 

    @param
    embeddings_per_chunk: embeddings for each chunk of the conversation
    embeddings_top_topics: embeddings of the documents from which the top topics are derived

    Output format: rows - every chunk, cols - topics 
    '''

    topics_per_chunk = []
    for i in range (0,len(embeddings_per_chunk)):
        
        #calculate the cosine similarity for the topic and the chunk
        embeddings = np.mean(embeddings_per_chunk[i], axis=0) #get the average vectors for the chunk(if there are 4 docs, we get the average of 4 docs for each of the 384 dimensions)
        embeddings = np.nan_to_num(embeddings, nan=0) #replace any nan vectors with 0
        embeddings = embeddings.reshape(1, -1) #reshape to a 2D matrix (needed for the cosine similarity function)

        topics = []
        
        # 
        for j in range(0,len(embeddings_top_topics)):
            
            topic = np.mean(embeddings_top_topics[j], axis=0) #get the average vectors for the chunk(if there are 4 docs, we get the average of 4 docs for each of the 384 dimensions)
            topic = np.nan_to_num(topic, nan=0) #replace any nan vectors with 0
            topic = topic.reshape(1, -1) #reshape to a 2D matrix (needed for the cosine similarity function)

            #calculate the cosine similarity betwen the topic embeddings and the chunk embeddings
            cos = cosine_similarity(embeddings,topic)
            topics.append(cos[0][0])

        topics_per_chunk.append(topics)
    
    return topics_per_chunk


def create_topics_without_stopwords(top_topics):

    '''
    Remove the stop words from the topic to visualize the topics better

    @param
    top_topics: The top topics without stopwords
    '''
    cols = []
    for i in range(0,len(top_topics)):
        filtered_words = [word[0] for word in top_topics[i] if word[0].lower() not in stopwords]
        cols.append(filtered_words)

    topic_list = []
    for i in range(0,len(cols)):
        topic = ""
        for j in range(0,len(cols[i])):
            topic = topic + cols[i][j] +", "

        topic_list.append(topic)
    
    return topic_list
    