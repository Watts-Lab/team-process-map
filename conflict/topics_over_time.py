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

    @return list of top n topics

    '''
    return [topic_model.get_topic(i) for i in range(n)]



def reduce_chunks(num_rows, max_num_chunks):
    '''
    Reduce the number of chunks

    @param
    num_rows: The total number of rows in the dataset
    max_num_chunks: The number of chunks to be created

    @return
    maximum number of chunks possible
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

    @return
    conversation df with a column for chat numbers
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

    @return
    embeddings used by the BERTopic model
    '''

    embeddings_matrix = model._extract_embeddings(df['message'].tolist())
    return embeddings_matrix

def get_rep_docs(model,topic_num):

    '''
    Get the representative docs for a topic and convert it to a df

    @param
    model: The BERTopic model
    topic_num: the number of topics we desire to extract

    @return 
    representative documents for each topic
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

    @return
    list of embeddings per topic
    '''

    #create embeddings of the representative documents for each topic
    process_topic = lambda topic_index: create_bert_vectors(get_rep_docs(model, topic_index), 'rep_docs')

    # Use a list comprehension to apply the lambda function to each topic
    embeddings_for_top_topics = [process_topic(j) for j in range(len(top_topics))]

    # Return the list of embeddings_for_top_topics
    return embeddings_for_top_topics



def get_embeddings_per_chunk(df,num_chunks):
    '''
    Get the embeddings for each chunk

    @param
    df: the conversation dataframe
    num_chunks: the number of chunks

    @return
    list of embeddings per chunk
    '''

    # Define a lambda function to process each chunk and return embeddings
    #if the df is empty, i.e there is a pause, create a pseudo 384-dimension embeddings with all values as 0
    process_chunk = lambda i: np.zeros((1, 384)) if df[df['chunk'] == i].empty else create_bert_vectors(df[df['chunk'] == i], 'message')

    # Use a list comprehension to apply the lambda function to each chunk
    chunk_embeddings_list = [process_chunk(i) for i in range(num_chunks)]

    # Return the list of chunk_embeddings_list
    return chunk_embeddings_list


def get_embeddings_per_convo(df):

    '''
    Get the embeddings for each chunk

    @param
    df: the conversation dataframe

    @return
    list of embeddings per conversation
    '''

    convo_max_embeddings_list = []
    min_chat = df['conversation_num'].min()
    max_chat = df['conversation_num'].max() + 1

    # Define a lambda function to process each conversation number and return embeddings
    #if the df is empty, i.e there is a pause, create a pseudo 384-dimension embeddings with all values as 0
    process_conversation = lambda i: np.zeros((1, 384)) if df[df['conversation_num'] == i].empty else create_bert_vectors(df[df['conversation_num'] == i], 'message')

    # Use a list comprehension to apply the lambda function to each conversation number
    convo_max_embeddings_list = [process_conversation(i) for i in range(min_chat, max_chat)]

    # Return the list of convo_max_embeddings_list
    return convo_max_embeddings_list


def get_similarity(embeddings_per_chunk,embeddings_top_topics):

    '''
    Calculate the cosine similarity betwee 

    @param
    embeddings_per_chunk: embeddings for each chunk of the conversation
    embeddings_top_topics: embeddings of the documents from which the top topics are derived

    @return 
    cosine similarity between the average embeddings of each chunk and each topic where rows - every chunk, cols - topics 
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

    @return data without stopwords

    '''

    # filter stopwords
    filter_stopwords = lambda topic: [word[0] for word in topic if word[0].lower() not in stopwords]

    # apply it to all columns
    cols = [filter_stopwords(top_topics[i]) for i in range(len(top_topics))]

    # Now, cols will contain the filtered words for each topic
    topic_list = []
    for i in range(0,len(cols)):
        topic = ""
        for j in range(0,len(cols[i])):
            topic = topic + cols[i][j] +", "

        topic_list.append(topic)
    
    return topic_list


def plot_topics_over_time(df,num_chunks,num_topics,seed_topic_list,approach_type):

    '''
    Calculates the cosine similarities between the average embeddings of each chunk and each conversation-topic

    @param
    df: conversation dataframe
    num_chunks: number of chunks to divide each conversation into
    num_topics: number of topics
    seed_topic_list: list of seed topics (used only in case of guided topic modeling)
    approach_type: guided/unguided

    @return: df of cosine similarities, where rows are the chunks, columns are Convo-Topics
    '''

    #get the total number of conversations. +1 as the range in python is 0 to n, exclusive of n
    total_convos = df['conversation_num'].max() + 1
    min_convo = df['conversation_num'].min()

    model = None

    if approach_type == 'guided':
        #fit the model
        model = fit_model_guided(df['message'].tolist(),seed_topic_list)
    else:
        model = fit_model_unguided(df['message'].tolist())

    #reduce the topics. We reduce to num_topics + 1 as there will always be a topic -1 with irrelevant topics
    model.reduce_topics(df['message'].tolist(),num_topics+1)

    #get the topics after reduction
    top_topics = get_top_topics(model,num_topics)

    #get the embeddings for the top topics
    embeddings_top_topics = get_embeddings_top_topics(top_topics,model)

    #remove the stopwords from the topics and recreate the topics without the stopwords
    topic_list = create_topics_without_stopwords(top_topics)

    #create an empty df. We will append columns for each Convo-Topic 
    df_for_plotting = None

    #iterate through all the conversations (e.g. 347 in Juries)
    for i in range (min_convo,total_convos):

        print("Convo Number " +str(i))

        #filter the df for a specific conversation number
        df_convo = df[df['conversation_num'] == i]

        #create chunks by dividing the conversation into equal units of time
        create_chunks(df_convo,num_chunks)

        #get the embeddings for each chunk
        embeddings_per_chunk = get_embeddings_per_chunk(df_convo,num_chunks)

        #get the similarity between the topic embeddings and the chunk embeddings
        topics_per_chunk = get_similarity(embeddings_per_chunk,embeddings_top_topics)

        #convert the similarity matrix to a dataframe
        topics_df = pd.DataFrame(topics_per_chunk)

        new_column_headings = []

        #column heading for the current conversation. For e.g. 
        # if it is conversation 131 and we have 2 topics, the headings will be Convo 131 Topic 0, Convo 131 Topic 1
        new_column_headings = [f"Convo {i} Topic {j}" for j in range(len(topic_list))]

        # Now, new_column_headings will contain the desired column headings
        if i == min_convo:#if its the first conversation, we don't need to append the df to an existing df
            df_for_plotting = topics_df
            df_for_plotting.columns = new_column_headings
        else:
            topics_df.columns = new_column_headings
            df_for_plotting = df_for_plotting.join(topics_df)

    #print the top topics
    for i in range(0,len(topic_list)):
        print("Topic " +str(i)+": "+topic_list[i])
    
    #return the df: rows are the chunks, columns are Convo-Topics
    return df_for_plotting


#unpivot the data
def unpivot_data(df,bucket_df):
    '''
    @param
    df: The conversation dataframe
    bucket_df: Pivot data after calculated cosine similarity

    @return:
    unpivoted data frame with cosine similarities
    '''

    # Add 'chunk' column with values 0, 1, 2, ...
    df['chunk'] = df.index

    # Reset the index to have a numeric range index
    df.reset_index(drop=True, inplace=True)

    # Unpivot the DataFrame to melt the data
    df_unpivoted = pd.melt(df, id_vars=['chunk'], var_name='convo_topic', value_name='cosine_similarity')

    # Split 'convo_topic' into 'Convo' and 'Topic', and convert them into numbers
    df_unpivoted[['convo', 'topic']] = df_unpivoted['convo_topic'].str.extract(r'Convo (\d+) Topic (\d+)')
    df_unpivoted['convo'] = df_unpivoted['convo'].astype('int')
    df_unpivoted['topic'] = df_unpivoted['topic'].astype('int')

    # Drop the original 'convo_topic' column
    df_unpivoted.drop(columns=['convo_topic'], inplace=True)

    # Reorder the columns so that 'Cosine Similarity' is the first column
    df_unpivoted = df_unpivoted[['cosine_similarity', 'chunk', 'convo', 'topic']]

    # Perform the VLOOKUP-like operation using merge
    result_df = pd.merge(df_unpivoted, bucket_df, left_on='convo',right_index=True, how='left')

    return result_df


def normalize(x):
    '''
    Normalization function around the mean
    '''
    return (x - x.mean()) / x.std()


def get_normalized_buckets(df,performance_metric):
    '''
    Adds the label "Below Mean

    @param
    df:conversation datafram
    performance_metric: performance metric for the respective task

    @return
    df with the label based on the normalized performance metric
    '''

    #filter to extract the performance metric
    df = df[[performance_metric]]

    #normalize the performance score
    df['normalized_performance_score'] = df[performance_metric].transform(normalize)

    #get the mean aftet normalization
    mean = df['normalized_performance_score'].mean()

    # Define a lambda function to label rows based on the mean
    label_rows = lambda value: 'Below Mean' if value < mean else 'Above Mean'

    # Apply the lambda function to create the 'normalized_label' column
    df['normalized_label'] = df['normalized_performance_score'].apply(label_rows)

    return df

# Define the function to calculate the mean from a bootstrap sample
def bootstrap_mean(data, n_bootstraps=1000):
    '''
    @param
    data:
    n_bootstraps : number of bootstraps, default is 1000
    '''
    valid_data = data.dropna()  # Remove NaN values after normalization
    if len(valid_data) > 1:  # Check if the valid_data has more than one value
        means = []
        for _ in range(n_bootstraps):

            #get an array of random samples
            sample = np.random.choice(valid_data, size=len(valid_data), replace=True)

            #append the mean of random sample to output. 
            means.append(sample.mean())
        # the len will be 1000 due to 1000 bootstraps and each element will be the mean of n random samples (n being the population size)
        return means 
    else: #we cannot bootstrap in this case, as we have only 1 value.
        return np.nan


def normalize_cosine_similarity(df,confidence_interval):
    '''
    @param
    df: the conversation dataframe
    confidence_interval: The % for confidence intervals (must be a whole number)

    @return: A df containing the normalized cosine similarities for each bucket-topic-chunk.
    '''

    # Group the DataFrame by 'chunk', 'topic', and 'label_2_buckets'
    grouped = df.groupby(['chunk', 'topic', 'normalized_label'])['cosine_similarity']

    lower_bound = (100 - confidence_interval) / 2
    upper_bound = 100 - lower_bound

    # Calculate the bootstrap confidence intervals for the mean of 'normalized_cosine_similarity' - This will give us the CIs for each chunk-topic-label
    #basically, we will take the 1000 values from above, create a distribution, and get the upper and lower bound based on the confidence interval specified
    confidence_intervals = grouped.apply(bootstrap_mean).dropna().apply(lambda x: np.percentile(x, [lower_bound, upper_bound]))

    #get the aggregated DF. 
    average_df = df.groupby(['chunk', 'topic', 'normalized_label'], as_index=False).agg(cosine_similarity_mean=('cosine_similarity', 'mean'))

    # Create a new DataFrame to store the confidence intervals 
    confidence_intervals_df = pd.DataFrame(confidence_intervals.tolist(), columns=['Lower CI', 'Upper CI'])

    # Concatenate the confidence intervals DataFrame with the original DataFrame
    result_df = pd.concat([average_df, confidence_intervals_df], axis=1)

    #add the boostrapped results to all the other convos
    return result_df

def convert_convo_nums(df):
    '''
    Use factorize to replace alphanumeric strings with conversation numbers
    '''

    df['conversation_num'] = pd.factorize(df['conversation_num'])[0]


def plot3(df,approach_type):
    '''
    plot the Topics over Time graph

    @param
    df: conversation dataframe
    approach_type: guided or unguided
    '''

    # Define line styles based on 'normalized_label'
    line_dash_map = {
        ('Above Mean',): 'solid',
        ('Below Mean',): 'dot'
    }

    # Create a color map for different topics
    topic_color_map = {
        topic: f'hsl({360 * topic / df["topic"].nunique()}, 50%, 50%)'
        for topic in df['topic'].unique()
    }

    # Create the figure and subplots
    fig = go.Figure()

    # Iterate over unique topics
    for topic in df['topic'].unique():
        topic_df = df[df['topic'] == topic]

        # Get the color for the current topic
        color = topic_color_map[topic]

        # Iterate over unique normalized_labels for each topic
        for label in topic_df['normalized_label'].unique():
            label_df = topic_df[topic_df['normalized_label'] == label]

            # Create line trace for each topic-chunk-normalized_label combination
            fig.add_trace(go.Scatter(
                x=label_df['chunk'],
                y=label_df['cosine_similarity_mean'],
                mode='lines',
                name=f'Topic {topic} ({label})',
                line=dict(color=color, dash=line_dash_map[(label,)]),
            ))

            # Create vertical bars for confidence intervals at each chunk position
            for i in label_df.index:
                fig.add_shape(
                    go.layout.Shape(
                        type='line',
                        x0=label_df.loc[i, 'chunk'],
                        x1=label_df.loc[i, 'chunk'],
                        y0=label_df.loc[i, 'Lower CI'],
                        y1=label_df.loc[i, 'Upper CI'],
                        line=dict(color=color, dash=line_dash_map[(label,)], width=2),
                    )
                )

        # Calculate the average for the topic-chunk combination
        avg_cosine_similarity = topic_df.groupby('chunk')['cosine_similarity_mean'].mean()

        # Create a thick line trace for the average values
        fig.add_trace(go.Scatter(
            x=avg_cosine_similarity.index,
            y=avg_cosine_similarity,
            mode='lines',
            name=f'Topic {topic} (Average)',
            line=dict(color=color, width=3),
        ))

    if approach_type == 'unguided':
        # Customize layout
        fig.update_layout(
            title='Topics Over Time - Unguided',
            xaxis_title='Chunk',
            yaxis_title='Cosine Similarity Mean',
            legend_title='Topic and Label',
            legend=dict(yanchor='top', y=1.03, xanchor='left', x=1.03),
            xaxis=dict(type='category', tickmode='linear'),
            margin=dict(l=80, r=80, t=100, b=80),  # Adjust the margin around the plot
        )
    else:
        # Customize layout
        fig.update_layout(
            title='Topics Over Time - Guided',
            xaxis_title='Chunk',
            yaxis_title='Cosine Similarity Mean',
            legend_title='Topic and Label',
            legend=dict(yanchor='top', y=1.03, xanchor='left', x=1.03),
            xaxis=dict(type='category', tickmode='linear'),
            margin=dict(l=80, r=80, t=100, b=80),  # Adjust the margin around the plot
        )
        
    # Show the plot
    fig.show()


def plot_after_normalization(df_for_plotting,df,performance_metric,confidence_interval,approach_type):
    '''

    @param
    df_for_plotting: df of cosine similarties where rows are the chunk numbers, columns are Convo-Topics
    df: The conversation dataframe
    performance_metric: The performance metric for the respective task
    confidence_interval:The % for confidence intervals (must be a whole number)
    approach_type: guided or unguided
    '''

    bucket_df = get_normalized_buckets(df,performance_metric)
    
    #
    output = unpivot_data(df_for_plotting,bucket_df,performance_metric)

    #bootstrap confidence intervals for c
    bootstrapped = normalize_cosine_similarity(output,confidence_interval)

    #plot the data with bootstrapped confidence intervals
    plot3(bootstrapped,approach_type)


def train_and_plot(chat_df,seed_topic_list,num_topics,num_chunks,performance_metric,confidence_interval,approach_type):
    '''
    Ulimate function where everything comes together
    @param

    chat_df: The conversation dataframe
    seed_topic_list: The list of seed topics for guided topic modeling
    num_topics: The number of topics
    num_chunks: The number of chunks
    performance_metric: The performance metric for the respective task
    confidence_interval: The % for confidence intervals (must be a whole number)
    approach_type: guided or unguided

    '''

    #if the conversation IDs are not numbers, convert them to numbers
    if pd.api.types.is_string_dtype(chat_df['conversation_num']):
        convert_convo_nums(chat_df)

    #generate the dataframe for plotting
    df_for_plotting = plot_topics_over_time(chat_df,num_chunks,num_topics,seed_topic_list,approach_type)

    #normalize the performance metric, bootstrap confidence intervals and plot them
    plot_after_normalization(df_for_plotting,chat_df,performance_metric,confidence_interval,approach_type)
    