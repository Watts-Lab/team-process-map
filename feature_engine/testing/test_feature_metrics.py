import pytest
import pandas as pd

test_chat_df =  pd.read_csv("../output/chat/test_chat_level_chat.csv")
test_conv_df =  pd.read_csv("../output/conv/test_conv_level_conv.csv")
num_features_chat = test_chat_df.columns.size - 7
num_tested_chat = test_chat_df['expected_column'].nunique()
num_features_conv = test_conv_df.columns.size - 7
num_tested_conv = test_conv_df['expected_column'].nunique()

with open('test.log', 'w') as f:
    f.write(f'Tested {num_tested_chat} features out of {num_features_chat} chat level features: {num_tested_chat/num_features_chat * 100:.2f}% Coverage!\n')
    f.write(f'Tested {num_tested_conv} features out of {num_features_conv} conv level features: {num_tested_conv/num_features_conv * 100:.2f}% Coverage!\n')
    pass

# generate coverage for tests

@pytest.mark.parametrize("row", test_chat_df.iterrows())
def test_chat_unit_equality(row):
    actual = row[1][row[1]['expected_column']]
    expected = row[1]['expected_value']
    
    try:
        assert round(float(actual), 3) == round(float(expected), 3)
    except AssertionError:

        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST FAILED------\n")
            file.write(f"Testing {row[1]['expected_column']} for message: {row[1]['message_original']}\n")
            file.write(f"Expected value: {expected}\n")
            file.write(f"Actual value: {actual}\n")

        raise  # Re-raise the AssertionError to mark the test as failed


test_ner = pd.read_csv('../output/chat/test_named_entity_chat_level.csv')

@pytest.mark.parametrize("row", test_ner.iterrows())
def test_named_entity_recognition(row):
    if pd.isnull(row[1]['expected_value']):
        assert row[1]['named_entities'] == "()"
    else:
        expected = row[1]['expected_value'].split(',')       
        parsed_actual = row[1]['named_entities'].replace(" ","").replace("(","").replace(")", "").split(',')
        actual = parsed_actual[0::2]

        if actual and actual[-1] == '':
            # Remove the last element
            actual.pop()    

        try:   
            assert len(expected) == len(actual)
            for named_entity in expected:
                assert named_entity.lower().strip() in actual
        except AssertionError:

            with open('test.log', 'a') as file:
                file.write("\n")
                file.write("------TEST FAILED------\n")
                file.write(f"Testing NER for message: {row[1]['message_original']}\n")
                file.write(f"Expected value: {expected}\n")
                file.write(f"Actual value: {actual}\n")

            raise  # Re-raise the AssertionError to mark the test as failed


@pytest.mark.parametrize("conversation_num, conversation_rows", test_conv_df.groupby('conversation_num'))
def test_conv_unit_equality(conversation_num, conversation_rows):
    test_failed = False
    expected_out = ""
    actual_out = ""

    for _, row in conversation_rows.iterrows():
        actual = row[row['expected_column']]
        expected = row['expected_value']
    
    try:
        assert round(actual, 3) == round(expected, 3)
    except AssertionError:
        expected_out = expected
        actual_out = actual
        test_failed = True

    if test_failed:
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST FAILED------\n")
            file.write(f"Testing {row['expected_column']} for conversation_num: {conversation_num}\n")
            file.write(f"Expected value: {expected_out}\n")
            file.write(f"Actual value: {actual_out}\n")

        raise


# testing complex features 
test_chat_complex_df =  pd.read_csv("../output/chat/test_chat_level_chat_complex.csv")

# Helper function to generate batches of three rows
def get_batches(dataframe, batch_size=3):
    batches = []
    rows = list(dataframe.iterrows())
    for i in range(0, len(rows), batch_size):
        batches.append(rows[i:i + batch_size])
    return batches

# Assuming test_chat_complex_df is your DataFrame
batches = get_batches(test_chat_complex_df, batch_size=3)

@pytest.mark.parametrize("batch", batches)
def test_chat_complex(batch):
    feature = batch[0][1]['feature']
    og_result = batch[0][1][feature]
    inv_result = batch[1][1][feature]
    dir_result = batch[2][1][feature]

    inv_distance = og_result - inv_result
    dir_distance = og_result - dir_result

    # calculate ratio between inv and dir
    ratio = dir_distance / inv_distance
    
    try:
        
        assert ratio > 1
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST PASSED------\n")
            file.write(f"Testing {feature} for message: {batch[0][1]['message']}\n")
            file.write(f"Inv message: {batch[1][1]['message']}\n")
            file.write(f"Dir message: {batch[2][1]['message']}\n")
            file.write(f"Ratio (DIR / INV): {ratio}\n")
    except AssertionError:
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST FAILED------\n")
            file.write(f"Testing {feature} for message: {batch[0][1]['message']}\n")
            file.write(f"Inv message: {batch[1][1]['message']}\n")
            file.write(f"Dir message: {batch[2][1]['message']}\n")
            file.write(f"Ratio (DIR / INV): {ratio}\n")

        raise  # Re-raise the AssertionError to mark the test as failed

test_chat_complex_df =  pd.read_csv("../output/conv/test_conv_level_conv_complex.csv")
batches = get_batches(test_chat_complex_df, batch_size=3)

@pytest.mark.parametrize("batch", batches)
def test_conv_complex(batch):
    feature = batch[0][1]['feature']
    og_result = batch[0][1][feature]
    inv_result = batch[1][1][feature]
    dir_result = batch[2][1][feature]

    inv_distance = og_result - inv_result
    dir_distance = og_result - dir_result

    # calculate ratio between inv and dir
    ratio = dir_distance / inv_distance
    
    try:
        
        assert ratio > 1
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST PASSED------\n")
            file.write(f"Testing {feature} for conversation: {batch[0][1]['conversation_num']}\n")
            file.write(f"Inv conversation: {batch[1][1]['conversation_num']}\n")
            file.write(f"Dir conversation: {batch[2][1]['conversation_num']}\n")
            file.write(f"Ratio (DIR / INV): {ratio}\n")
    except AssertionError:
        with open('test.log', 'a') as file:
            file.write("\n")
            file.write("------TEST FAILED------\n")
            file.write(f"Testing {feature} for conversation: {batch[0][1]['conversation_num']}\n")
            file.write(f"Inv conversation: {batch[1][1]['conversation_num']}\n")
            file.write(f"Dir conversation: {batch[2][1]['conversation_num']}\n")
            file.write(f"Ratio (DIR / INV): {ratio}\n")

        raise  # Re-raise the AssertionError to mark the test as failed
