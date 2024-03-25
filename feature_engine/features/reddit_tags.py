import numpy as np
import string
import re

"""
file: reddit_tags.py
---
Detects Reddit-Specific HTML tags and other features
"""

"""
function: count_all_caps

Returns the number of all-caps words in a message
"""

def count_all_caps(text):
    words = text.split()
    # Check if the word is all uppercase, is alphabetical, and has more than one letter. Differentiating all caps vs acronyms?
    all_caps_count = sum(
        1 for word in words 
        if word.strip(string.punctuation).isupper()  
        and word.strip(string.punctuation).isalpha() 
        and len(word.strip(string.punctuation)) > 1 
        or (word.strip(string.punctuation) != "I" and len(word.strip(string.punctuation)) == 1)
    ) 
    return all_caps_count

"""
function: count_links

Returns the number of links in a message.
"""
def count_links(text):
    link_pattern = r'http[s]?://[^\s]+|\b\S+?\.(com|org|net|edu|gov|io)\b'
    links = re.findall(link_pattern, text)
    return len(links)

"""
function: count_user_references

Returns the number of user references in a message, indicated by format u/username.
"""
def count_user_references(text):
    user_refs = re.findall(r'\bu/[^\s]+', text)
    return len(user_refs)

"""
function: count_bold

Returns the number of **bolded**, *italicized*, or ***bolded and italicized*** words in a message.
"""
def count_emphasis(text):
    formatted_texts = re.findall(r'\*{1,3}(.*?)\*{1,3}', text)
    return len(formatted_texts)

"""
function: count_bullet_points

Returns the number of bullet points in a message starting with asterisks or dashes.
"""
def count_bullet_points(text):
    normalized_text = text.replace('\\n', '\n')
    bullet_points = re.findall(r'^[\*\-] .+', normalized_text, flags=re.MULTILINE)
    return len(bullet_points)

"""
function: count_numbering

Returns the number of numberings in a message, indicated by a format like "1. ".
"""
def count_numbering(text):
    normalized_text = text.replace('\\n', '\n')
    numberings = re.findall(r'^\d+\. .+', normalized_text, flags=re.MULTILINE)
    return len(numberings)

"""
function: count_line_breaks

Returns the number of paragraphs / line breaks in a message.
"""
def count_line_breaks(text):
    normalized_text = text.replace('\\n', '\n').replace('\\r', '\n').replace('\r\n', '\n').replace('\r', '\n')
    text_single_breaks = re.sub(r'\n+', '\n', normalized_text)
    line_break_count = text_single_breaks.count('\n') + 1  
    return line_break_count

"""
function: count_quotes

Returns the number instances of text enclosed in quotation marks in a message.
"""
def count_quotes(text):
    double_quoted_texts = re.findall(r'"[^"\\]*(?:\\.[^"\\]*)*"', text)
    single_quoted_texts = re.findall(r"'[^'\\]*(?:\\.[^'\\]*)*'", text)    
    return len(double_quoted_texts) + len(single_quoted_texts)

"""
function: count_responding_to_someone

Returns the number of block quote responses, indicating if the message is quoting someone else by ">" or "&gt;".
"""
def count_responding_to_someone(text):
    normalized_text = text.replace('&gt;', '>')
    pattern = r'^>.*'
    responses = re.findall(pattern, normalized_text, re.M)
    return len(responses)

"""
function: count_ellipses

Returns the number of ellipses (three or more consecutive dots) in a message.
"""
def count_ellipses(text):
    ellipses = re.findall(r'\.{3,}', text)
    return len(ellipses)

"""
function: count_parentheses

Returns the number of instances of text enclosed in parentheses in a message (includes nested parentheses).
"""
def count_parentheses(text):
    count = 0
    stack = []

    for char in text:
        if char == '(':
            stack.append(char)
        elif char == ')' and stack:
            stack.pop()
            count += 1

    return count

"""
function: count_emojis

Returns the number of instances of emojis in a message.
"""
def count_emojis(text):
    emoji_pattern = r'[:;]-?\)+'
    emojis = re.findall(emoji_pattern, text)
    return len(emojis)


# print(count_all_caps("HELLO WORLD, THIS IS A TEST. hi HI. hi HI hi HI"))  # Test count_all_caps

# print(count_links("Check out this [link](https://example.com) and this one http://example.org"))  # Test count_links

# print(count_user_references("Hello u/user1 and u/user2, hi hi hi?"))  # Test count_user_references

# print(count_emphasis("This is **bold**, *italics*, and this is not. This is ***bolded and italicized***"))  # Test count_emphasis

# print(count_bullet_points("* item 1\n* item 2\n- item 3"))  # Test count_bullet_points

# print(count_numbering("1. first\n2. second\n3. third"))  # Test count_numbering

# print(count_line_breaks("This is the first line.\nThis is the second line.\nThis is the third line."))  # Test count_line_breaks

# print(count_line_breaks("this is a line with\rA different kind of return value\rUsing carriage return instead of the newline character"))  # Test count_line_breaks

# print(count_quotes("\"This is a quote.\" She said, \"Here's another.\""))  # Test count_quotes

# print(count_quotes("\"I can't believe you use single quotes to quote people,\" she said. \"Well, he replied, \'sometimes single quotes are useful when you nest quotes inside other quotes,\' according to my English teacher\" Then she said: \'okay\'"))  # Test count_quotes

# print(count_responding_to_someone("> here I am making a quote\nI respond to it\n> I quote again\nI respond to that too"))  # Test count_responding_to_someone

# print(count_ellipses("Well... I'm not sure... Maybe..."))  # Test count_ellipses

# print(count_parentheses("This is a sentence (with some text in parentheses)."))  # Test count_parentheses
