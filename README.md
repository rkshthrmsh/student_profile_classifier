# student_profile_classifier beta

## Project Aim  
To enable prospective graduate students to evaluate their profiles.

## Description  
In every graduate admission cycle students are expected to furnish a Statment of Purpose (SoP) as part of the application.
This project aims to help students self-evaluate their SoPs for the particular program they are applying to.

## Method
  1. __Data Collection__: Samples of admitted and rejected SoPs are collected and collated continuously in the _pos_samples.txt_ and _neg_samples.txt_ files, repsectively.
  2. __Data Clean-up__:
     - Natural Language Processing: Each profile from the above files is tokenized and tagged using `nltk` to extract the important words (nouns, verbs).  
       For example, student profile: `I have always enjoyed science. I studied computer science at XYZ University.`
       Tokenized and tagged profile: `[('I', 'PRP'), ('have', 'VBP'), ('always', 'RB'), ('enjoyed', 'VBN'), ('science', 'NN'), ('.', '.'), ('I', 'PRP'), ('studied', 'VBD'), ('computer', 'NN'), ('science', 'NN'), ('at', 'IN'), ('XYZ', 'NNP'), ('University', 'NNP'), ('.', '.')]`
     - Lemmatize: The extracted words are lemmatized so that each student profile can be compared for the presence or absence of these words.  
       Same example: `['have', 'enjoy', 'science', 'study', 'computer', 'science', 'xyz', 'university']`
     - Uniquifying and sorting: The reduced profile is further simplified by removing duplicate occurences of lemmatized words.  
       Same example: `['computer', 'enjoy', 'have', 'science', 'study', 'university', 'xyz']`
     - Storing: Each sorted and simplified student profile is stored in  _clean_file.txt_.
  3. __Clean Data to Binary Vectors__: 
     - Vector Definition: Simplified profiles are combined as follows to obtain vectors.  
       Simplified Profile 1: `['computer', 'enjoy', 'have', 'science', 'study', 'university', 'xyz']`  
       Simplified Profile 2: `['abc', 'aim', 'become', 'computer', 'learning', 'machine', 'scientist', 'study', 'university']`  
       Combined + uniquified: `['abc', 'aim', 'become', 'computer', 'enjoy', 'have', 'learning', 'machine', 'science', 'scientist', 'study', 'university', 'xyz']` <-- has length 13  
       Assuming combined + uniquified as a vector: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` <-- also has length 13  
       From this,  
       Simplified Profile 1 vector: `[0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1]` <-- based on the words present in Simplified Profile 1  
       Simplified Profile 2 vector: `[1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0]`
     - Storing: The vectors are stored in _vector_file.txt_.
