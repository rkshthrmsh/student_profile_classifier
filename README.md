# student_profile_classifier beta

Project Aim: To enable prospective graduate students to evaluate their profiles.

Description: In every graduate admission cycle students are expected to furnish a Statment of Purpose (SoP) as part of the application.
This project aims to help students self-evaluate their SoPs for the particular program they are applying to.

Method:
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
