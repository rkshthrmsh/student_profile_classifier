import os
import numpy as np
import nltk
import tensorflow as tf



############# File Clean-up ################################
''' Steps:
    1. Tag sentence and keep only lemmatized Nouns and Verbs
    2. For every entry remove duplicate words.
    3. Return number of negative samples and positive samples.
'''
def file_cleanup():
    num_nsamples, num_psamples = 0, 0     # Store number of negative samples and positive samples
    f_clean = open('clean_file.txt', 'w')
    for file in ['neg_samples.txt', 'pos_samples.txt', 'profile.txt']:
    #for file in ['test.txt']:    # test feature
        with open(file, 'r') as f_sample:
            for line in f_sample:
                tagged_line = nltk.pos_tag(nltk.word_tokenize(line))    # Tag line
                imp_words = [nltk.WordNetLemmatizer().lemmatize(i.lower(), j[0].lower()) for i, j in tagged_line if j[0] in ['N', 'V']] # Lemmatize and keep only Nouns and Verbs
                imp_words = [i for i in imp_words if (i[0].isalpha() and len(i) > 2)] # Keep only words that are more than 2 letters
                uniq_words = sorted(set(imp_words)) # Remove duplicate words
                uniq_line = " ".join(str(i) for i in uniq_words) # Make a line for writing into a file
                f_clean.write('%s\n' %uniq_line)

                if file == 'neg_samples.txt':
                    num_nsamples += 1
                elif file == 'pos_samples.txt':
                    num_psamples += 1

    f_clean.close()
    return num_nsamples, num_psamples


############# Files to Binary Vector #######################
''' Steps:
    1. Open clean.txt and combine all individually uniquified sentences.
    2. Remove duplicates in this combined line to get all the unique words in the file.
    3. Create feature vectors for each sample based on these unique words.
    4. Save the feature vectors into vector_file.txt
'''
def vectorize_file(num_nsamples, num_psamples):
    with open('clean_file.txt', 'r') as f_clean:
    #with open('test.txt', 'r') as file:  # testing
        combined_line = " ".join(line for line in f_clean)          # Combine all the lines from clean_file.txt
        tokenized_combined_line = nltk.word_tokenize(combined_line) # Tokenize the combined lines
        
        # Clean-up: Drop non-alhpnumeric words and small (< 2 letter) words
        tokenized_combined_line = [i for i in tokenized_combined_line if (i[0].isalpha() and len(i) > 2)]
        uniq_token_line = sorted(set(tokenized_combined_line))      # Uniquify the combined_line
        vector_len = len(uniq_token_line)                           # Number of unique words in uniq_token_line

    f_vector = open('vector_file.txt', 'w')
    with open('clean_file.txt', 'r') as f_clean:
    #with open('test.txt', 'r') as f_clean:     # test feature
        current_line = 0
        for line in f_clean:
            current_line = current_line + 1                     # Keep track of current line number for adding label
            tokenized_line = nltk.word_tokenize(line)
            feature_vector = format(0, '0%sb' %vector_len)      # Instantiate a default feature vector with vector_len number of 0s
            feature_vector_list = list(feature_vector)          # Convert into a list for easy manipulation of list members

            # Create feature vector for the line
            # For every word in tokenized_line assert (set to 1) the appropriate list member of feature_vector_list
            for i in tokenized_line:
                for j in range(vector_len):
                    if i == uniq_token_line[j]:
                        feature_vector_list[j] = '1'
                    else:
                        continue
            # Adding labels to the features
            if current_line < (num_nsamples + 1):
                label = '0'
            elif current_line < (num_nsamples + num_psamples + 1):
                label = '1'
            else:
                label = ''

            # Join all list members and label
            feature_vector = ''.join('%s%s' %((''.join(feature_vector_list), label)))

            # Save feature vector to file
            f_vector.write('%s\n' %feature_vector)

    f_vector.close()
    return uniq_token_line

############# Binary Vectors to labelled data for NN #######################
''' Steps:
    1. From vector_file.txt create a labelled data matrix that can be fed to the neural network.
    2. Each line of the vector_file.txt is converted into a column of the labelled data matrix.
'''
def vector_file2labelled_data():
    with open ('vector_file.txt', 'r') as f_vector:
        
        labelled_data_list = []  # Empty labelled_data_list

        # Populate labelled_data_list with lines from vector_file.txt
        for line in f_vector:
            bin_string = line.strip('\n') # Remove \n from the end of the line
            bin_string_list = list(bin_string) # Convert bin_string to list
            labelled_data_list.append([int(bin_string_list[i]) for i in range(len(bin_string_list))]) # Append each sample to labelled_data_list list

    # labelled_data matrix from labelled_data_list
    labelled_data = np.matrix(labelled_data_list[0:np.size(labelled_data_list) - 1])

    # Features of the profile from labelled_data_list
    profile_features = np.matrix(labelled_data_list[np.size(labelled_data_list) - 1])
    
    return labelled_data, profile_features

############# Neural Network #######################
def nn(labelled_data, prediction_features, hidden1_units):

    # Random array generation for selecting training, cross-validation and test samples
    features_size = np.size(labelled_data, 1) - 1       # Number of features
    samples_size = np.size(labelled_data, 0)            # Number of samples
    rand_arr = np.arange(samples_size)                  # Random array for shuffling samples
    np.random.shuffle(rand_arr)

    # Training Samples
    train_size = int(np.floor(0.9 * samples_size))      # Number of training samples
    
    train_inputs = labelled_data[rand_arr[0:train_size], 0:features_size]    # Training features
    train_labels = labelled_data[rand_arr[0:train_size], features_size]     # Training labels

    # Cross Validation Samples
    #crossval_size = samples_size - train_size           # Number of cross validation samples
    crossval_size = int(np.floor(0.05 * samples_size))
    
    crossval_inputs = labelled_data[rand_arr[train_size:train_size + crossval_size], 0:features_size]   # Cross Validation features
    crossval_labels = labelled_data[rand_arr[train_size:train_size + crossval_size], features_size]     # Cross Validation labels

    # Test Samples
    test_size = samples_size - (train_size + crossval_size)

    test_inputs = labelled_data[rand_arr[train_size + crossval_size:samples_size], 0:features_size]   # Test features
    test_labels = labelled_data[rand_arr[train_size + crossval_size:samples_size], features_size]     # Test labels
                                
    # Input Features
    a0 = tf.placeholder(tf.float32, [None, features_size])

    # Hidden Layer 1
    W1 = tf.Variable(tf.truncated_normal([features_size, hidden1_units],
                                              stddev=1.0 / np.sqrt(float(features_size))), name='W1')
    b1 = tf.Variable(tf.zeros([hidden1_units]), name='b1')
    a1 = tf.nn.sigmoid(tf.matmul(a0, W1) + b1)

    # Final Layer Shallow
    W_final = tf.Variable(tf.truncated_normal([hidden1_units, 1],
                                         stddev=1.0 / np.sqrt(float(hidden1_units))), name='W_final')
    b_final = tf.Variable(tf.zeros([1]), name='b_final')
                                         
    y = tf.nn.sigmoid(tf.matmul(a1, W_final) + b_final)
    
##    # Hidden Layer 2
##    W2 = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
##                                         stddev=1.0 / np.sqrt(float(hidden1_units))), name='W2')
##    b2 = tf.Variable(tf.zeros([hidden2_units]), name='b2')
##    a2 = tf.nn.sigmoid(tf.matmul(a1, W2) + b2)
##    
##    # Final Layer
##    W_final = tf.Variable(tf.truncated_normal([hidden2_units, 1],
##                                         stddev=1.0 / np.sqrt(float(hidden2_units))), name='W_final')
##    b_final = tf.Variable(tf.zeros([1]), name='b_final')
##                                         
##    y = tf.nn.sigmoid(tf.matmul(a2, W_final) + b_final)
    
    # Labels
    y_ = tf.placeholder(tf.float32, [None, 1])

    # Loss
    loss = tf.reduce_mean(tf.square(y - y_))

    # Initialize Session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    #Train
    learning_rate = 0.5
    steps = 1000
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    for i in range(steps):
        sess.run(train_step, feed_dict={a0: train_inputs,
                                        y_: train_labels})
        if np.mod(i, 100) == 0:
                print('%d: %f' %(i, sess.run(tf.reduce_mean(loss), feed_dict={a0: train_inputs,
                                                                              y_: train_labels})))

    # Cross Validation Error
    print('Cross Validation Error: %f' %sess.run(tf.reduce_mean(loss), feed_dict={a0: crossval_inputs,
                                                                                  y_: crossval_labels}))

    # Test Results
    for i in range(test_size):
        results = int(sess.run(y, feed_dict={a0: test_inputs[i],
                                             y_: test_labels[i]}) > 0.8)
##        results = sess.run(y, feed_dict={a0: test_inputs[i],
##                                         y_: test_labels[i]})
        print('Label: %d Prediction: %f' %(test_labels[i], results))

    # Prediction
    print('Prediction: %f (True if > 0.8)' %(sess.run(y, feed_dict={a0: prediction_features})))
    
def main():

    # Cleanup the neg_samples.txt and pos_samples.txt files
    # Creates clean_file.txt file with sorted, unique words
    # Returns num_nsamples
    num_nsamples, num_psamples = file_cleanup()

    
    # Convert clean.txt to feature + labels file vector_file.txt
    _ = vectorize_file(num_nsamples, num_psamples)

    # Get labelled data from vector_file.txt
    # Returns labelled data matrix
    labelled_data, prediction_features = vector_file2labelled_data()

    # Feed labelled_data, prediction_features to neural network
    nn(labelled_data, prediction_features, 20)


def main_analysis():

    num_nsamples, num_psamples = file_cleanup()

    uniq_token_line = vectorize_file(num_nsamples, num_psamples)
    uniq_token_line_array = np.array(uniq_token_line)

    labelled_data, _ = vector_file2labelled_data()

    # Sum the elements for all posititve samples row-wise (excluding the label column) and convert into array for list indexability
    freq_words = np.squeeze(np.asarray(np.sum(labelled_data[num_nsamples : , : np.size(labelled_data, 1) -1], axis=0)))

    n = 20
    top_n_index = np.argsort(-freq_words)[:n]
    print(uniq_token_line_array[top_n_index])
