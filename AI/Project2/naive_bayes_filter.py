# CS 482
# Project 1 - Naive Bayes Filter
# Author: Dennis Browm
# Import libraries
import numpy as np
import pandas as pd
import argparse

class NaiveBayesFilter:
    def __init__(self, test_set_path):
        self.test_set_path = test_set_path
        self.vocabulary = set()
        self.training_set= None
        self.test_set = None
        self.p_spam = None
        self.p_ham = None
        self.p_word_spam = {}
        self.p_word_ham = {}
        self.spam_len = None
        self.ham_len = None
        self.vocabulary_len = None
        

    def read_csv(self):
        self.training_set = pd.read_csv('train.csv', sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')
        self.test_set = pd.read_csv(self.test_set_path, sep=',', header=0, names=['v1', 'v2'], encoding = 'utf-8')


    def data_cleaning(self):
        #self.read_csv()
        # Make everything lowercase
        self.training_set['v2'] = self.training_set['v2'].str.lower()
        # Remove numbers
        self.training_set['v2'] = self.training_set['v2'].replace(to_replace=r'\d', value='', regex=True)
        # Remove special characters
        self.training_set['v2'] = self.training_set['v2'].replace(to_replace=r'[^a-zA-Z0-9\s]', value='', regex=True)

        stop_words = [
            'a', 'an', 'the',
            'and', 'but', 'or', 'for', 'nor',
            'on', 'in', 'at', 'by', 'with',
            'I', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'you', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their',
            'mine', 'yours', 'hers', 'ours', 'theirs',
            'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'
        ]

        # Stop words to remove
        stop_words_pattern = r'\b(' + '|'.join(stop_words) + r')\b'

        # Remove standalone stop words from the 'v2' column
        self.training_set['v2'] = self.training_set['v2'].replace(to_replace=stop_words_pattern, value='', regex=True)

        self.training_set['tokenized'] = self.training_set['v2'].apply(lambda x: x.split())
        
        for document in self.training_set['v2']:
            # Add to the vocabulary set
            self.vocabulary.update(set(document.split()))
            
        vectorized_documents = []

        # Iterate through each document in the 'v2' column
        for document in self.training_set['v2']:
            # vector with zeros for each word in the vocabulary
            vector = {word: 0 for word in self.vocabulary}

            # Count words in the document
            for word in document.split():
                if word in vector:
                    vector[word] += 1

            # Append the vector
            vectorized_documents.append(vector)

        # Convert the list of dictionaries to a DataFrame
        self.vectorized_column = pd.DataFrame(vectorized_documents)

        self.new_dataframe = pd.concat([self.training_set, self.vectorized_column], axis=1)
        self.new_dataframe = self.new_dataframe[~self.new_dataframe.duplicated(subset=['v1','v2'])]

        columns_to_include = ['v1','v2']
        new_dataframe_sub = self.new_dataframe[columns_to_include]        

        self.result_dict = new_dataframe_sub.set_index('v2').to_dict(orient='index')
        
      
        # DONE - Normalization
        # DONE - Replace addresses (hhtp, email), numbers (plain, phone), money symbols
        # DONE - Remove the stop-words

        # Lemmatization - Graduate Students

        # Stemming - Gradutate Students

        # DONE - Tokenization
 
        # DONE - Vectorization

        # Remove duplicates - Can you think of any data structure that can help you remove duplicates?

        # Create the dictionary
        
        # Convert to dataframe 

        # Separate the spam and ham dataframes
        pass

    def fit_bayes(self):
  
        # probability of spam and ham in data set
        self.p_spam = len(self.training_set[self.training_set['v1'] == 'spam']) / len(self.training_set)
        self.p_ham = 1 - self.p_spam
        
       # print("p(spam): ", self.p_spam)
       # print("p(ham): ", self.p_ham)

        # Spam messages in data
        spam_messages = self.training_set[self.training_set['v1'] == 'spam']
        # Ham messages in data
        ham_messages = self.training_set[self.training_set['v1'] == 'ham']
        
        # Spa, Ham, and vocab lengths
        self.spam_len = len(spam_messages)
        self.ham_len = len(ham_messages)
        self.vocabulary_len = len(set(' '.join(self.training_set['v2']).split()))

       # print("Nspam: ", self.spam_len)
       # print("Nham: ", self.ham_len)
        
        alpha = 1

        spam_word_counts = {}
        ham_word_counts = {}

        # quantize words
        for index, row in spam_messages.iterrows():
            for word in row['v2'].split():
                spam_word_counts[word] = spam_word_counts.get(word, 0) + 1

        for index, row in ham_messages.iterrows():
            for word in row['v2'].split():
                ham_word_counts[word] = ham_word_counts.get(word, 0) + 1
        
        
        # Calculate P(wi|Spam) and P(wi|Ham)
        for word in set(spam_word_counts.keys()).union(set(ham_word_counts.keys())):
            p_word_spam = (spam_word_counts.get(word, 0) + alpha) / (self.spam_len + alpha * self.vocabulary_len)
            p_word_ham = (ham_word_counts.get(word, 0) + alpha) / (self.ham_len + alpha * self.vocabulary_len)

            self.p_word_spam[word] = p_word_spam
            self.p_word_ham[word] = p_word_ham
        
        
   
    def train(self):
        self.read_csv()
        self.data_cleaning()
        self.fit_bayes()
    
    def sms_classify(self, message):
        '''
        classifies a single message as spam or ham
        Takes in as input a new sms (w1, w2, ..., wn),
        performs the same data cleaning steps as in the training set,
        calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn),
        compares them and outcomes whether the message is spam or not.
        '''
        
        # message to lower
        message = message.lower()
        # Remove numbers
        message = ''.join(char if char.isalpha() or char.isspace() else '' for char in message)

        #print(message)
        stop_words = [
            'a', 'an', 'the',
            'and', 'but', 'or', 'for', 'nor',
            'on', 'in', 'at', 'by', 'with',
            'I', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'you', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their',
            'mine', 'yours', 'hers', 'ours', 'theirs',
            'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'
        ]
        # split message into words
        words = message.split()
        
        # Remove stop words
        words = [word for word in words if word not in stop_words]
     
        p_spam_product = 1.0
        p_ham_product = 1.0
        # Calculate the log likelihood of the message for spam and ham
        for word in words:
            p_spam_product *= self.p_word_spam.get(word, 1 / (self.spam_len + 1))
            p_ham_product *= self.p_word_ham.get(word, 1 / (self.ham_len + 1))
   
           


        if p_ham_product > p_spam_product:
           # print("Found Ham message! ", message)
            return 'ham'
        elif p_spam_product > p_ham_product:
            #print("Found SPAMM message! ", message)
            return 'spam'
        else:
           # print("needs human classification ", message)
            return 'needs human classification'

        # Calculate posterior probabilities
       # p_spam_given_message = self.p_spam * likelihood_spam
        #p_ham_given_message = self.p_ham * likelihood_ham

        # if p_ham_given_message > p_spam_given_message:
        #     return 'ham'
        # elif p_spam_given_message > p_ham_given_message:
        #     return 'spam'
        # else:
        #     return 'needs human classification'


    def classify_test(self):
        '''
        Calculate the accuracy of the algorithm on the test set and returns 
        the accuracy as a percentage.
        '''

        self.train()

        correct_classify = 0
        total_messages = len(self.test_set)

        for index, row in self.test_set.iterrows():
            message = row['v2']
            actual_label = row['v1']

            predict = self.sms_classify(message)

            if predict == actual_label:
                correct_classify += 1

        accuracy = (correct_classify / total_messages) * 100
        return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('--test_dataset', type=str, default = "test.csv", help='path to test dataset')
    args = parser.parse_args()
    classifier = NaiveBayesFilter(args.test_dataset)
    acc = classifier.classify_test()
    print("Accuracy: ", acc)
