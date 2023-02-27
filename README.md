# Fake_News_Detection

Implementation of parse_data_line 
For its implementation I had to see where is the statement and the label in the data line for that I had displayed 
the dataframe head which was loaded with the help of pandas. 
By doing so it turned out that the label is at position 1 and statement is at position 2 for every data line. 
I had then converted the label using the convert_label function provided and then returned the tuple containing 
just the converted label and statement.

Implementation of pre_process function 
Initially to convert the given text to list of token. I had firstly used regular expression to separate punctuation 
from the end and the beginning of the string and then separated or splitted the text into tokens by the space (using regular expression) 

Implementation of to_feature_vector 
It is a simple function which just take a list of token and return a dictionary whose keys are the words in the list of 
token and value is its occurance in the list. 
I had iterated every word in the list using a for loop. There is a get method of dictionary where we pass two 
attributes. It will search for first attribute in the key of the dictionary to return its value or will return the 2nd
attribute instead. I fetched the count of the particular word in both local_dict(which has to be returned) and the 
global_feature_dict(which needs to be maintained as per the question) and updated or incremented its count by 
1. And then returned the local_dict. 

 Implementation of cross_validate function 
As per the theory of cross validation given the dataset and number of folds we need to divide the dataset into 
given number of folds we will then apply for loop on number of folds and use one fold at every iteration for 
testing and rest of the dataset for training. 
This is how I had splitted the dataset at every iteration into two parts train_d and test_d. I had trained the 
classifier with train_d using the train_classifier function which was made available. Both train_d and test_d are list 
of tuples for every tuple index 0 is statement and index 1 is label. 
I took all the elements at index 0 of test_d using a lambda function and then calculated the predictions for it using 
the given predict_labels function and store the results into a variable results which was defined before. 
I had stored the actual labels of test_d into a variable actual which was initialized before. Now that I had the 
actual and predicted labels of test_d. I had calculated the precision, recall, f score, accuracy using predefined 
functions. 
I had stored all these four values into 4 lists. After the for loop I had calculated the mean of all these four lists 
using numpy and stored into a dictionary and then returned that dictionary cv_results as stated. 

 Error Analysis 
I took the first fold of 10 folds of the train_data. Test and train data is stored into variables namely 
test_error_data and train_error_data respectively. I had trained the classifier with train_error_data using the 
train_classifier function, predicted labels for test_error_data with predict_labels function and stored actual labels 
of test_error_data in variables namely results and actual. 
I had printed the false positives and false negatives for fake label using actual and results with the help of 
confusion_matrix method of sklearn.metrics which was 182 and 163 respectively. 
At the end I ran the final code snippet of the notebook by setting functions_complete variable to True. I had also added 
the accuracy score to get the final accuracy which was 0.5627

 Optimizing pre-processing 
For optimizing pre processing I had performed stopword removal and stemming or lemmatising. I had done 
stopword removal using stopwords of English in nltk.corpus and stemming through PorterStemmer of nltk.stem. 
After removing stopwords and stemming the average accuracy of cross validation decreased to 0.5602. But as it 
was necessary so I kept it. 
Then I tried using bigrams instead of unigrams. For this I had used glue_tokens function(ref lab) and bigrams from 
nltk.utils. And instead of dictionary I had used Counter from collections. Other than words or glued words I had 
also returned the number of words in a statement with length label in the Counter. 
All these is done in bigram_features function. Then instead of to_feature_vector I had used bigram_features 
function. 
Applying the bigram model or by using bigram_features function the average accuracy of cross validation 
increased to 0.5779 
As it was had a positive impact I had decided to use trigram features. But with that the average accuracy of cross 
validation decreased to 0.5714. And as I keep on increasing the value of n in n-gram, the average accuracy of cross 
validation keeps on decreasing. 
Therefore, I, decided to select the bigram model.

Using other meta data in the file 
I had changed the load_data function and stored all the other columns of the data line into another Counter 
namely additional_features and added that Counter as well in the tuple which contain text and label and which is 
then appended to raw_data. 
Changes in split_and_preprocess_data function 
Initially for loop was taking two variable tuple from raw_data. Now it had 3 variables namely text, label and 
additional_features. I got another Counter of bigram model using bigram_features along with pre_process 
function. 
I then merged the two Counters and store in dict_final. Then the dict_final along with label is appended as a tuple 
in either train_data or test_data 

**Conclusion
After adding additional features to optimize the classifier performance. I got a tremendous increase in the average 
accuracy of cross validation and it became 0.7177. Then I finally set the function_complete variable to True and displayed 
precision, recall, f score and accuracy of the model. 
The final accuracy was 0.7086 and I wrapped up on that
