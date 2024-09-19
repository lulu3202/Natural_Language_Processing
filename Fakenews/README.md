## Fake News with Counter Vector + Multinomial NB Algorithm

About Dataset
We have a matrix of shape (4244, 56922):
	• 4244 rows, meaning dataset contains 4244 documents (individual pieces of text)
	• The matrix tracks the presence and frequency of 56922 unique words across those documents
	• The goal is to predict whether the news is fake or real


How Counter Vector Works
In Natural Language Processing (NLP), when you use the CountVectorizer for preprocessing text, the goal is to convert text into numerical form so that machine learning algorithms can process it. 

1. What is CountVectorizer?
CountVectorizer is a tool that converts a collection of text documents into a matrix of token counts. Each column in this matrix represents a word (token) from the vocabulary, and each row represents a document from your dataset.
	• Rows: Represent individual documents.
	• Columns: Represent words (tokens) from the entire vocabulary.
	• Values: Represent how many times each word appears in the document.
2. Steps in CountVectorizer:
	• Tokenization: It breaks the documents into individual words (tokens).
	• Building Vocabulary: It creates a list of all unique words in your corpus (vocabulary).
	• Count Frequency: It counts how often each word appears in each document.
3. What does count_train.toarray() represent?
When you call count_train.toarray(), it prints the entire matrix, where:
	• Each row represents a document.
	• Each column represents a word from the vocabulary.
	• The value in each cell shows how many times the corresponding word (column) appears in that document (row).
Example:
Let's say we have two documents:
	1. "I love NLP"
	2. "I love learning NLP"
The CountVectorizer will create a matrix like this:
	I	love	NLP	learning
D1	1	1	1	0
D2	1	1	1	1
Here:
	• Document 1 ("I love NLP") becomes [1, 1, 1, 0] (1 occurrence of "I", "love", "NLP", and 0 of "learning").
	• Document 2 ("I love learning NLP") becomes [1, 1, 1, 1].
So, calling count_train.toarray() will output a 2D array with these values.


Code Snippet to test out our model 
Step 1: CountVectorizer(stop_words='english')
	• CountVectorizer is used to convert the text documents into a matrix of token counts. Each word (except for stop words) becomes a feature in the matrix.
	• stop_words='english' removes common English stop words like "and," "the," and "is" because they don't carry much useful information for most NLP tasks.

Step 2: count_train = count_vectorizer.fit_transform(X_train)
	• fit_transform(X_train) performs two operations:
		1. fit: It builds a vocabulary of unique words from the training dataset (X_train). These words become the columns (features) in the matrix.
		2. transform: It converts each document in X_train into a sparse matrix of word counts, where each row represents a document, and each column represents the count of a specific word from the vocabulary in that document.
For example, if X_train contains documents like:

X_train = [
    "I love NLP",
    "NLP is fun",
    "Deep learning is part of NLP"]
	• CountVectorizer will first build a vocabulary like ['deep', 'fun', 'learning', 'love', 'nlp', 'part'] after removing stop words.
	• Then, it will convert each document into a vector of word counts based on this vocabulary.
Step 4: count_train[[0]]
	• count_train[[0]] selects the vector for the first document in the training data (i.e., the first row of the matrix). In NLP terms, this is the vectorized representation of the first document.
	• This is a sparse matrix representing the word counts for just that one document.
For example, if the first document is "I love NLP", and your vocabulary is ['deep', 'fun', 'learning', 'love', 'nlp', 'part'], then count_train[[0]] might look like:


[0 0 0 1 1 0]
This means that:
	• The word 'love' appears once in the first document (hence 1 in the 'love' column).
	• The word 'nlp' appears once in the first document.
	• The other words ('deep', 'fun', 'learning', 'part') do not appear in the document.

Step 5: clf.predict(count_train[[0]])
	• clf is a trained classifier (such as a logistic regression, SVM, or another model).
	• predict(count_train[[0]]) passes the vector representation of the first document (count_train[[0]]) to the classifier to make a prediction.
	• The classifier makes its prediction based on the word counts/features of that specific document.

For example, if the task is document classification (e.g., classifying news as FAKE or REAL), the classifier will use the word counts in count_train[[0]] to predict the class label of the first document in the training set.

Summary of Key Concepts:

• Training: You train the classifier using the vectorized text data and their associated labels.
• Prediction: After training, you can use the classifier to predict labels for new, unseen text data.
• Evaluation: You can evaluate the model's performance by comparing predictions with actual labels in a test set.
• count_train[[0]]: The vectorized representation (word counts) of the first document in the training set.
• CountVectorizer: Converts raw text into a matrix of word counts.
• Classifier (clf): Makes predictions based on the document’s word count vector.

## Fake News with TF-IDF Vector + Multinomial NB Algorithm

TF-IDF (Term Frequency-Inverse Document Frequency) is a popular technique used in text mining and information retrieval to represent text documents in a numerical format. It combines two metrics to quantify the importance of each word in a document relative to a collection of documents (corpus).

Components of TF-IDF
	1. Term Frequency (TF)
		○ Definition: Measures how frequently a term (word) appears in a document.
		○ Formula: TF(t,d)=Number of times term t appears in document dTotal number of terms in document d\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}TF(t,d)=Total number of terms in document dNumber of times term t appears in document d​
		○ Purpose: This metric helps capture how important a term is within a specific document. Words that appear more frequently in a document are considered more relevant to that document.
	2. Inverse Document Frequency (IDF)
		○ Definition: Measures how important a term is across the entire corpus. It decreases the weight of terms that occur frequently across many documents, as these terms are less informative.
		○ Formula: IDF(t,D)=log⁡Total number of documents in corpus DNumber of documents containing term t\text{IDF}(t, D) = \log \frac{\text{Total number of documents in corpus } D}{\text{Number of documents containing term } t}IDF(t,D)=logNumber of documents containing term tTotal number of documents in corpus D​
		○ Purpose: This metric helps reduce the impact of terms that appear in many documents, highlighting terms that are unique to a smaller number of documents.
	3. TF-IDF Score
		○ Definition: Combines TF and IDF to give a score that reflects both the term's importance within a document and its significance across the corpus.
		○ Purpose: This score balances the term's frequency in a document with its rarity across all documents, providing a more nuanced measure of its importance.

Example Calculation
Let's calculate TF-IDF for a term in a simple corpus with 3 documents:
Corpus:
	1. "I love machine learning"
	2. "Machine learning is fascinating"
	3. "Deep learning is a part of machine learning"

Term to Analyze: "machine"

Step 1: Calculate TF
	• For Document 1: "I love machine learning"
		○ Total words: 4
		○ Occurrences of "machine": 1
		○ TF = 1 / 4 = 0.25
	• For Document 2: "Machine learning is fascinating"
		○ Total words: 4
		○ Occurrences of "machine": 1
		○ TF = 1 / 4 = 0.25
	• For Document 3: "Deep learning is a part of machine learning"
		○ Total words: 8
		○ Occurrences of "machine": 1
		○ TF = 1 / 8 = 0.125

Step 2: Calculate IDF
	• Total number of documents (N): 3
	• Number of documents containing "machine" (n): 3
IDF("machine")=log⁡33=log⁡1=0\text{IDF}(\text{"machine"}) = \log \frac{3}{3} = \log 1 = 0IDF("machine")=log33​=log1=0
Since "machine" appears in all documents, its IDF is 0, indicating it is not particularly informative across the corpus.
Step 3: Calculate TF-IDF
	• For Document 1:
		○ TF-IDF = 0.25 × 0 = 0
	• For Document 2:
		○ TF-IDF = 0.25 × 0 = 0
	• For Document 3:
		○ TF-IDF = 0.125 × 0 = 0

Interpretation of TF-IDF 
	• TF-IDF > 0: The word is important for the document and adds value in distinguishing it from others.
	• TF-IDF = 0: The word is either not present in the document or is too common across the corpus to be useful for distinguishing documents.
This interpretation helps in various NLP tasks by highlighting the words that carry the most meaning and relevance within a given document and across a collection of documents.

Summary
	• TF-IDF is a statistical measure to evaluate the relevance of a word in a document relative to the entire corpus.
	• TF measures the frequency of the term in a document.
	• IDF measures the term’s importance across all documents.
	• TF-IDF combines these to provide a balanced measure of term importance.
This approach helps in various NLP tasks, such as text classification, information retrieval, and clustering, by effectively representing the textual data for analysis.

## Fake News with Hashing Vector + Multinomial NB Algorithm
About Hashing Vectors

Hashing vectors is a technique used to convert data, like text, into a fixed-size numerical representation. Here’s a simple way to think about it:
	1. Purpose: The goal of hashing vectors is to transform complex data into a format that machine learning algorithms can easily work with. This is especially useful for handling large datasets or text data, where you need a consistent and manageable way to represent and process the information.
	2. How It Works:
		○ Input: You start with raw data, like a word or a document.
		○ Hash Function: A hash function takes this data and maps it to a fixed-size vector (a list of numbers). It does this using a mathematical formula.
		○ Output: The result is a vector with numbers that represent the original data in a compressed form.
	3. Why Hashing?:
		○ Efficiency: It helps to reduce the size of the data while preserving its key features, making it easier and faster to work with.
		○ Simplicity: It simplifies complex data (like words) into a numerical format that machine learning models can handle.
	4. Example:
		○ Imagine you have the word "apple." A hash function might convert "apple" into a vector like [0.12, 0.87, 0.45, ...]. This vector is a fixed-size numerical representation of the word "apple."

In Summary: Hashing vectors convert data into fixed-size numerical representations, making it easier for algorithms to process and analyze. It’s a way to manage and simplify complex data, especially in machine learning and data processing tasks.

Hashing vs Count vs TF - IDF Vectors
	• Hashing Vectors are great for efficiency and handling large datasets but can be less interpretable.
	• Count Vectors are simple and easy to understand, focusing on word frequency, but can be sparse and may not handle document uniqueness well.
	• TF-IDF Vectors offer a more nuanced representation by factoring in word importance across documents, making them ideal for tasks where distinguishing between more and less significant words is crucial.
Choosing between these methods depends on the specifics of your task, the size of your dataset, and your need for interpretability versus efficiency.

Examples of Each Method
	1. Hashing Vectors
		○ Example: Suppose you have a document with the text: "Machine learning is exciting."
		○ Hashing: A hash function might convert this text into a fixed-size vector, like [0.23, 0.89, 0.45, ...], where each number is a result of applying a hashing algorithm.
	2. Count Vectors
		○ Example: Using the same text: "Machine learning is exciting."
		○ Count Vector: A count vector might represent this text as a vector where each dimension corresponds to a unique word from a predefined vocabulary, and the value represents the number of times each word appears. For example, if the vocabulary is ["Machine", "learning", "is", "exciting"], the count vector could be [1, 1, 1, 1] (since each word appears once).
	3. TF-IDF Vectors
		○ Example: For the same text: "Machine learning is exciting."
		○ TF-IDF Vector: This method not only considers the frequency of each word in the document but also how unique each word is across the entire dataset. If "Machine" and "learning" are common in many documents but "exciting" is rare, the TF-IDF vector might give "exciting" a higher value because it’s more unique. The result could be something like [0.5, 0.3, 0.2, 0.7], reflecting the term frequency and the inverse document frequency.
		○ Cons: More complex, can still be sparse, requires a corpus to calculate IDF.





