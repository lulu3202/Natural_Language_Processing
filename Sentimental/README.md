## Whatsapp Text Analyzer

Data input is whatsapp chat export between 2 people in csv format

Data Preprocessing:
	• Convert text file raw data to table
	
### Sentiment Analyzer
	• NLTK already has a Sentiment_Intensity_Analyzer function which is used in this notebook
	• No preprocessing is required here 
	• Polarity scores is an in-built function that provides negative, neutral, positive and compound value for every input
	• Def sentimalAnalysis function will help to import required libraries and add polarity scores values to every row via apply method (instead of using for loop) using lambda function. i.e apply and lambda together will replace for loop
	• Result will be a compilation of polarity scores for the provided text data input

### Topic Modelling
	• This is an Unsupervised algorithm 
	• For eg, 5 pages of pdf, will provide information on which specific topic has been dealt with
	• Unlike sentiment analyzer, preprocessing is required here (either using tfid/counter/hashing)
	•  We create tfid vector, and using fit_transform we create model and fit our input into it
	• Post vectorization, we use NMF algorithm for topic modelling via decomposition method
	• There are several combos here, I have tried, TFID vector + NMF model, TFID Vector + LDA model and Counter vector + LDA model 
		

