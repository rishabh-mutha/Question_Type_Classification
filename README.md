# Question_Type_Classification
An NLP based model to identify question type: Given a question, the aim is to identify the category it belongs to. The four categories to handle are : Who, What, When, Affirmation(yes/no).

Instructions to run the model:

1. Run python train_test_model.py
  this program will train on the given sample question data in LabelledData.txt file, report the training and testing accuracies on test.label.txt file and finally save the predicted labels in predictedlabel_file.csv

2. The models will be trained and saved in models directory
  so classify_question.py can be run directly without having to run train_test_model.py
  
3. To get predicted question type on any custom user input question, run python  classify_question.py
  it allows user to input text and output follows
  
4. some sample input test cases and their corresponding outputs:

Enter the question: how is the weather?
['unknown']
Enter the question: what is today's date?
['what']
Enter the question: is it raining?
['affirmation']
Enter the question: what time does the train leave?
['when']
Enter the question: who is the inventor of cricket?
['who']
