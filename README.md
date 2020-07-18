# Identify Potential Charity Donors with Supervised Learning
Our client (CharityML) observed that every donation they had received in the past came from someone that was making more than $50,000 annually.  Since these individuals are considered high potential donors, our goal here is to create machine learning algorithms that can predict whether someone makes above this $50,000 threshold.  I will first preprocess the data by using techniques such as log transformation and one hot encoding.  Then, I will use a few algorithms to conduct testing and determine the best one to use for this dataset.  Aftrward, I will optimize the selected model, evaluate the results, and highlight features with the most predictive power.

The attributes for this dataset are: *age* , *working class*, *level of education*, *marital status*, *occupation*, *relationship*, *race*, *sex*, *monetary capital*, *average hours worked per week*, and *native country*.


## Results
[finding_donors.ipynb](https://github.com/sclkan/Identify-Potential-Charity-Donors/blob/master/finding_donors.ipynb)

## Algorithms and Techniques
- Support Vector Machines (SVM)
- Gradient Boosting Model (GBM)
- K-Nearest Neighbors (KNN)
- Randomized Search with Cross-Validation

## Summary
I begin by testing the dataset with SVM, GBM, and KNN  as our learners need to have great flexibility due to the large amount of features (103 columns) after one-hot encoding.  After comparing the performance of all three models, we can see that gradient boosting is the best approach here as it is superior in accuracy, F-score, and prediction time for this scenario.  I then fine-tune this model by using a randomized search with cross-validation to find the most suitable value for hyperparameters such as *learning rate*, *max depth*, *min splits*, etc.  

The optimized GBM model ends up having an accuracy of 87% and an F-score of 75% in predicting whether an individual makes more than $50,000 annually (our target for donations).  The most important features for making predictions are:

1. martial status - married with a civilian spouse
2. capital gain
3. number of education
4. capital loss
5. age


## Source
[US Census Bureau data provided by UC Irvine](https://archive.ics.uci.edu/ml/datasets/Census+Income)

## Python Libraries
Scikit-learn, Pandas, NumPy, Matplotlib
