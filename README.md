# Identify Potential Charity Donors with Supervised Learning
Our client (CharityML) had observed that every donation they received in the past came from someone that was making more than $50,000 annually.  These individuals are classified as high potential donors.  Our goal here is to create supervised machine learning models to help predict whether an individual makes above this threshold based on attributes such as: *age* , *working class*, *level of education*, *marital status*, *occupation*, *relationship*, *race*, *sex*, *monetary capital*, *average hours worked per week*, and *native country*.

## Results
[finding_donors.ipynb](https://github.com/sclkan/Identify-Potential-Charity-Donors/blob/master/finding_donors.ipynb)

## Models and Methods
- Support Vector Machines (SVM)
- Gradient Boosting Model (GBM)
- K-Nearest Neighbors (KNN)
- Randomized Search with Cross Validation

## Summary
We have chosen SVM, GBM, KNN to be our initial models as this dataset requires approaches with great flexibility due to its high number of features.  After training and testing all three models, we could see that gradient boosting is the best fit as it proved to be superior in accuracy, F-score and prediction time.  We then fine-tune this model by using a randomized search with cross-validation to find the most suitable hyperparameters.  

Our optimized GBM ends up having an accuracy of 87% and an F-score of 75% in predicting whether an individual makes more than $50,000 annually (our target for donations).

## Source
[US Census Bureau data provided by UC Irvine](https://archive.ics.uci.edu/ml/datasets/Census+Income)

## Python Libraries
Scikit-learn, Pandas, NumPy, Matplotlib
