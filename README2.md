# Challenge21_deep-learning-challenge

## Overview

This project aim to help the nonprofit foundation Alphabet Soup to build a tool that can help it select the applicants for funding with the best chance of success in their ventures using the CSV containing more than 34,000 organisations that have received funding from Alphabet Soup over the years. 

## Objective

The primary goal of this project is:
- To find out whether we can predict whether applicants will be successful if funded by Alphabet Soup.


## Dataset

The CSV file weblink `https://static.bc-edx.com/data/dla-1-2/m21/lms/starter/charity_data.csv`is provided and contain the following fields:

- `EIN` and `NAME` —Identification columns
- `APPLICATION_TYPE` —Alphabet Soup application type
- `AFFILIATION` —Affiliated sector of industry
- `CLASSIFICATION` —Government organisation classification
- `USE_CASE` —Use case for funding
- `ORGANIZATION` —Organisation type
- `STATUS` —Active status
- `INCOME_AMT` —Income classification
- `SPECIAL_CONSIDERATIONS` —Special considerations for application
- `ASK_AMT` —Funding amount requested
- `IS_SUCCESSFUL` —Was the money used effectively
    
The `IS_SUCCESSFUL` column will be used as `target` while the the rest of the columns as `features` minus the `EIN` and `NAME` columns in the machine learning and neural networks.

The target column has 34299 total rows with 18261 label as 1 ('successful') and 16038 label as 0 ('unsuccessful'). So, this show fairly balance with slight biased towards successful data.

## Tools and Libraries
- `Python`: Used for data preprocessing, initial analysis, and visualization.
- `Pandas`: Utilized for data manipulation and analysis.
- `Jupyter Notebook`: Employed as the development environment.
- `sklearn`: module used to do neural network deep prediction.
- `train_test_split()`: Function to split the original dataset to Train and Test datasets.
- `StandardScaler()` and `transform()`: To scale the training and testing features datasets.
- `tensorflow.keras.models.Sequential()`: Function to access the neural network deep prediction.
- `add()`: To add hidden layers.
- `summary()`: To see the summary of the model architecture built.
- `compile()`: Function to compile the model.
- `fit()`: Function to train the model.
- `evaluate()`: To evaluate the model using the test data.
- `save()`: To save the model to HDFS file.


## Workflow
The following is the workflow employed in performing the neural network deep prediction:

1. Import the necessary dependencies: 
	- from `sklearn` import `train_test_split`,  `StandardScaler`
	- `pandas`
	- `tensorflow`
2. Preprocessing the data.
	- Read in the `charity_data.csv` and identify `IS_SUCCESSFUL` as the target variable while the rest of the dataset are used as `features` minus the `EIN` and `NAME` columns.
	- Determine the number of unique values for each columns.
		- Used cutoff_value of `200` for `APPLICATION_TYPE` column to reduce to 9 bins.
		- Used cutoff_value of `850` for `CLASSIFICATION` column to reduce to 6 bins.
	- Use `pd.get_dummies()` to encode categorical codes for `INCOME_AMT`, `APPLICATION_TYPE`, `CLASSIFICATION`, `AFFILIATION`, `USE_CASE`, `ORGANIZATION` and `SPECIAL_CONSIDERATIONS`.
	- Use `train_test_split` function to split the data into training and testing datasets.
	- Use `StandardScaler()` and `transform()` to scale the training and testing features datasets.
3. Compile, Train, and Evaluate the Model.  Check `AlphabetSoupCharity.ipynb` for the python codes. 
	- Use `tensorflow.keras.models.Sequential()` to create the initial neural network model with 2 hiddens layers and one output layer.
		- `Inital Model` - Structure: 2 hidden layers (80 neurons, 30 neurons) + Output layer (1 neuron)
	- `compile()` the model with `adam` as `optimizer`
	- `fit()` to train the model with epochs = 100.
	- `evaluate()` the model usign testt data and note the `loss` and `accuracy` numbers.
	- `save()` to save the model to HDFS file.
4. Optimise the Model.  Check `AlphabetSoupCharity_Optimisation.ipynb` for the python codes. Repeat the Step no.3 but with slightly different parameters as listed below:
	- `Optimsation Model No.1` - Structure: 2 hidden layers (160 neurons, 60 neurons) + Output layer (1 neuron)
	- `Optimsation Model No.2` - Structure: 4 hidden layers (160 neurons, 100 neurons, 80 neurons, 60 neurons) + Output layer (1 neuron)
	- `Optimsation Model No.3` - Structure: 2 hidden layers (60 neurons, 40 neurons) + Output layer (1 neuron)

5. Finally, the assessment of the results of the neural network deep learning prediction is carried out by comparing the `loss` and `accuracy` scores.


## Usage

1. **Setup Environment:**
   - Download Jupyter Notebook so that you can download the uploaded files and view within your local machine.

2. **Educational Purposes:**
   - Feel free to download the uploaded pages so that you can also explore the dataset and gain insights.

## Main Results and Findings.
The results of the testing of different neural network architectures with varying numbers of hidden layers and neurons in each layer to find the best-performing model for the data are given as below:

| Model Name | Structure                                | Accuracy | Loss    |
|------------|------------------------------------------|----------|---------|
| Initial    | 2 hidden layers (80 neurons, 30 neurons) + Output layer (1 neuron) | 72.67%   | 0.5655  |
| Model 1    | 2 hidden layers (160 neurons, 60 neurons) + Output layer (1 neuron) | 72.80%   | 0.5740  |
| Model 2    | 4 hidden layers (160 neurons, 100 neurons, 80 neurons, 60 neurons) + Output layer (1 neuron) | 72.49%   | 0.5946  |
| Model 3    | 2 hidden layers (60 neurons, 40 neurons) + Output layer (1 neuron)  | 72.71%   | 0.5653  |


### Analysis:

#### Accuracy Comparison:
- Model 1 has the highest accuracy at 72.80%, closely followed by Model 3 at 72.71%.
- The initial model and Model 2 have accuracies of 72.67% and 72.49%, respectively.

#### Loss Comparison:
- Model 3 has the lowest loss at 0.5653, followed by the initial model at 0.5655.
- Model 1 has a loss of 0.5740, and Model 2 has the highest loss at 0.5946.

### Observations:
- Model 1 has a slightly higher accuracy compared to the others, but it also has a higher loss than the initial model and Model 3.
- Model 3 has a lower loss than the initial model, indicating potentially better performance in terms of minimizing errors.
- The initial model performs quite similarly to Model 3 in terms of accuracy and has a slightly higher loss.

### Conclusion:
- Model 1 might have a marginal edge in accuracy, but considering the balance between accuracy and loss, Model 3 or the initial model could be competitive choices, especially if a lower loss is of significance.
- Further analysis, validation on a test set, and consideration of other factors beyond accuracy and loss (such as computational efficiency, robustness, interpretability) are crucial in determining the most suitable model for deployment.

### Further Testing:
Further testing could be done to see whether it is possible to reach the >75% accuracy with low loss such as follow:

1. Hyperparameter Tuning:
- Learning Rate Adjustment: Experiment with different learning rates to find the optimal one that allows the model to converge faster without overshooting.
- Activation Functions: Test different activation functions (eLU, tanh, etc.) to see which suits the data best.  This project has tried ReLU and sigmoid.
- Batch Size and Epochs: Reduce or increase the batch sizes or more epochs improve performance.

2. Model Architectures.
- Different Architectures: Experiment with different architectures like convolutional neural networks (CNNs), recurrent neural networks (RNNs), or their variants depending on the nature of the data.

## References

1. Inspired by lectures notes and ChatGPT.
