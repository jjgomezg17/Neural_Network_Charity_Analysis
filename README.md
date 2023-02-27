# Neural_Network_Charity_Analysis

## Overview of the analysis:

### The purpose of this analysis is to create a binary classifier using machine learning and neural networks to predict whether applicants will be successful if funded by Alphabet Soup, using data from over 34,000 organizations that have received funding from Alphabet Soup. The analysis involves preprocessing the data, compiling, training, and evaluating the model, optimizing the model, and submitting a written report on the neural network model.

## Results

### Data Preprocessing

#### What variable are considered the target for your model?

##### The variable "IS_SUCCESSFUL" which indicates whether the funding provided by Alphabet Soup was used effectively or not, is considered the target variable for the model. The objective of the model is to predict whether an organization will be successful or not in using the funding provided by Alphabet Soup based on other features available in the dataset.

#### What variable(s) are considered to be the features for your model?

##### In this case, the features considered for the model would be all the variables/columns present in the dataset except for the "EIN" and "NAME" identification columns, "ASK_AMT" which is the funding amount requested, and "IS_SUCCESSFUL" which is the target variable. So the features for the model would be: "APPLICATION_TYPE", "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION", "STATUS", "INCOME_AMT", and "SPECIAL_CONSIDERATIONS". These variables provide information about the organizations that have received funding and can help predict whether a new applicant is likely to be successful or not in using the funding provided by Alphabet Soup.

#### What variable(s) are neither targets nor features, and should be removed from the input data?

##### The variables "EIN" and "NAME" are identification columns and do not provide any meaningful information that can help predict the success or failure of an organization in using the funding provided by Alphabet Soup. Therefore, these columns can be removed from the input data and are not considered as either targets or features for the model.

### Compiling, Training, and Evaluating the Model

#### How many neurons, layers, and activation functions did you select for your neural network model, and why?

##### The neural network model used consists of three hidden layers with 100, 80, and 50 neurons respectively, followed by an output layer with a single neuron. The activation function used in each hidden layer is ReLU (Rectified Linear Unit) and in the output layer is the sigmoid function.
##### The number of neurons and layers in a neural network can be chosen based on the complexity of the problem, the size of the dataset, and the available computational resources. In this case, the model has three hidden layers, which can help capture more complex relationships among the features and improve the accuracy of the predictions.
##### The ReLU activation function is commonly used in deep learning models and is known for its ability to handle non-linearity in the data. The sigmoid activation function is suitable for binary classification problems as it maps the output of the neural network to a value between 0 and 1, which can be interpreted as the probability of the input belonging to the positive class.
##### Overall, the selected configuration of neurons, layers, and activation functions for the model is a common choice for solving binary classification problems with neural networks, and has shown to be effective in achieving high accuracy.

#### Were you able to achieve the target model performance?

##### The target model performance was not achieved. The model was only able to achieve an accuracy of 65% even after trying different neural network configurations and re-categorizing different variables.
##### Achieving the desired model performance can be challenging, and it often requires careful analysis of the data and extensive experimentation with different model architectures and hyperparameters. It is important to keep in mind that not all problems can be solved with a high level of accuracy, and sometimes a more realistic approach is to optimize the model as much as possible while balancing the resources and time available.
##### However, there may still be opportunities to improve the model performance by further exploring the data and experimenting with different modeling techniques. It may also be useful to seek guidance from domain experts or consult the relevant literature to gain additional insights into the problem and potential solutions.

#### What steps did you take to try and increase model performance?

##### To increase model performance, I recategorized some of the variables by adding values to categories and used the same logic of the original variable on the cases of the variables INCOME_AMT, SPECIAL_CONSIDERATIONS, and ORGANIZATION. I also stopped using the variable ASK_AMT.
##### I then tried several neural network models with different configurations:
##### The first model had 4 dense layers with 100, 80, 50, and 1 neurons, respectively, and used the ReLU activation function in the first three layers and the sigmoid activation function in the last layer.
##### The second model used batch normalization layers after each dense layer and the ReLU activation function.
##### The third model used dropout regularization with a rate of 0.2 after each dense layer and the ReLU activation function.
##### The fourth model used a function that created a model with given hyperparameters, including the learning rate, batch size, number of hidden layers, and number of hidden units. The model had a dynamic number of dense layers with the same number of neurons and used the ReLU activation function. The model also used the sigmoid activation function in the output layer.
##### However, despite trying several models and configurations, I was only able to achieve a model performance of 65%.

## Summary

### The purpose of this analysis was to create a binary classifier using machine learning and neural networks to predict whether applicants will be successful if funded by Alphabet Soup. The model was trained on data from over 34,000 organizations that have received funding from Alphabet Soup.
### The data preprocessing phase involved identifying the target variable ("IS_SUCCESSFUL") and features, which were all the columns except for "EIN", "NAME", "ASK_AMT", and "IS_SUCCESSFUL". The input data was then split into training and testing sets, and scaled using the StandardScaler function.
### The neural network model used consisted of three hidden layers with 100, 80, and 50 neurons respectively, followed by an output layer with a single neuron. The activation function used in each hidden layer was ReLU, and in the output layer was the sigmoid function. However, the model was only able to achieve an accuracy of 65%.
### To increase model performance, the variables were recategorized, and several neural network models were tried with different configurations, such as using batch normalization layers, dropout regularization, and dynamic number of dense layers. Despite these efforts, the desired model performance was not achieved.

### Future Analysis

### To improve the model performance, further data exploration and feature engineering may be necessary. One possible approach is to perform a feature importance analysis to identify the most relevant variables for predicting the success of an organization. Additionally, different modeling techniques, such as ensemble learning or other types of neural networks, could be explored.
### Moreover, it may be useful to consult domain experts or relevant literature to gain additional insights into the problem and potential solutions. Finally, it could be beneficial to increase the size of the dataset or collect additional data to improve the accuracy of the model.
