# Review of Deep Learning

## Machine Learning Problems

|                           Regression                            |                     Classification                      |
| :-------------------------------------------------------------: | :-----------------------------------------------------: |
|   Continuous variable (e.g. salaries, product price, ...etc)    |         Discrete variable (e.g. product class)          |
| Function Approximation (aka: curve fitting, regression, ...etc) |       Decision boundary (discriminative function)       |
|      Usually the objective is to minimise the square error      | Maximising the probability of an output given the input |
|                                                                 |                 Non-linear classifiers                  |

---

## Supervised learning (Annotation)

Solves both the regression and the classification problems.

---

## Learning algorithm ingredients

- [Model](#model): Hypothesis assumed about the data

  - Supervised, unsupervised, reinforcement
  - Decision Trees, neural networks, SVM, ...etc.

- [Objective function](#objectives)

  - Mean squared error, classification error, NLL, ...etc.
  - Evaluation criteria: Intrinsic, extrinsic.

- [Optimisation technique](#optimisation-algorithm)

  - SGD, SVD, ...etc.

- [Data](#data): A sample.

---

## Objectives

### Mean Squared Error (MSE)

Error minimisation through searching through the parameter space to find the optimal parameters $\theta^*$.

---

## Model

The hypothesis that governs the in-out relationship

|                     Parametric                      |                             Non-parametric                             |
| :-------------------------------------------------: | :--------------------------------------------------------------------: |
| The entire model is known except for the parameters | We don't have idea about parameters, and works with a different search |
|                       Weights                       |                                                                        |

### Hyper-parameters

pre-defined parameters that the model don't alter, like learning rate.

Choosing the hyper-parameters is done using grid search, where a **brute force** search is done (searching inside the entire grid), or through **random search**.

### Cross Validation

A method for hyper-parameters choosing.

For each hyper-parameter setting:

- Train on $90$% of the training data.
- Test on held out $10$% of the training data.
- Repeat until all $10$ folds are consumed.
- Calculate average error.
- Select the setting with the least average error.

---

## Data

Data means a sample data, cause it is infeasible to collect all data.

<!-- | Train | Validation/Development | Test | -->
<!-- | :---: | :--------------------: | :--: | -->

<table text-align="center">
	<thead>
		<tr>
			<th colspan="2">Train</th>
			<th>Test</th>
		</tr>
		<tr>
			<th>K-Fold CV</th>
			<th>Validation/Development</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td></td>
			<td></td>
			<td>
				Never before seen by the model<br />
				for generalisation purposes
			</td>
		</tr>
	</tbody>
</table>

---

## Optimisation Algorithm

Gradient Descent (Perceptron Algorithm)

---

## Evaluation Criteria

|            Intrinsic             |              Extrinsic               |
| :------------------------------: | :----------------------------------: |
| Verification upon self-judgement | Verification upon external judgement |

---

## Training Cycle

1. Design a model.
2. Use CV to select hyper-parameters.
3. Train on train data.
4. Evaluation on hold-out data.
5. Test the model on test data.

---

## Problems in ML

| Underfitting |  Overfitting  |
| :----------: | :-----------: |
|  high bias   | high variance |

### Errors

|   Problem    |      Training error       |       Testing error       |
| :----------: | :-----------------------: | :-----------------------: |
| Overfitting  | below desired performance | above desired performance |
| Underfitting | above desired performance | above desired performance |

### Error Analysis

- Testing the impact of each single feature.
- Ablative analysis, adding/removing a single feature one at a time.

---

## Solutions to ML Problems

|     High variance (Overfitting)     |     High bias (Underfitting)     |
| :---------------------------------: | :------------------------------: |
|         Larger training set         |                                  |
| Less number of learnable parameters |        More complex model        |
|       Less number of features       |          More features           |
|           Regularisation            | Different optimisation algorithm |
|           Early Stopping            |                                  |

---

## Neural Network

### Single Neuron

Linear regression can be modelled by a single neuron with identity for activation function. Regression can have a closed form solution using the normal equation.

> The `MSE` objective is derived from `MLE`; where &sigma; is assumed to be $1$. It is good for regression, but bad for multi-class classification.

Single neuron can also be used for classification, but using activation function (e.g. step, sigmoid). This type of problems have no closed form solutions, so it is approached by Support Vector Machines `SVM`s. In neural networks, the solution is numerical.

### Numerical Solutions

|    1st Order     |         2nd Order         |
| :--------------: | :-----------------------: |
|  1st derivative  |      2nd derivative       |
| Gradient Descent | Newton's method (Hessian) |

---

## Cross Entropy (MaxEnt)

Softmax:

$$
h_{\theta}(X^{(i)})=
\frac{1}{\sum_{j=1}^ke^{\theta_k^Tz^{(i)}}}\begin{bmatrix}e^{\theta_1^Tz^{(i)}} \\ e^{\theta_2^Tz^{(i)}} \\ \vdots \\ e^{\theta_k^Tz^{(i)}}\end{bmatrix}
$$

---

## General GD algorithm

repeat for $K$ epochs or until convergence:

- for each record
  - feed forward
  - compute error
  - compute gradient
  - feed backward
  - update parameters

---

## Hyper-parameters tuning

### Learning rate &alpha;

Adaptive learning rate (AdaGrad, RMSProp, Adam)

---

### Momentum

A term added to increase 'momentum' in the direction that minimises the error; in doing so makes convergence faster.

---

### Network architecture

Overfitting and underfitting impact the network architecture, or rather the solutions for them, as the need for less parameters/features arise, the architecture should be simpler. And vice versa for the underfitting.

---

### Solving Overfitting

Using regularisation instead of unconstrained parameters.

|                           L1 Regularisation `lasso`                           |                           L2 Regularisation `ridge`                           |
| :---------------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
| adding norm 1 of all parameters in the system to the objective/error function | adding norm 2 of all parameters in the system to the objective/error function |

A less expensive alternative is early stopping, by monitoring the dev/validation error, if it starts to increase whereas training error decreases, then the algorithm started overfitting.

> Also could use both regularisation w/ early stopping

#### Dropout

Randomly selecting weights/neurons **not** to learn during certain epochs.

---

### Batch size

Using the entire dataset inside an epoch is vanilla gradient descent. The data could be broken into chunks to have batch, mini-batch & stochastic gradient descent.

#### Batch normalisation

For each batch/mini-batch, calculate the mean and variance and normalise the batch. $z=\frac{x-\mu}{\sigma}$.

During training, the means & variances are kept to update the global mean & variance.

---

### Activation function

Most famous are sigmoid & $tanh$ functions.

The $tanh$ function is useful around sparse features (having lots of zeros); as at $0$, the $tanh$ has value of $0$, so no accumulation happens there, unlike sigmoid which would have a value of $0.5$ at $0$.

Most used activation function is `ReLU` *Re*ctified *L*inear *U*nit.

---

## Deep Neural Networks `DNN`

Multi-layer Perceptron `MLP`

---

## Back-propagation

---
