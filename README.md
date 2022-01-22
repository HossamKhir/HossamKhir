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
