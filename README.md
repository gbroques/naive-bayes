# Naive Bayes from Scratch in Python

[![Build Status](https://travis-ci.org/gbroques/naive-bayes.svg?branch=master)](https://travis-ci.org/gbroques/naive-bayes)

A custom implementation of a Naive Bayes Classifier written from scratch in Python 3.

![Bayes Theorem](bayes-theorem.png)
Source: http://www.saedsayad.com/naive_bayesian.htm

> In machine learning, naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

From [Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier).

## Dataset

#### Loan Defaulters

| Home Owner | Marital Status | Annual Income | Defaulted Borrower |
| ---------- | -------------- | ------------- | ------------------ |
| Yes        | Single         | $125000       | No                 |
| No         | Married        | $100000       | No                 |
| No         | Single         | $70000        | No                 |
| Yes        | Married        | $120000       | No                 |
| No         | Divorced       | $95000        | Yes                |
| No         | Married        | $60000        | No                 |
| Yes        | Divorced       | $220000       | No                 |
| No         | Single         | $85000        | Yes                |
| No         | Married        | $75000        | No                 |
| No         | Single         | $90000        | Yes                |

**Source:** *Introduction to Data Mining* (1st Edition) by Pang-Ning Tan

Figure 5.9, Page 230

## How to Run
Please run with Python 3 or greater.

`python main`