import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  scores_mat_grad = (500,10)
  scores_mat_grad = np.ones(scores_mat_grad)
  loss = 0.0
  for i in xrange(num_train):
    Num_Classes_Above = 0
    score_grad = np.zeros(10)
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        score_grad[j] = 1
        Num_Classes_Above += 1
    score_grad[y[i]] = (-Num_Classes_Above)
    scores_mat_grad[i] = score_grad
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  dW = ((scores_mat_grad.T).dot(X) / 500) + (reg * W).T
  dW = dW.T
  
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_train = X.shape[0]
  num_class = W.shape[1]
  
  scores = (W.T).dot(X.T)
  
  correctscore = scores[y,range(0,num_train)]
  nScore = (num_class,num_train)
  nScore = np.ones(nScore)
  nScore = nScore*correctscore
  scorenegation = scores - nScore
  scorenegation = scorenegation + 1
  margins = np.maximum(0,scorenegation)
  margins[y,range(0,num_train)] = 0
  
  scores_mat_grad = np.zeros(scores.shape)
  num_pos = np.sum(margins > 0, axis=0)
  scores_mat_grad[margins > 0] = 1
  scores_mat_grad[y, xrange(num_train)] = -1 * num_pos
  dW = (scores_mat_grad.dot(X) / num_train) + (reg * W).T
  dW = dW.T
  
  loss_i = np.sum(margins)
  loss = loss_i/X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)
  print(scorenegation)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
