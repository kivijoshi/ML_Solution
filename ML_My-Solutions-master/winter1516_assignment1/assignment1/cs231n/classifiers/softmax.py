import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  
      
          

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  loss = 0.0
  
  for i in xrange(num_train):
      scores = X[i].dot(W)
      MaxedScore = scores
      MaxedScore -= np.max(MaxedScore)
      p = np.exp(MaxedScore) / np.sum(np.exp(MaxedScore))
      loss = loss + -np.log(p[y[i]])
      
      temp_p = np.reshape(p,(1,10))
      temp_xi = np.reshape(X[i],(3073,1))
      
      grad_other = temp_xi*temp_p
      grad_other[:,y[i]] = grad_other[:,y[i]] - X[i]
      dW = dW + grad_other
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW = dW/num_train;
  dW += reg * W
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0
  grad = np.zeros_like(W)
  num_train,dim = X.shape
  scores = (W.T).dot(X.T) # [K, N]
  # Shift scores so that the highest value is 0
  scores -= np.max(scores)
  scores_exp = np.exp(scores)
  correct_scores_exp = scores_exp[y,xrange(num_train)] # [N, ]
  scores_exp_sum = np.sum(scores_exp, axis=0) # [N, ]
  loss = -np.sum(np.log(correct_scores_exp / scores_exp_sum))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  scores_exp_normalized = scores_exp / scores_exp_sum
  # deal with the correct class
  scores_exp_normalized[y, xrange(num_train)] -= 1 # [K, N]
  grad = scores_exp_normalized.dot(X)
  grad /= num_train
  grad = grad.T
  grad += reg * W
  return loss, grad

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################