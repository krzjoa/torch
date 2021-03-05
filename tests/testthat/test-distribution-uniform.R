#' Note: consider PyTorch - like test schema
#' See: https://github.com/pytorch/pytorch/blob/master/test/distributions/test_distributions.py
#' TODO: add more unit tests

test_that("Uniform distribution - basic tests", {
  
  low     <- torch_zeros(5, 5, requires_grad = TRUE)
  high    <- (torch_ones(5, 5) * 3)$requires_grad_()
  low_1d  <- torch_zeros(1, requires_grad = TRUE)
  high_1d <- (torch_ones(1) * 3)$requires_grad_()
  
})
