test_that("Test argument_to_torch_type", {
  tensor <- torch_tensor(1:10)
  
  expect_failure(
    argument_to_torch_type(list(), "Tensor", "__and__")
  )
})