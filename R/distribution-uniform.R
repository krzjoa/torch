#' @include distributions.R
#' @include distributions-utils.R
#' @include distributions-constraints.R
#' @include utils.R

Uniform <- R6::R6Class(
  "torch_Uniform",
  lock_objects = FALSE,
  inherit = Distribution,
  
  public = list(
    # TODO allow (loc,scale) parameterization to allow independent constraints.
    .arg_constraints = list(low   = constraint_dependent(is_descrete = FALSE, event_dim = 0), 
                            high  = constraint_dependent(is_descrete = FALSE, event_dim = 0)),
    has_rsample = TRUE,
    
    initialize = function(low, high, validate_args = NULL){
      broadcasted <- broadcast_all(list(low, high))
      self$low    <- broadcasted[[1]]
      self$high   <- broadcasted[[2]]
      
      if (inherits(low, 'numeric') & inherits(low, 'numeric'))
        batch_shape <- NULL
      else
        batch_shape <- self$low$size()
      
      super$initialize(batch_shape, validate_args = validate_args)
      
      if (self$.validate_args & !torch_lt(self$low, self$high)$all())
        value_error("Uniform is not defined when low>= high")
    },
    
    expand = function(batch_shape, .instance=None){
      new <- self$.get_checked_instance(self, .instance)
      new$low  <- self$low$expand(batch_shape)
      new$high <- self$high$expand(batch_shape)
      new$.__enclos_env__$super$initialize(batch_shape, validate_args = FALSE)
      new$.validate_args <- self$.validate_args
      new
    },
    
    #' TODO: implement somehow functionality of 
    #' constraints.dependent_property(is_discrete=False, event_dim=0)
    support = function(){
      constraint_interval(self$low, self$high)
    },
    
    rsample = function(sample_shape = NULL){
      shape <- self$.extended_shape(sample_shape)
      rand <- torch_rand(shape, dtype = self$low$dtype, device = self$low$device)
      self$low + rand * (self$high - self$low)
    },
    
    log_prob = function(value){
      if (self$.validate_args)
        self$.validate_sample(value)
      lb <- self$low$le(value)$type_as(self$low)
      ub <- self$low$gt(value)$type_as(self$low)
      torch_log(lb$mul(ub) - torch_log(self$high - self$low))
    },
    
    cdf = function(value){
      if (self$.validate_args)
        self$.validate_sample(value)
      result <- (value - self$low) / (self$high - self$low)
      result$clamp(min = 0, max = 1)
    },
    
    icdf = function(value){
     result <- value * (self$high - self$low) + self$low
     result
    },
    
    entropy = function(){
      torch_log(self$high - self$low)
    }
  ),
  
  active = list(
    
    mean = function(){
      (self$high + self$low) / 2
    },
    
    stddev = function(){
      (self$high - self$low) / 12**0.5
    },
  
    variance = function(){
      (self$high - self$low)$pow(2) / 12
    }
  )
)

#' Generates uniformly distributed random samples from the half-open interval
#' `[low, high)`.
#' @param low (numeric or torch_tensor): lower range (inclusive).
#' @param high (numeric or torch_tensor): upper range (exclusive).
#' @examples 
#' m <- distr_uniform(torch_tensor(0.0), torch_tensor(5.0))
#' m$sample()  # uniformly distributed in the range [0.0, 5.0)
#' @export
distr_uniform <- function(low, high, validate_args = NULL){
  Uniform$new(low, high, validate_args)
}
