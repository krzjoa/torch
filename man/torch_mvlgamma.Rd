% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gen-namespace-docs.R,
%   R/gen-namespace-examples.R, R/gen-namespace.R
\name{torch_mvlgamma}
\alias{torch_mvlgamma}
\title{Mvlgamma}
\usage{
torch_mvlgamma(self, p)
}
\arguments{
\item{self}{(Tensor) the tensor to compute the multivariate log-gamma function}

\item{p}{(int) the number of dimensions}
}
\description{
Mvlgamma
}
\section{mvlgamma(input, p) -> Tensor }{


Computes the \verb{multivariate log-gamma function <https://en.wikipedia.org/wiki/Multivariate_gamma_function>}_) with dimension
\eqn{p} element-wise, given by

\deqn{
    \log(\Gamma_{p}(a)) = C + \displaystyle \sum_{i=1}^{p} \log\left(\Gamma\left(a - \frac{i - 1}{2}\right)\right)
}
where \eqn{C = \log(\pi) \times \frac{p (p - 1)}{4}} and \eqn{\Gamma(\cdot)} is the Gamma function.

All elements must be greater than \eqn{\frac{p - 1}{2}}, otherwise an error would be thrown.
}

\examples{
if (torch_is_installed()) {

a = torch_empty(c(2, 3))$uniform_(1, 2)
a
torch_mvlgamma(a, 2)
}
}
