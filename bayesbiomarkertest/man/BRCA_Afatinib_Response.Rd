% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/data.R
\docType{data}
\name{BRCA_Afatinib_Response}
\alias{BRCA_Afatinib_Response}
\title{Drug response dataset}
\format{A tibble with 47 rows (corresponding to cell lines) and 6 columns:
\describe{
  \item{Cell.Line}{Name of the cell line}
  \item{COSMIC.ID}{Unique ID for the cell line}
  \item{status}{Biomarker status. In this case the biomarker is copy number variation for CDK12, and 1=Copy Number gain; 0=No copy number gain.}
  \item{stat.mean}{Estimated mean value of the summary drug response statistic (scaled log IC50).}
  \item{stat.std}{Estimated standard error value of the summary drug response statistic (scaled log IC50).}
  \item{tissue.label}{Tissue type that the cell lines came from (in this case, all tissues are from breast cancers, BRCA)}
}}
\source{
Wang et at. 2020
}
\usage{
BRCA_Afatinib_Response
}
\description{
A dataset with 47 breast cancer (BRCA) cell lines, their summary drug response (scaled log IC50),
and standard error. This dataset is derived from an experiment where the cell lines were exposed to the
anti-cancer drug Afatinib.
}
\details{
#'
}
\keyword{datasets}
