#' Bayesian biomarker testing using drug response summary statistic estimates with Stan
#'
#' @export
#' @param data Tibble or data frame containing mean and standard deviation
#' estimates for the summary statistics in columns `stat.mean` and `stat.sd`,
#' and biomarker status (0 or 1) in column `status`.
#' @param min_samples Minimum number of samples needed.
#' @param min_pos Minimum number of positive (biomarker status 1) samples needed.
#' @param alpha Tuning parameter for the influence of the estimated standard deviations; the effective standard deviation becomes `stat.sd=stat_sd^alpha`. `alpha` should be between 0 and 1.
#' @param scaling_flag Whether to scale (standardize) the summary statistics (default: `TRUE`)
#' @param iter Number of iterations.
#' @param thin A positive integer specifying the period for saving samples.
#' @param pars Parameters of interest (see `rstan::sampling`).
#' @param include Whether to include or exclude parameters of interest.
#' @param refresh How often to refresh the progress bar. The default is to not display the progress bar (`refresh=-1`).
#' @param ... Arguments passed to `rstan::sampling`.
#' @return A vector of posterior samples for the effect size parameter of the biomarker status.
#' @examples
#' # Run the sampler to get posterior samples of the association between
#' # the summary statistics and the biomarker, using the Afatinib example
#' # dataset
#' beta_samples = bayesian_biomarker_test_stan(BRCA_Afatinib_Response)
#'
#' # Plot the posterior distribution of the effect size parameters
#' hist(beta_samples)
#'
bayesian_biomarker_test_stan <- function(data, min_samples=10, min_pos=1, alpha=1,
                                         scaling_flag=TRUE,
                                         iter = 4000,
                                         thin=5, pars=c('shifted_mean'),
                                         include=FALSE, refresh=-1, ...) {
  n_samples = dim(data)[1]
  n_pos = sum(data$status == 1)

  beta_samples = NULL

  if(n_samples <= min_samples) {
    warning(paste0('Number of samples is smaller than ', min_samples, '. Consider changing min_samples argument.'))
  } else if(n_pos <= min_pos) {
    warning(paste0('Number of positive samples is smaller than ', min_pos, '. Consider changing min_pos argument.'))
  } else{
    if(scaling_flag) {
      scaled_stat = scale(data$stat.mean)
      scaled_std = (data$stat.std / attr(scaled_stat, 'scaled:scale'))^alpha
    } else {
      scaled_stat = data$stat.mean
      scaled_std = data$stat.std^alpha
    }

    # Bayesian meta-analysis
    cl_dat = list(N = n_samples,
                   suff_stat = c(scaled_stat),
                   cl_std = scaled_std,
                   bm = data$status)


    fit = rstan::sampling(stanmodels$bayesian_biomarker_test_tissue, data = cl_dat,
                          iter = iter, thin=thin, pars=pars,
                          include=include, refresh=refresh, ...)

    beta_samples = rstan::extract(fit, 'beta_biomarker')[[1]]
  }

  return(beta_samples)
}
