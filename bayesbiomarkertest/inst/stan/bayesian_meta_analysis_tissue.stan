// Bayesian meta-analysis with standard Gaussian distributions

data {
  int<lower=0> N; // number of samples
  real suff_stat[N]; // sufficient statistic
  vector[N] cl_std; // standard deviation of statistic
  vector[N] bm; // biomarker assignments
}

parameters {
  real global_mean; // global mean
  real<lower=0.001> beta_std; // global standard deviation
  real<lower=0.001> ss_std; // standard deviation of sufficient statistic
  real beta_biomarker_raw; // biomarker parameter
}

transformed parameters {
  vector[N] shifted_mean; // global mean plus covariates
  real beta_biomarker;

  beta_biomarker = beta_biomarker_raw*beta_std;
  shifted_mean = global_mean + beta_biomarker * bm;

}

model {
  beta_std ~ exponential(10);
  beta_biomarker_raw ~ normal(0,1);

  suff_stat ~ normal(shifted_mean, sqrt(square(cl_std) + square(ss_std)));
}
