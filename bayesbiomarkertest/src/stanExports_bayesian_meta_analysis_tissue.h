// Generated by rstantools.  Do not edit by hand.

/*
    bayesbiomarkertest is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    bayesbiomarkertest is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with bayesbiomarkertest.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef MODELS_HPP
#define MODELS_HPP
#define STAN__SERVICES__COMMAND_HPP
#include <rstan/rstaninc.hpp>
// Code generated by Stan version 2.19.1
#include <stan/model/model_header.hpp>
namespace model_bayesian_meta_analysis_tissue_namespace {
using std::istream;
using std::string;
using std::stringstream;
using std::vector;
using stan::io::dump;
using stan::math::lgamma;
using stan::model::prob_grad;
using namespace stan::math;
static int current_statement_begin__;
stan::io::program_reader prog_reader__() {
    stan::io::program_reader reader;
    reader.add_event(0, 0, "start", "model_bayesian_meta_analysis_tissue");
    reader.add_event(33, 31, "end", "model_bayesian_meta_analysis_tissue");
    return reader;
}
#include <stan_meta_header.hpp>
class model_bayesian_meta_analysis_tissue : public prob_grad {
private:
        int N;
        std::vector<double> suff_stat;
        vector_d cl_std;
        vector_d bm;
public:
    model_bayesian_meta_analysis_tissue(stan::io::var_context& context__,
        std::ostream* pstream__ = 0)
        : prob_grad(0) {
        ctor_body(context__, 0, pstream__);
    }
    model_bayesian_meta_analysis_tissue(stan::io::var_context& context__,
        unsigned int random_seed__,
        std::ostream* pstream__ = 0)
        : prob_grad(0) {
        ctor_body(context__, random_seed__, pstream__);
    }
    void ctor_body(stan::io::var_context& context__,
                   unsigned int random_seed__,
                   std::ostream* pstream__) {
        typedef double local_scalar_t__;
        boost::ecuyer1988 base_rng__ =
          stan::services::util::create_rng(random_seed__, 0);
        (void) base_rng__;  // suppress unused var warning
        current_statement_begin__ = -1;
        static const char* function__ = "model_bayesian_meta_analysis_tissue_namespace::model_bayesian_meta_analysis_tissue";
        (void) function__;  // dummy to suppress unused var warning
        size_t pos__;
        (void) pos__;  // dummy to suppress unused var warning
        std::vector<int> vals_i__;
        std::vector<double> vals_r__;
        local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // suppress unused var warning
        try {
            // initialize data block variables from context__
            current_statement_begin__ = 4;
            context__.validate_dims("data initialization", "N", "int", context__.to_vec());
            N = int(0);
            vals_i__ = context__.vals_i("N");
            pos__ = 0;
            N = vals_i__[pos__++];
            check_greater_or_equal(function__, "N", N, 0);
            current_statement_begin__ = 5;
            validate_non_negative_index("suff_stat", "N", N);
            context__.validate_dims("data initialization", "suff_stat", "double", context__.to_vec(N));
            suff_stat = std::vector<double>(N, double(0));
            vals_r__ = context__.vals_r("suff_stat");
            pos__ = 0;
            size_t suff_stat_k_0_max__ = N;
            for (size_t k_0__ = 0; k_0__ < suff_stat_k_0_max__; ++k_0__) {
                suff_stat[k_0__] = vals_r__[pos__++];
            }
            current_statement_begin__ = 6;
            validate_non_negative_index("cl_std", "N", N);
            context__.validate_dims("data initialization", "cl_std", "vector_d", context__.to_vec(N));
            cl_std = Eigen::Matrix<double, Eigen::Dynamic, 1>(N);
            vals_r__ = context__.vals_r("cl_std");
            pos__ = 0;
            size_t cl_std_j_1_max__ = N;
            for (size_t j_1__ = 0; j_1__ < cl_std_j_1_max__; ++j_1__) {
                cl_std(j_1__) = vals_r__[pos__++];
            }
            current_statement_begin__ = 7;
            validate_non_negative_index("bm", "N", N);
            context__.validate_dims("data initialization", "bm", "vector_d", context__.to_vec(N));
            bm = Eigen::Matrix<double, Eigen::Dynamic, 1>(N);
            vals_r__ = context__.vals_r("bm");
            pos__ = 0;
            size_t bm_j_1_max__ = N;
            for (size_t j_1__ = 0; j_1__ < bm_j_1_max__; ++j_1__) {
                bm(j_1__) = vals_r__[pos__++];
            }
            // initialize transformed data variables
            // execute transformed data statements
            // validate transformed data
            // validate, set parameter ranges
            num_params_r__ = 0U;
            param_ranges_i__.clear();
            current_statement_begin__ = 11;
            num_params_r__ += 1;
            current_statement_begin__ = 12;
            num_params_r__ += 1;
            current_statement_begin__ = 13;
            num_params_r__ += 1;
            current_statement_begin__ = 14;
            num_params_r__ += 1;
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e, current_statement_begin__, prog_reader__());
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }
    }
    ~model_bayesian_meta_analysis_tissue() { }
    void transform_inits(const stan::io::var_context& context__,
                         std::vector<int>& params_i__,
                         std::vector<double>& params_r__,
                         std::ostream* pstream__) const {
        typedef double local_scalar_t__;
        stan::io::writer<double> writer__(params_r__, params_i__);
        size_t pos__;
        (void) pos__; // dummy call to supress warning
        std::vector<double> vals_r__;
        std::vector<int> vals_i__;
        current_statement_begin__ = 11;
        if (!(context__.contains_r("global_mean")))
            stan::lang::rethrow_located(std::runtime_error(std::string("Variable global_mean missing")), current_statement_begin__, prog_reader__());
        vals_r__ = context__.vals_r("global_mean");
        pos__ = 0U;
        context__.validate_dims("parameter initialization", "global_mean", "double", context__.to_vec());
        double global_mean(0);
        global_mean = vals_r__[pos__++];
        try {
            writer__.scalar_unconstrain(global_mean);
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(std::runtime_error(std::string("Error transforming variable global_mean: ") + e.what()), current_statement_begin__, prog_reader__());
        }
        current_statement_begin__ = 12;
        if (!(context__.contains_r("beta_std")))
            stan::lang::rethrow_located(std::runtime_error(std::string("Variable beta_std missing")), current_statement_begin__, prog_reader__());
        vals_r__ = context__.vals_r("beta_std");
        pos__ = 0U;
        context__.validate_dims("parameter initialization", "beta_std", "double", context__.to_vec());
        double beta_std(0);
        beta_std = vals_r__[pos__++];
        try {
            writer__.scalar_lb_unconstrain(0.001, beta_std);
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(std::runtime_error(std::string("Error transforming variable beta_std: ") + e.what()), current_statement_begin__, prog_reader__());
        }
        current_statement_begin__ = 13;
        if (!(context__.contains_r("ss_std")))
            stan::lang::rethrow_located(std::runtime_error(std::string("Variable ss_std missing")), current_statement_begin__, prog_reader__());
        vals_r__ = context__.vals_r("ss_std");
        pos__ = 0U;
        context__.validate_dims("parameter initialization", "ss_std", "double", context__.to_vec());
        double ss_std(0);
        ss_std = vals_r__[pos__++];
        try {
            writer__.scalar_lb_unconstrain(0.001, ss_std);
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(std::runtime_error(std::string("Error transforming variable ss_std: ") + e.what()), current_statement_begin__, prog_reader__());
        }
        current_statement_begin__ = 14;
        if (!(context__.contains_r("beta_biomarker_raw")))
            stan::lang::rethrow_located(std::runtime_error(std::string("Variable beta_biomarker_raw missing")), current_statement_begin__, prog_reader__());
        vals_r__ = context__.vals_r("beta_biomarker_raw");
        pos__ = 0U;
        context__.validate_dims("parameter initialization", "beta_biomarker_raw", "double", context__.to_vec());
        double beta_biomarker_raw(0);
        beta_biomarker_raw = vals_r__[pos__++];
        try {
            writer__.scalar_unconstrain(beta_biomarker_raw);
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(std::runtime_error(std::string("Error transforming variable beta_biomarker_raw: ") + e.what()), current_statement_begin__, prog_reader__());
        }
        params_r__ = writer__.data_r();
        params_i__ = writer__.data_i();
    }
    void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream__) const {
      std::vector<double> params_r_vec;
      std::vector<int> params_i_vec;
      transform_inits(context, params_i_vec, params_r_vec, pstream__);
      params_r.resize(params_r_vec.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r(i) = params_r_vec[i];
    }
    template <bool propto__, bool jacobian__, typename T__>
    T__ log_prob(std::vector<T__>& params_r__,
                 std::vector<int>& params_i__,
                 std::ostream* pstream__ = 0) const {
        typedef T__ local_scalar_t__;
        local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // dummy to suppress unused var warning
        T__ lp__(0.0);
        stan::math::accumulator<T__> lp_accum__;
        try {
            stan::io::reader<local_scalar_t__> in__(params_r__, params_i__);
            // model parameters
            current_statement_begin__ = 11;
            local_scalar_t__ global_mean;
            (void) global_mean;  // dummy to suppress unused var warning
            if (jacobian__)
                global_mean = in__.scalar_constrain(lp__);
            else
                global_mean = in__.scalar_constrain();
            current_statement_begin__ = 12;
            local_scalar_t__ beta_std;
            (void) beta_std;  // dummy to suppress unused var warning
            if (jacobian__)
                beta_std = in__.scalar_lb_constrain(0.001, lp__);
            else
                beta_std = in__.scalar_lb_constrain(0.001);
            current_statement_begin__ = 13;
            local_scalar_t__ ss_std;
            (void) ss_std;  // dummy to suppress unused var warning
            if (jacobian__)
                ss_std = in__.scalar_lb_constrain(0.001, lp__);
            else
                ss_std = in__.scalar_lb_constrain(0.001);
            current_statement_begin__ = 14;
            local_scalar_t__ beta_biomarker_raw;
            (void) beta_biomarker_raw;  // dummy to suppress unused var warning
            if (jacobian__)
                beta_biomarker_raw = in__.scalar_constrain(lp__);
            else
                beta_biomarker_raw = in__.scalar_constrain();
            // transformed parameters
            current_statement_begin__ = 18;
            validate_non_negative_index("shifted_mean", "N", N);
            Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, 1> shifted_mean(N);
            stan::math::initialize(shifted_mean, DUMMY_VAR__);
            stan::math::fill(shifted_mean, DUMMY_VAR__);
            current_statement_begin__ = 19;
            local_scalar_t__ beta_biomarker;
            (void) beta_biomarker;  // dummy to suppress unused var warning
            stan::math::initialize(beta_biomarker, DUMMY_VAR__);
            stan::math::fill(beta_biomarker, DUMMY_VAR__);
            // transformed parameters block statements
            current_statement_begin__ = 21;
            stan::math::assign(beta_biomarker, (beta_biomarker_raw * beta_std));
            current_statement_begin__ = 22;
            stan::math::assign(shifted_mean, add(global_mean, multiply(beta_biomarker, bm)));
            // validate transformed parameters
            const char* function__ = "validate transformed params";
            (void) function__;  // dummy to suppress unused var warning
            current_statement_begin__ = 18;
            size_t shifted_mean_j_1_max__ = N;
            for (size_t j_1__ = 0; j_1__ < shifted_mean_j_1_max__; ++j_1__) {
                if (stan::math::is_uninitialized(shifted_mean(j_1__))) {
                    std::stringstream msg__;
                    msg__ << "Undefined transformed parameter: shifted_mean" << "(" << j_1__ << ")";
                    stan::lang::rethrow_located(std::runtime_error(std::string("Error initializing variable shifted_mean: ") + msg__.str()), current_statement_begin__, prog_reader__());
                }
            }
            current_statement_begin__ = 19;
            if (stan::math::is_uninitialized(beta_biomarker)) {
                std::stringstream msg__;
                msg__ << "Undefined transformed parameter: beta_biomarker";
                stan::lang::rethrow_located(std::runtime_error(std::string("Error initializing variable beta_biomarker: ") + msg__.str()), current_statement_begin__, prog_reader__());
            }
            // model body
            current_statement_begin__ = 27;
            lp_accum__.add(exponential_log<propto__>(beta_std, 10));
            current_statement_begin__ = 28;
            lp_accum__.add(normal_log<propto__>(beta_biomarker_raw, 0, 1));
            current_statement_begin__ = 30;
            lp_accum__.add(normal_log<propto__>(suff_stat, shifted_mean, stan::math::sqrt(add(square(cl_std), square(ss_std)))));
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e, current_statement_begin__, prog_reader__());
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }
        lp_accum__.add(lp__);
        return lp_accum__.sum();
    } // log_prob()
    template <bool propto, bool jacobian, typename T_>
    T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
               std::ostream* pstream = 0) const {
      std::vector<T_> vec_params_r;
      vec_params_r.reserve(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        vec_params_r.push_back(params_r(i));
      std::vector<int> vec_params_i;
      return log_prob<propto,jacobian,T_>(vec_params_r, vec_params_i, pstream);
    }
    void get_param_names(std::vector<std::string>& names__) const {
        names__.resize(0);
        names__.push_back("global_mean");
        names__.push_back("beta_std");
        names__.push_back("ss_std");
        names__.push_back("beta_biomarker_raw");
        names__.push_back("shifted_mean");
        names__.push_back("beta_biomarker");
    }
    void get_dims(std::vector<std::vector<size_t> >& dimss__) const {
        dimss__.resize(0);
        std::vector<size_t> dims__;
        dims__.resize(0);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dims__.push_back(N);
        dimss__.push_back(dims__);
        dims__.resize(0);
        dimss__.push_back(dims__);
    }
    template <typename RNG>
    void write_array(RNG& base_rng__,
                     std::vector<double>& params_r__,
                     std::vector<int>& params_i__,
                     std::vector<double>& vars__,
                     bool include_tparams__ = true,
                     bool include_gqs__ = true,
                     std::ostream* pstream__ = 0) const {
        typedef double local_scalar_t__;
        vars__.resize(0);
        stan::io::reader<local_scalar_t__> in__(params_r__, params_i__);
        static const char* function__ = "model_bayesian_meta_analysis_tissue_namespace::write_array";
        (void) function__;  // dummy to suppress unused var warning
        // read-transform, write parameters
        double global_mean = in__.scalar_constrain();
        vars__.push_back(global_mean);
        double beta_std = in__.scalar_lb_constrain(0.001);
        vars__.push_back(beta_std);
        double ss_std = in__.scalar_lb_constrain(0.001);
        vars__.push_back(ss_std);
        double beta_biomarker_raw = in__.scalar_constrain();
        vars__.push_back(beta_biomarker_raw);
        double lp__ = 0.0;
        (void) lp__;  // dummy to suppress unused var warning
        stan::math::accumulator<double> lp_accum__;
        local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
        (void) DUMMY_VAR__;  // suppress unused var warning
        if (!include_tparams__ && !include_gqs__) return;
        try {
            // declare and define transformed parameters
            current_statement_begin__ = 18;
            validate_non_negative_index("shifted_mean", "N", N);
            Eigen::Matrix<double, Eigen::Dynamic, 1> shifted_mean(N);
            stan::math::initialize(shifted_mean, DUMMY_VAR__);
            stan::math::fill(shifted_mean, DUMMY_VAR__);
            current_statement_begin__ = 19;
            double beta_biomarker;
            (void) beta_biomarker;  // dummy to suppress unused var warning
            stan::math::initialize(beta_biomarker, DUMMY_VAR__);
            stan::math::fill(beta_biomarker, DUMMY_VAR__);
            // do transformed parameters statements
            current_statement_begin__ = 21;
            stan::math::assign(beta_biomarker, (beta_biomarker_raw * beta_std));
            current_statement_begin__ = 22;
            stan::math::assign(shifted_mean, add(global_mean, multiply(beta_biomarker, bm)));
            if (!include_gqs__ && !include_tparams__) return;
            // validate transformed parameters
            const char* function__ = "validate transformed params";
            (void) function__;  // dummy to suppress unused var warning
            // write transformed parameters
            if (include_tparams__) {
                size_t shifted_mean_j_1_max__ = N;
                for (size_t j_1__ = 0; j_1__ < shifted_mean_j_1_max__; ++j_1__) {
                    vars__.push_back(shifted_mean(j_1__));
                }
                vars__.push_back(beta_biomarker);
            }
            if (!include_gqs__) return;
        } catch (const std::exception& e) {
            stan::lang::rethrow_located(e, current_statement_begin__, prog_reader__());
            // Next line prevents compiler griping about no return
            throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***");
        }
    }
    template <typename RNG>
    void write_array(RNG& base_rng,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                     Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                     bool include_tparams = true,
                     bool include_gqs = true,
                     std::ostream* pstream = 0) const {
      std::vector<double> params_r_vec(params_r.size());
      for (int i = 0; i < params_r.size(); ++i)
        params_r_vec[i] = params_r(i);
      std::vector<double> vars_vec;
      std::vector<int> params_i_vec;
      write_array(base_rng, params_r_vec, params_i_vec, vars_vec, include_tparams, include_gqs, pstream);
      vars.resize(vars_vec.size());
      for (int i = 0; i < vars.size(); ++i)
        vars(i) = vars_vec[i];
    }
    static std::string model_name() {
        return "model_bayesian_meta_analysis_tissue";
    }
    void constrained_param_names(std::vector<std::string>& param_names__,
                                 bool include_tparams__ = true,
                                 bool include_gqs__ = true) const {
        std::stringstream param_name_stream__;
        param_name_stream__.str(std::string());
        param_name_stream__ << "global_mean";
        param_names__.push_back(param_name_stream__.str());
        param_name_stream__.str(std::string());
        param_name_stream__ << "beta_std";
        param_names__.push_back(param_name_stream__.str());
        param_name_stream__.str(std::string());
        param_name_stream__ << "ss_std";
        param_names__.push_back(param_name_stream__.str());
        param_name_stream__.str(std::string());
        param_name_stream__ << "beta_biomarker_raw";
        param_names__.push_back(param_name_stream__.str());
        if (!include_gqs__ && !include_tparams__) return;
        if (include_tparams__) {
            size_t shifted_mean_j_1_max__ = N;
            for (size_t j_1__ = 0; j_1__ < shifted_mean_j_1_max__; ++j_1__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "shifted_mean" << '.' << j_1__ + 1;
                param_names__.push_back(param_name_stream__.str());
            }
            param_name_stream__.str(std::string());
            param_name_stream__ << "beta_biomarker";
            param_names__.push_back(param_name_stream__.str());
        }
        if (!include_gqs__) return;
    }
    void unconstrained_param_names(std::vector<std::string>& param_names__,
                                   bool include_tparams__ = true,
                                   bool include_gqs__ = true) const {
        std::stringstream param_name_stream__;
        param_name_stream__.str(std::string());
        param_name_stream__ << "global_mean";
        param_names__.push_back(param_name_stream__.str());
        param_name_stream__.str(std::string());
        param_name_stream__ << "beta_std";
        param_names__.push_back(param_name_stream__.str());
        param_name_stream__.str(std::string());
        param_name_stream__ << "ss_std";
        param_names__.push_back(param_name_stream__.str());
        param_name_stream__.str(std::string());
        param_name_stream__ << "beta_biomarker_raw";
        param_names__.push_back(param_name_stream__.str());
        if (!include_gqs__ && !include_tparams__) return;
        if (include_tparams__) {
            size_t shifted_mean_j_1_max__ = N;
            for (size_t j_1__ = 0; j_1__ < shifted_mean_j_1_max__; ++j_1__) {
                param_name_stream__.str(std::string());
                param_name_stream__ << "shifted_mean" << '.' << j_1__ + 1;
                param_names__.push_back(param_name_stream__.str());
            }
            param_name_stream__.str(std::string());
            param_name_stream__ << "beta_biomarker";
            param_names__.push_back(param_name_stream__.str());
        }
        if (!include_gqs__) return;
    }
}; // model
}  // namespace
typedef model_bayesian_meta_analysis_tissue_namespace::model_bayesian_meta_analysis_tissue stan_model;
#endif
