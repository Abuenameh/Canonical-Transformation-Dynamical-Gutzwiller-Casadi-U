/* 
 * File:   casadi.hpp
 * Author: Abuenameh
 *
 * Created on 06 November 2014, 17:45
 */

#ifndef CASADI_HPP
#define	CASADI_HPP

#include <typeinfo>

#include <casadi/casadi.hpp>
#include <casadi/solvers/rk_integrator.hpp>
#include <casadi/solvers/collocation_integrator.hpp>
#include <casadi/interfaces/sundials/cvodes_interface.hpp>
#include <casadi/core/function/custom_function.hpp>

using namespace casadi;

#include <boost/date_time.hpp>

using namespace boost::posix_time;

#include <nlopt.hpp>

using namespace nlopt;

#include "gutzwiller.hpp"

class DynamicsProblem {
public:
    DynamicsProblem(double tauf);

    ~DynamicsProblem() {
        delete integrator;
    }

    void setTau(double tau_);

    static void setup(double Ji_, double Jf_, double mu_, double U_, vector<double>& xi_, vector<double>& f0_, double dt_);

    static double E(const vector<double>& f, vector<double>& grad);

    void evolve();

    vector<double> getGS() {
        return x0;
    };

    string& getGSRuntime() {
        return gsruntime;
    }

    string& getRuntime() {
        return runtime;
    }

    string& getGSResult() {
        return gsresult;
    }

    double getQ() {
        return Q;
    }

    double getRho() {
        return pd;
    }

    vector<complex<double>> getB0() {
        return b0;
    }

    vector<complex<double>> getBf() {
        return bf;
    }

    double getEi() {
        return E0;
    }

    double getEf() {
        return Ef;
    }

    double getU0() {
        return U00;
    }

    vector<double> getJ0() {
        return J0;
    }

    vector<vector<complex<double>>> getF0() {
        return f0;
    }

    vector<vector<complex<double>>> getFf() {
        return ff;
    }

    void start() {
        start_time = microsec_clock::local_time();
    }

    void stop() {
        stop_time = microsec_clock::local_time();
    }

private:

    ptime start_time;
    ptime stop_time;

    static double scale;

    complex<SX> HS();
    SX W();
    static SX energy(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu);
    static SX energy(int i, int n, vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu);
    SX energya(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu);
    SX energy0(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu);
    SX energync(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu);
    static SX canonical(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu);
    static SX canonical(int i, int n, vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu);
    SX canonicala(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu);

    double tf;
    
    static double dt;

    static vector<double> xi;
    static double U00;
    static vector<double> J0;

    vector<Function> odes;

    Function ode_func;
    //    CvodesInterface* integrator;
    RkIntegrator* integrator;

    static vector<double> sparams;
    static vector<double> gsparams;
    static vector<double> x0;

    vector<double> params;

    static vector<Function> Efunc;
    static vector<Function> Egradf;

    string gsruntime;
    string gsresult;

    string runtime;

    double E0;
    double Ef;
    double Q;
    double pd;
    vector<complex<double>> b0;
    vector<complex<double>> bf;

    vector<vector<complex<double>>> f0;
    vector<vector<complex<double>>> ff;

};

double energyfunc(const vector<double>& x, vector<double>& grad, void *data);

#endif	/* CASADI_HPP */

