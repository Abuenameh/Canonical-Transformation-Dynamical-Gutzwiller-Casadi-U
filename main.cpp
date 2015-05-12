/* 
 * File:   main.cpp
 * Author: Abuenameh
 *
 * Created on 17 November 2014, 22:05
 */

#include <cstdlib>
#include <queue>

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/progress.hpp>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time.hpp>

using namespace boost;
using namespace boost::random;
using namespace boost::filesystem;
using namespace boost::posix_time;

#include <nlopt.hpp>

using namespace nlopt;

#include "gutzwiller.hpp"
#include "mathematica.hpp"
#include "casadi.hpp"

#include <casadi/interfaces/sundials/cvodes_interface.hpp>

double UWi(double W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

struct input {
    double tau;
};

struct results {
    double tau;
    double Ei;
    double Ef;
    double Q;
    double p;
    //    vector<vector<double>> bs;
    double U0;
    vector<double> J0;
    vector<complex<double>> b0;
    vector<complex<double>> bf;
    vector<vector<complex<double>>> f0;
    vector<vector<complex<double>>> ff;
    string runtime;
};

boost::mutex progress_mutex;
boost::mutex inputs_mutex;
boost::mutex problem_mutex;

boost::random::mt19937 rng;
boost::random::uniform_real_distribution<> uni(-1, 1);

void threadfunc(double Ji, double Jf, double mu, vector<double> xi, double tauf, queue<input>& inputs, vector<results>& res, progress_display& progress, barrier& bar, int thread) {
    DynamicsProblem* prob;

    {
        boost::mutex::scoped_lock lock(problem_mutex);

//        vector<double> f0(2 * L*dim, 1);
//        rng.seed();
//        for (int i = 0; i < 2 * L * dim; i++) {
//            f0[i] = uni(rng);
//        }
//
//        prob = new DynamicsProblem(Wi, Wf, mu, xi, f0);
        
        prob = new DynamicsProblem(tauf);
    }
    
    bar.wait();

    for (;;) {
        input in;
        {
            boost::mutex::scoped_lock lock(inputs_mutex);
            if (inputs.empty()) {
                break;
            }
            in = inputs.front();
            inputs.pop();
        }
        double tau = in.tau;

        results pointRes;
        pointRes.tau = tau;

        //    int ndim = 2 * L*dim;

        //        vector<double> f0(2 * L*dim, 1);

        //        {
        //            boost::mutex::scoped_lock lock(problem_mutex);
        //            prob->setParameters(Wi, Wf, tau, xi, mu);
        //            rng.seed();
        //            for (int i = 0; i < 2*L*dim; i++) {
        //                f0[i] = uni(rng);
        //            }
        //            
        //        }
        //        prob->setInitial(f0);
        try {
            //        prob->solve();
            prob->setTau(tau);
            prob->evolve();
            pointRes.Ei = prob->getEi();
            pointRes.Ef = prob->getEf();
            pointRes.Q = prob->getQ();
            pointRes.p = prob->getRho();
            pointRes.U0 = prob->getU0();
            pointRes.J0 = prob->getJ0();
            pointRes.b0 = prob->getB0();
            pointRes.bf = prob->getBf();
                        pointRes.f0 = prob->getF0();
                        pointRes.ff = prob->getFf();
            //            pointRes.bs = prob->getBs();
            pointRes.runtime = prob->getRuntime();
        }
        catch (std::exception& e) {
            cerr << "Failed at tau = " << tau << endl;
            cerr << e.what() << endl;
            pointRes.Ei = numeric_limits<double>::quiet_NaN();
            pointRes.Ef = numeric_limits<double>::quiet_NaN();
            pointRes.Q = numeric_limits<double>::quiet_NaN();
            pointRes.p = numeric_limits<double>::quiet_NaN();
            pointRes.U0 = numeric_limits<double>::quiet_NaN();
            pointRes.J0 = vector<double>(L, numeric_limits<double>::quiet_NaN());
            pointRes.b0 = vector<complex<double>>(L, numeric_limits<double>::quiet_NaN());
            pointRes.bf = vector<complex<double>>(L, numeric_limits<double>::quiet_NaN());
                        pointRes.f0 = vector<vector<complex<double>>>(L, vector<complex<double>>(dim, numeric_limits<double>::quiet_NaN()));
                        pointRes.ff = vector<vector<complex<double>>>(L, vector<complex<double>>(dim, numeric_limits<double>::quiet_NaN()));
            //            pointRes.bs = vector<vector<double>>();
            pointRes.runtime = "Failed";
        }

        //        pointRes.Ei = prob->getEi();
        //        pointRes.Ef = prob->getEf();
        //        pointRes.Q = prob->getQ();
        //        pointRes.p = prob->getRho();
        //        pointRes.bs = prob->getBs();
        //        pointRes.runtime = prob->getRuntime();

        {
            boost::mutex::scoped_lock lock(inputs_mutex);
            res.push_back(pointRes);
        }

        {
            boost::mutex::scoped_lock lock(progress_mutex);
            ++progress;
        }
    }
    
    bar.wait();

    {
        boost::mutex::scoped_lock lock(problem_mutex);
        delete prob;
    }


}

/*
 * 
 */
int main(int argc, char** argv) {

    ptime begin = microsec_clock::local_time();

    random::mt19937 rng;
    random::uniform_real_distribution<> uni(-1, 1);

    int seed = lexical_cast<int>(argv[1]);

    double Ji = lexical_cast<double>(argv[2]);
    double Jf = lexical_cast<double>(argv[3]);

    double mu = lexical_cast<double>(argv[4]);

    double Ui = 1;//UWi(Wi);

    double D = lexical_cast<double>(argv[5]);

    double taui = lexical_cast<double>(argv[6]);
    double tauf = lexical_cast<double>(argv[7]);
    int ntaus = lexical_cast<int>(argv[8]);
    //    double tf = 2*tau;
    //    int nsteps = lexical_cast<int>(argv[9]);

        double dt = lexical_cast<double>(argv[9]);
    //    int dnsav = lexical_cast<int>(argv[8]);

    //    int nsteps = (int) ceil(2 * tau / dt);
    //    double h = 2 * tau / nsteps;


    int numthreads = lexical_cast<int>(argv[10]);

    int resi = lexical_cast<int>(argv[11]);
    //    int resi = 0;
    //    if (argc > 9) {
    //        resi = lexical_cast<int>(argv[9]);
    //    }

    int subresi = -1;
    int seed2 = 0;
    if (argc > 12) {
//        subresi = lexical_cast<int>(argv[12]);
        seed2 = lexical_cast<int>(argv[12]);
    }

#ifdef AMAZON
    //    path resdir("/home/ubuntu/Results/Canonical Transformation Dynamical Gutzwiller");
    path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/Canonical Transformation Dynamical Gutzwiller J");
#else
    path resdir("/Users/Abuenameh/Documents/Simulation Results/Canonical Transformation Dynamical Gutzwiller J");
    //        path resdir("/Users/Abuenameh/Documents/Simulation Results/Dynamical Gutzwiller Hartmann Comparison");
#endif
    if (!exists(resdir)) {
        cerr << "Results directory " << resdir << " does not exist!" << endl;
        exit(1);
    }
    ostringstream oss;
    if (subresi == -1) {
        oss << "res." << resi << ".txt";
    }
    else {
        oss << "res." << resi << "." << subresi << ".txt";
    }
    path resfile = resdir / oss.str();
    //#ifndef AMAZON
    while (exists(resfile)) {
        resi++;
        oss.str("");
        if (subresi == -1) {
            oss << "res." << resi << ".txt";
        }
        else {
            oss << "res." << resi << "." << subresi << ".txt";
        }
        resfile = resdir / oss.str();
    }
    //#endif
    //        if (seed == -1) {
    //            resi = -1;
    if (seed < 0) {
        resi = seed;
        oss.str("");
        oss << "res." << resi << ".txt";
        resfile = resdir / oss.str();
    }
    vector<double> xi(L, 1);
    rng.seed(seed);
    if (seed > -1) {
        for (int j = 0; j < L; j++) {
            xi[j] = (1 + D * uni(rng));
        }
    }

    //        double Ui = UWi(Wi);
    double mui = mu * Ui;

    filesystem::ofstream os(resfile);
        printMath(os, "seed", resi, seed);
        printMath(os, "Delta", resi, D);
    //    printMath(os, "Wres", resi, Wi);
    printMath(os, "mures", resi, mui);
    printMath(os, "Ures", resi, Ui);
    printMath(os, "xires", resi, xi);
    os << flush;

    printMath(os, "dtres", resi, dt);
    //    printMath(os, "tauires", resi, taui);
    //    printMath(os, "taufres", resi, tauf);
    //    printMath(os, "ntausres", resi, ntaus);

    os << flush;

        printMath(os, "Jires", resi, Ji);
        printMath(os, "Jfres", resi, Jf);
    os << flush;

    cout << "Res: " << resi << endl;

    queue<input> inputs;
    if (ntaus == 1) {
        input in;
        in.tau = taui;
        inputs.push(in);
    }
    else {
        for (int i = 0; i < ntaus; i++) {
            input in;
            double tau = taui + i * (tauf - taui) / (ntaus - 1);
            in.tau = tau;
            inputs.push(in);
        }
    }

    progress_display progress(inputs.size());

    vector<results> res;

        vector<double> f0(2 * L*dim, 1);
        rng.seed(seed2);
        for (int i = 0; i < 2 * L * dim; i++) {
            f0[i] = uni(rng);
        }

    DynamicsProblem::setup(Ji, Jf, mui, xi, f0, dt);
    
    barrier bar(numthreads);
    
    thread_group threads;
    for (int i = 0; i < numthreads; i++) {
        threads.create_thread(bind(&threadfunc, Ji, Jf, mui, xi, tauf, boost::ref(inputs), boost::ref(res), boost::ref(progress), boost::ref(bar), i));
    }
    threads.join_all();

    vector<double> taures;
    vector<double> Eires;
    vector<double> Efres;
    vector<double> Qres;
    vector<double> pres;
    vector<double> U0res;
    vector<vector<double>> J0res;
    vector<vector<complex<double>>> b0res;
    vector<vector<complex<double>>> bfres;
    vector<vector < vector<complex<double>>>> f0res;
    vector<vector < vector<complex<double>>>> ffres;
    //    vector<vector<vector<double>>> bsres;
    vector<string> runtimeres;

    for (results ires : res) {
        taures.push_back(ires.tau);
        Eires.push_back(ires.Ei);
        Efres.push_back(ires.Ef);
        Qres.push_back(ires.Q);
        pres.push_back(ires.p);
        U0res.push_back(ires.U0);
        J0res.push_back(ires.J0);
        b0res.push_back(ires.b0);
        bfres.push_back(ires.bf);
                f0res.push_back(ires.f0);
                ffres.push_back(ires.ff);
        //        bsres.push_back(ires.bs);
        runtimeres.push_back(ires.runtime);
    }

    printMath(os, "taures", resi, subresi, taures);
    printMath(os, "Eires", resi, subresi, Eires);
    printMath(os, "Efres", resi, subresi, Efres);
    printMath(os, "Qres", resi, subresi, Qres);
    printMath(os, "pres", resi, subresi, pres);
    printMath(os, "U0res", resi, subresi, U0res);
    printMath(os, "J0res", resi, subresi, J0res);
    printMath(os, "b0res", resi, subresi, b0res);
    printMath(os, "bfres", resi, subresi, bfres);
    printMath(os, "f0res", resi, subresi, f0res);
    printMath(os, "ffres", resi, subresi, ffres);
    //    printMath(os, "bsres", resi, bsres);
    printMath(os, "runtime", resi, subresi, runtimeres);

    ptime end = microsec_clock::local_time();
    time_period period(begin, end);
    cout << endl << period.length() << endl << endl;

    if (subresi == -1) {
        os << "totalruntime[" << resi << "]=\"" << period.length() << "\";" << endl;
    }
    else {
        os << "subtotalruntime[" << resi << "," << subresi << "]=\"" << period.length() << "\";" << endl;
    }

    return 0;

    //        cout << setprecision(10);
    //
    //    double Wi = 2e11;
    //    double Wf = 1e11;
    //    double tau = 1e-6;
    //    double mu = 0.5;
    //    vector<double> xi(L, 1);
    //
    //    int ndim = 2 * L*dim;
    //
    //    vector<double> f0(ndim, 1);
    //
    //    vector<double> Eiv, Efv, Qv, pv;
    //    vector<vector<double>> bv;
    //
    //    DynamicsProblem prob;
    //
    //    for (int i = 0; i < 10; i++) {
    //        prob.setParameters(Wi, Wf, tau + i * 0.1e-6, xi, mu);
    //        prob.setInitial(f0);
    //        prob.solve();
    //        prob.evolve();
    //
    //        Eiv.push_back(prob.getEi());
    //        Efv.push_back(prob.getEf());
    //        Qv.push_back(prob.getQ());
    //        pv.push_back(prob.getRho());
    //        bv.push_back(prob.getBs());
    //    }
    //    
    //    cout << math(bv) << endl;
    //    cout << math(Eiv) << endl;
    //    cout << math(Efv) << endl;
    //    cout << math(Qv) << endl;
    //    cout << math(pv) << endl;

    //    cout << prob.getGSRuntime() << endl;
    //    cout << prob.getRuntime() << endl;

    return 0;


    return 0;
}

