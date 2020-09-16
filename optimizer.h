//
// Created by unicorn on 2020/9/14.
//

#ifndef UNT_OPTIMIZER_H
#define UNT_OPTIMIZER_H
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
using namespace ceres;
struct ProcjetionFunctor{
    template <typename T>
    bool operator()(const T* const x,T *residual)const{
        T x0=x[0],x1=x[1];
        T a1=T(a[0])*ceres::pow(x0,2)+T(a[2]+a[1])*x0*x1+T(a[3])*ceres::pow(x[1],2);
        T b1=T(b[0])*ceres::pow(x0,2)+T(b[2]+b[1])*x0*x1+T(b[3])*ceres::pow(x[1],2);
        T ap1=T(ap[0])*ceres::pow(x0,2)+T(ap[2]+ap[1])*x0*x1+T(ap[3])*ceres::pow(x[1],2);
        T bp1=T(bp[0])*ceres::pow(x0,2)+T(bp[2]+bp[1])*x0*x1+T(bp[3])*ceres::pow(x[1],2);
        residual[0]=(a1/b1)+(ap1/bp1);
        return true;
    }
    ProcjetionFunctor(cv::Mat &A,cv::Mat &B,cv::Mat &Ap,cv::Mat &Bp){
        A=A.reshape(0,1);
        B=B.reshape(0,1);
        Ap=Ap.reshape(0,1);
        Bp=Bp.reshape(0,1);
        for (int i = 0; i <4 ; ++i) {
            a[i]=A.at<double>(0,i);
            b[i]=B.at<double>(0,i);
            ap[i]=Ap.at<double>(0,i);
            bp[i]=Bp.at<double>(0,i);
        }
    }
    double a[4];
    double b[4];
    double ap[4];
    double bp[4];
};

void refineX(cv::Mat &A,cv::Mat &B,cv::Mat &Ap,cv::Mat &Bp,cv::Vec2d &z,double *x){
    cv::Mat As=A.rowRange(0,2).colRange(0,2).clone();
    cv::Mat Bs=B.rowRange(0,2).colRange(0,2).clone();
    cv::Mat Aps=Ap.rowRange(0,2).colRange(0,2).clone();
    cv::Mat Bps=Bp.rowRange(0,2).colRange(0,2).clone();
    x[0]=z[0];
    x[1]=z[1];
    Problem problem;
    CostFunction* cost_function =new AutoDiffCostFunction<ProcjetionFunctor, 1, 2>(new ProcjetionFunctor(As,Bs,Aps,Bps));
    problem.AddResidualBlock(cost_function, NULL, x);
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
}


#endif //UNT_OPTIMIZER_H
