//
// Created by unicorn on 2020/9/13.
//

#ifndef UNT_STEREO_RECTIFY_H
#define UNT_STEREO_RECTIFY_H
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
void computeCorrespond(const cv::Mat &img1,const cv::Mat &img2,vector<cv::KeyPoint> &kps1,vector<cv::KeyPoint> &kps2,vector<cv::DMatch> &matches){
    cv::Mat mask;
    cv::Ptr<cv::Feature2D> brisk= cv::BRISK::create(30,3,1.0f);
    cv::Mat desp1,desp2;
    brisk->detectAndCompute(img1,mask,kps1,desp1);
    brisk->detectAndCompute(img2,mask,kps2,desp2);
    cv::Ptr<cv::DescriptorMatcher> matcher=new cv::BFMatcher(cv::NORM_HAMMING,true);
    matcher->match(desp1,desp2,matches);
}



cv::Mat cross(cv::Vec3d e){
    cv::Mat ex=(cv::Mat_<double>(3,3)<<0, -e[2], e[1],
            e[2], 0, -e[0],
            -e[1], e[0], 0);
    return ex.clone();
}
void getLeftCoeffAB(cv::Mat &img1,cv::Mat &A,cv::Mat &B,cv::Vec3d ve){
    int w=img1.cols,h=img1.rows;
    cv::Mat PPT=(cv::Mat_<double>(3,3)<<w*w-1,0,0,
            0,h*h-1,0,
            0,0,0);
    PPT*=((w*h)/12.0);
    cv::Mat pcpct=(cv::Mat_<double>(3,3)<<pow(w-1,2),(w-1)*(h-1),2*(w-1),
            (w-1)*(h-1),pow(h-1,2),2*(h-1),
            2*(w-1),2*(h-1),4);
    pcpct/=4.0;
    A= cross(ve).t() * PPT * cross(ve);
    B= cross(ve).t() * pcpct * cross(ve);
}

void getRightCoeffAB(cv::Mat &img2,cv::Mat &A,cv::Mat &B,cv::Mat &F){
    int w=img2.cols,h=img2.rows;
    cv::Mat PPT=(cv::Mat_<double>(3,3)<<w*w-1,0,0,
            0,h*h-1,0,
            0,0,0);
    PPT*=((w*h)/12.0);
    cv::Mat pcpct=(cv::Mat_<double>(3,3)<<pow(w-1,2),(w-1)*(h-1),2*(w-1),
            (w-1)*(h-1),pow(h-1,2),2*(h-1),
            2*(w-1),2*(h-1),4);
    pcpct/=4.0;
    A=F.t()*PPT*F;
    B=F.t()*pcpct*F;
}

cv::Vec2d getInitZ(cv::Mat &A,cv::Mat &B){
    cv::SVD cholesky(A, cv::SVD::FULL_UV);
    cv::Mat sqrtW=cv::Mat::zeros(3,3,CV_64F);
    for (int i = 0; i <3 ; ++i) {
        sqrtW.at<double>(i,i)=sqrt(cholesky.w.at<double>(i));
    }
    cv::Mat L=cholesky.u*sqrtW;
    cv::Mat D=L.t();
    cv::Mat C=D.inv().t()*B*D.inv();
    cv::SVD Cz(C.colRange(0,2).rowRange(0,2));
    cv::Vec2d z=Cz.vt.row(0);
    return z;
}
double GetVc(const cv::Mat &img,const cv::Mat &Hp){
    vector<cv::Point2d> corners(4), corners_trans(4);

    corners[0] = cv::Point2d(0,0);
    corners[1] = cv::Point2d(img.cols,0);
    corners[2] = cv::Point2d(img.cols,img.rows);
    corners[3] = cv::Point2d(0,img.rows);

    perspectiveTransform(corners, corners_trans, Hp);

    double min_y;
    min_y = INT_MAX;

    for (int j = 0; j < 4; j++) {
        min_y = min(corners_trans[j].y, min_y);
    }
    return min_y;
}

void getSimilarityTransform(const cv::Mat &F,const cv::Mat &w,const cv::Mat &wp,const double vcp,cv::Mat &Hr,cv::Mat &Hr_p){

    double F32=F.at<double>(2,1);
    double F33=F.at<double>(2,2);
    double F31=F.at<double>(2,0);
    double F23=F.at<double>(1,2);
    double F13=F.at<double>(0,2);
    double wa=w.at<double>(0,0);
    double wb=w.at<double>(1,0);
    double wap=wp.at<double>(0,0);
    double wbp=wp.at<double>(1,0);
    Hr=(cv::Mat_<double>(3,3)<<
                             F32-wb*F33,wa*F33-F31,0,
                            F31-wa*F33,F32-wb*F33,F33+vcp,
                            0,0,1);
    Hr_p=(cv::Mat_<double>(3,3)<<
                    -F23+wbp*F33,-wap*F33+F13,0,
                    -F13+wap*F33,-F23+wbp*F33,vcp,
                    0,0,1);
}




void getShearingTransform(const cv::Mat &img,const cv::Mat &H,cv::Mat &HS){
    int w = img.cols;
    int h = img.rows;
    cv::Point2d a((w-1)/2, 0);
    cv::Point2d b(w-1, (h-1)/2);
    cv::Point2d c((w-1)/2, h-1);
    cv::Point2d d(0, (h-1)/2);

    vector<cv::Point2d> midpoints, midpoints_hat;
    midpoints.push_back(a);
    midpoints.push_back(b);
    midpoints.push_back(c);
    midpoints.push_back(d);

    perspectiveTransform(midpoints, midpoints_hat, H);

    cv::Point2d x = midpoints_hat[1] - midpoints_hat[3];
    cv::Point2d y = midpoints_hat[2] - midpoints_hat[0];

    double coeff_a = (h*h*x.y*x.y + w*w*y.y*y.y) / (h*w * (x.y*y.x - x.x*y.y));
    double coeff_b = (h*h*x.x*x.y + w*w*y.x*y.y) / (h*w * (x.x*y.y - x.y*y.x));

    HS = cv::Mat::eye(3, 3, CV_64F);
    HS.at<double>(0,0) = coeff_a;
    HS.at<double>(0,1) = coeff_b;

    if( coeff_a < 0 ){
        coeff_a *= -1;
        coeff_b *= -1;
        HS.at<double>(0,0) = coeff_a;
        HS.at<double>(0,1) = coeff_b;
    }

}

double getScale(const cv::Mat &img1,const cv::Mat &img2,const cv::Mat &H,const cv::Mat &Hp){
    double A = img1.cols*img1.rows + img2.cols*img2.rows;
    double Ap = 0;

    vector<cv::Point2f> corners(4), corners_trans(4);

    corners[0] = cv::Point2f(0,0);
    corners[1] = cv::Point2f(img1.cols,0);
    corners[2] = cv::Point2f(img1.cols,img1.rows);
    corners[3] = cv::Point2f(0,img1.rows);

    perspectiveTransform(corners, corners_trans, H);
    Ap += contourArea(corners_trans);

    corners[0] = cv::Point2f(0,0);
    corners[1] = cv::Point2f(img2.cols,0);
    corners[2] = cv::Point2f(img2.cols,img2.rows);
    corners[3] = cv::Point2f(0,img2.rows);

    perspectiveTransform(corners, corners_trans, Hp);
    Ap += contourArea(corners_trans);

    return sqrt(A/Ap);

}

bool isImageInverted(const cv::Mat &img, const cv::Mat &homography){
    vector<cv::Point2d> corners(2), corners_trans(2);

    corners[0] = cv::Point2d(0,0);
    corners[1] = cv::Point2d(0,img.rows);

    perspectiveTransform(corners, corners_trans, homography);

    return corners_trans[1].y - corners_trans[0].y < 0.0;
}
void refineH(const cv::Mat &img1,const cv::Mat &img2,cv::Mat &H,cv::Mat &H_prime){

    double s=getScale(img1,img2,H,H_prime);

    cv::Mat Trans = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat Trans_prime = cv::Mat::eye(3, 3, CV_64F);

    cv::Mat Scale = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat Scale_prime = cv::Mat::eye(3, 3, CV_64F);

    Scale.at<double>(0,0) = Scale.at<double>(1,1) = s;
    Scale_prime.at<double>(0,0) = Scale_prime.at<double>(1,1) = s;

    if(isImageInverted(img1, Scale*H)){
        Scale.at<double>(0,0) = Scale.at<double>(1,1) = -s;
        Scale_prime.at<double>(0,0) = Scale_prime.at<double>(1,1) = -s;
    }

    vector<cv::Point2d> corners(4), corners_trans(4);
    corners[0] = cv::Point2d(0,0);
    corners[1] = cv::Point2d(img1.cols,0);
    corners[2] = cv::Point2d(img1.cols,img1.rows);
    corners[3] = cv::Point2d(0,img1.rows);

    perspectiveTransform(corners, corners_trans, Scale*H);

    double min_x_1 =INT_MAX, min_y_1 = INT_MAX;
    for (int j = 0; j < 4; j++) {
        min_x_1 = min(corners_trans[j].x, min_x_1);
        min_y_1 = min(corners_trans[j].y, min_y_1);
    }

    corners[0] = cv::Point2d(0,0);
    corners[1] = cv::Point2d(img2.cols,0);
    corners[2] = cv::Point2d(img2.cols,img2.rows);
    corners[3] = cv::Point2d(0,img2.rows);

    perspectiveTransform(corners, corners_trans, Scale_prime*H_prime);

    double min_x_2 =INT_MAX, min_y_2 = INT_MAX;
    for (int j = 0; j < 4; j++) {
        min_x_2 = min(corners_trans[j].x, min_x_2);
        min_y_2 = min(corners_trans[j].y, min_y_2);
    }

    double min_y = min_y_1 < min_y_2 ? min_y_1 : min_y_2;

    Trans.at<double>(0,2) = -min_x_1;
    Trans_prime.at<double>(0,2) = -min_x_2;

    Trans.at<double>(1,2) = Trans_prime.at<double>(1,2) = -min_y;

    H= Trans*Scale*H;
    H_prime = Trans_prime*Scale_prime*H_prime;
}

cv::Size getSize(const cv::Mat &img1,const cv::Mat &H){
    vector<cv::Point2d> corners_all(4), corners_all_t(4);
    double min_x, min_y, max_x, max_y;
    min_x = min_y = INT_MAX;
    max_x = max_y = INT_MIN;

    corners_all[0] = cv::Point2d(0,0);
    corners_all[1] = cv::Point2d(img1.cols,0);
    corners_all[2] = cv::Point2d(img1.cols,img1.rows);
    corners_all[3] = cv::Point2d(0,img1.rows);

    perspectiveTransform(corners_all, corners_all_t, H);

    for (int j = 0; j < 4; j++) {
        min_x = min(corners_all_t[j].x, min_x);
        max_x = max(corners_all_t[j].x, max_x);

        min_y = min(corners_all_t[j].y, min_y);
        max_y = max(corners_all_t[j].y, max_y);
    }

    int img_1_cols = max_x - min_x;
    int img_1_rows = max_y - min_y;
    return cv::Size(img_1_cols,img_1_rows);
}

#endif //UNT_STEREO_RECTIFY_H
