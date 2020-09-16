//
// Created by unicorn on 2020/9/15.
//

#ifndef UNT_VIEWER_H
#define UNT_VIEWER_H
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
void drawEpilines(const  cv::Mat &img1,const cv::Mat &img2,const  vector<cv::KeyPoint> &kps1,const vector<cv::KeyPoint> &kps2,
                  vector<cv::DMatch> &matches,const cv::Mat &F,const cv::Point2f e,const cv::Point2f e_prime){
    cv::RNG rng;
    cv::theRNG().state = clock();
    cv::Mat epoImage2=img2.clone();
    cv::Mat epoImage1=img1.clone();
    for (auto iter=matches.begin(),next=matches.begin();iter!=matches.end();iter=next) {
        next++;
        cv::Scalar color(rng.uniform(0, 255),
                         rng.uniform(0, 255),
                         rng.uniform(0, 255));

        cv::Point2f p1=kps1[iter->queryIdx].pt;
        cv::Mat_<double> p1m=cv::Mat_<double>(3,1);
        p1m<<p1.x,p1.y,1.0;

        cv::Point2f p2=kps2[iter->trainIdx].pt;
        cv::Mat_<double> p2m=cv::Mat_<double>(3,1);
        p2m<<p2.x,p2.y,1.0;


        circle(epoImage2,p2,4,color,CV_FILLED);
        circle(epoImage1,p1,4,color,CV_FILLED);


        cv::Mat_<double> n=F*p1m;
        cv::Point2f pr1;
        pr1.x=0.0;
        pr1.y=-n.at<double>(2,0)/n.at<double>(1,0);
        cv::Point2f pr2;
        pr2.x=img2.cols;
        pr2.y=(-n.at<double>(2,0)-n.at<double>(0,0)*img2.cols)/n.at<double>(1,0);
        cv::line(epoImage2,pr1,pr2,color);


        cv::Mat_<double> n1=F.t()*p2m;
        cv::Point2f pl1;
        pl1.x=0.0;
        pl1.y=-n1.at<double>(2,0)/n1.at<double>(1,0);
        cv::Point2f pl2;
        pl2.x=img1.cols;
        pl2.y=(-n1.at<double>(2,0)-n1.at<double>(0,0)*img1.cols)/n1.at<double>(1,0);
        cv::line(epoImage1,pl1,pl2,color);

    }
    cv::circle(epoImage2,e_prime,4,cv::Scalar(255,255,0),2);
    cv::circle(epoImage1,e,4,cv::Scalar(255,255,0),2);
    cv::imshow("original right",epoImage2);
    cv::imshow("original left",epoImage1);
}
void viewProjectiveResult(const cv::Mat &img,const cv::Mat &Hp,string winname){
    cv::Mat Hps=Hp.clone();
    double A = img.cols*img.rows;
    double Ap = 0;

    vector<cv::Point2f> corners(4), corners_trans(4);

    corners[0] = cv::Point2f(0,0);
    corners[1] = cv::Point2f(img.cols,0);
    corners[2] = cv::Point2f(img.cols,img.rows);
    corners[3] = cv::Point2f(0,img.rows);

    perspectiveTransform(corners, corners_trans, Hp);
    Ap += contourArea(corners_trans);
    double s= sqrt(A/Ap);
    Hps.at<double>(0,0)*=s;
    Hps.at<double>(1,1)*=s;

    double umax=s*max(corners_trans[1].x,corners_trans[2].x);
    double vmax=s*max(corners_trans[2].y,corners_trans[3].y);

    cv::Mat result;
    cv::warpPerspective(img, result, Hps,cv::Size(umax,vmax));
    cv::imshow(winname,result);
}

void viewSimilarityResult(const cv::Mat &img,const cv::Mat &H,string winname){
    cv::Mat Hrp=H.clone();
    double A = img.cols*img.rows;
    double Ap = 0;

    vector<cv::Point2f> corners(4), corners_trans(4);

    corners[0] = cv::Point2f(0,0);
    corners[1] = cv::Point2f(img.cols,0);
    corners[2] = cv::Point2f(img.cols,img.rows);
    corners[3] = cv::Point2f(0,img.rows);

    perspectiveTransform(corners, corners_trans, Hrp);
    Ap += contourArea(corners_trans);
    double s= sqrt(A/Ap);
    if(isImageInverted(img, H)){
        Hrp.at<double>(0,0) *= -s;
        Hrp.at<double>(1,1) *= -s;
        perspectiveTransform(corners, corners_trans, Hrp);
    }else{
        Hrp.at<double>(0,0)*=s;
        Hrp.at<double>(1,1)*=s;
        perspectiveTransform(corners, corners_trans, Hrp);
    }



    double umax=max(corners_trans[1].x,corners_trans[2].x);
    double umin=min(corners_trans[0].x,corners_trans[3].x);
    double vmax=max(corners_trans[2].y,corners_trans[3].y);
    double vmin=min(corners_trans[0].y,corners_trans[1].y);

    cv::Mat result;
    cv::warpPerspective(img, result, Hrp,cv::Size(umax-umin,vmax-vmin));
    cv::imshow(winname,result);
}

void drawEpilines(const  cv::Mat &img1,const cv::Mat &img2,const  vector<cv::Point2f> &vp1,const vector<cv::Point2f> &vp2
        ,const cv::Mat &F,const cv::Point2f e,const cv::Point2f e_prime){
    cv::RNG rng;
    cv::theRNG().state = clock();
    cv::Mat epoImage2=img2.clone();
    cv::Mat epoImage1=img1.clone();

    for (int m = 0; m <vp1.size() ; ++m) {
        cv::Scalar color(rng.uniform(0, 255),
                         rng.uniform(0, 255),
                         rng.uniform(0, 255));

        cv::Point2f p1=vp1[m];
        cv::Point2f p2=vp2[m];
        circle(epoImage1,p1,4,color,CV_FILLED);
        cv::Mat_<double> p1m=cv::Mat_<double>(3,1);
        p1m<<p1.x,p1.y,1.0;
        cv::Mat_<double> n=F*p1m;

        cv::Point2f pr1;
        pr1.x=0.0;
        pr1.y=-n.at<double>(2,0)/n.at<double>(1,0);

        cv::Point2f pr2;
        pr2.x=img2.cols;
        pr2.y=(-n.at<double>(2,0)-n.at<double>(0,0)*img2.cols)/n.at<double>(1,0);
        cv::line(epoImage2,pr1,pr2,color);


        circle(epoImage2,p2,4,color,CV_FILLED);
        cv::Mat_<double> p2m=cv::Mat_<double>(3,1);
        p2m<<p2.x,p2.y,1.0;
        cv::Mat_<double> n1=F.t()*p2m;

        cv::Point2f pl1;
        pl1.x=0.0;
        pl1.y=-n1.at<double>(2,0)/n1.at<double>(1,0);

        cv::Point2f pl2;
        pl2.x=img1.cols;
        pl2.y=(-n1.at<double>(2,0)-n1.at<double>(0,0)*img1.cols)/n1.at<double>(1,0);
        cv::line(epoImage1,pl1,pl2,color);

    }
    cv::circle(epoImage2,e_prime,4,cv::Scalar(255,255,0),2);
    cv::circle(epoImage1,e,4,cv::Scalar(255,255,0),2);

    cv::imshow("Final2",epoImage2);
    cv::imshow("Final1",epoImage1);
}

void drawMatches(const cv::Mat &img1,const  cv::Mat &img2,const  vector<cv::Point2f> p1,const  vector<cv::Point2f> p2,
        const  cv::Mat &H,const  cv::Mat &H_prime){
    {
        vector<cv::Point2f> vp1, vp2;
        perspectiveTransform(p1, vp1, H);
        perspectiveTransform(p2, vp2, H_prime);

        cv::Mat F=cv::findFundamentalMat(vp1,vp2,cv::FM_RANSAC,1.0,0.99);
        cv::Vec3d ve,veprime;
        cv::SVD svd(F);
        ve=svd.vt.row(2);
        veprime=(svd.u).col(2);
        cv::Point2d e(ve(0)/ve(2),ve(1)/ve(2));
        cv::Point2d e_prime(veprime(0)/veprime(2),veprime(1)/veprime(2));

        drawEpilines(img1,img2,vp1,vp2,F,e,e_prime);
    }
}
#endif //UNT_VIEWER_H
