#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "stereo_rectify.h"
#include "optimizer.h"
#include "viewer.h"
using namespace std;


int main() {
    string imgloc1="../img/1.png",imgloc2="../img/2.png";
    cv::Mat img1=cv::imread(imgloc1);
    cv::Mat img2=cv::imread(imgloc2);

    vector<cv::KeyPoint> kps1,kps2;
    vector<cv::DMatch> matches;
    computeCorrespond(img1,img2,kps1,kps2,matches);

    vector<cv::Point2f> p1,p2;
    for (int i = 0; i <matches.size() ; ++i) {
        p1.push_back(kps1[matches[i].queryIdx].pt);
        p2.push_back(kps2[matches[i].trainIdx].pt);
    }
    cv::Mat temp_F=cv::findFundamentalMat(p1,p2,cv::FM_RANSAC,1.0,0.99);

    vector<cv::DMatch> refined_matches;
    for (auto m:matches) {
        cv::Point2f p1=kps1[m.queryIdx].pt;
        cv::Mat_<double> p1m=cv::Mat_<double>(3,1);
        p1m<<p1.x,p1.y,1.0;

        cv::Point2f p2=kps2[m.trainIdx].pt;
        cv::Mat_<double> p2m=cv::Mat_<double>(3,1);
        p2m<<p2.x,p2.y,1.0;

        cv::Mat zereo=p2m.t()*temp_F*p1m;
        if(abs(cv::norm(zereo))>0.2){
            continue;
        }else{
            refined_matches.push_back(m);
        }
    }
    p1.clear();p2.clear();
    for (int i = 0; i <refined_matches.size() ; ++i) {
        p1.push_back(kps1[refined_matches[i].queryIdx].pt);
        p2.push_back(kps2[refined_matches[i].trainIdx].pt);
    }
    cv::Mat F=cv::findFundamentalMat(p1,p2,cv::FM_RANSAC,1.0,0.99);


    cv::Vec3d ve,veprime;
    cv::SVD svd(F);
    ve=svd.vt.row(2);
    veprime=(svd.u).col(2);
    cv::Point2f e(ve(0)/ve(2),ve(1)/ve(2));
    cv::Point2f e_prime(veprime(0)/veprime(2),veprime(1)/veprime(2));

    drawEpilines(img1,img2,kps1,kps2,refined_matches,F,e,e_prime);

    //Projective Transform
    cv::Mat A,B;
    cv::Mat Ap,Bp;
    getLeftCoeffAB(img1,A,B,ve);
    getRightCoeffAB(img2,Ap,Bp,F);
    cv::Vec2d z1=getInitZ(A,B);
    cv::Vec2d z2=getInitZ(Ap,Bp);
    cv::Vec2d z=(z1/cv::norm(z1)+z2/cv::norm(z2))/2.0;
    double x[2];
    refineX(A,B,Ap,Bp,z,x);
    cv::Vec3d Z;
    Z[0]=x[0]/x[1];Z[1]=1.0;Z[2]=0;

    cv::Mat w= cross(ve) * cv::Mat(Z);
    w/=w.at<double>(2,0);
    cv::Mat wp=F*cv::Mat(Z);
    wp/=wp.at<double>(2,0);

    cv::Mat Hp=cv::Mat::eye(3,3,CV_64F);
    Hp.at<double>(2,0)=w.at<double>(0,0);
    Hp.at<double>(2,1)=w.at<double>(1,0);

    cv::Mat Hp_prime=cv::Mat::eye(3,3,CV_64F);
    Hp_prime.at<double>(2,0)=wp.at<double>(0,0);
    Hp_prime.at<double>(2,1)=wp.at<double>(1,0);

//    cout<<"epipoles should locate at infinity"<<endl;
//    cout<<Hp*cv::Mat(ve)<<endl;
//    cout<<Hp_prime*cv::Mat(veprime)<<endl;

    viewProjectiveResult(img1,Hp,"image1 after projective transform");
    viewProjectiveResult(img2,Hp_prime,"image2 after projective transform");

    vector<cv::Point2f> hp1,hp2;
    cv::perspectiveTransform(p1,hp1,Hp);
    cv::perspectiveTransform(p2,hp2,Hp_prime);

    cv::Mat Hr,Hr_prime;
    double vcp=-min(GetVc(img1,Hp),GetVc(img2,Hp_prime));

    getSimilarityTransform(F,w,wp,vcp,Hr,Hr_prime);


    cv::Mat Ha,Ha_prime;
    Ha=Hr*Hp;
    Ha_prime=Hr_prime*Hp_prime;


    viewSimilarityResult(img1,Ha,"image1 after similarity and projective transform");
    viewSimilarityResult(img2,Ha_prime,"image2 after similarity and projective transform");

    cv::Mat Hs,Hs_prime;
    getShearingTransform(img1,Ha,Hs);
    getShearingTransform(img2,Ha_prime,Hs_prime);

    cv::Mat H=Hs*Ha;
    cv::Mat H_prime=Hs_prime*Ha_prime;
    refineH(img1,img2,H,H_prime);

    cv::Mat img1_H;
    cv::Mat img2_H;

    warpPerspective( img1, img1_H, H, getSize(img1,H));
    warpPerspective( img2, img2_H, H_prime, getSize(img2,H_prime));

    drawMatches(img1_H,img2_H,p1,p2,H,H_prime);
    cv::waitKey(0);

    return 0;
}
