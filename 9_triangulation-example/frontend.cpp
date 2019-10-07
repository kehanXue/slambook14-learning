//
// Created by kehan on 2019/10/7.
//

#include "frontend.h"


cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K)
{
    return cv::Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}


void find_feature_matches(cv::Mat &_img_1,
                          cv::Mat &_img_2,
                          vector<cv::KeyPoint> &_keypoints_1,
                          vector<cv::KeyPoint> &_keypoints_2,
                          vector<cv::DMatch> &_matches)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    orb->detect(_img_1, _keypoints_1);
    orb->detect(_img_2, _keypoints_2);

    cv::Mat descriptors_1, descriptors_2;
    orb->compute(_img_1, _keypoints_1, descriptors_1);
    orb->compute(_img_2, _keypoints_2, descriptors_2);

    cv::Mat outimg_1;
    cv::drawKeypoints(_img_1, _keypoints_1, outimg_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB Features", outimg_1);

    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, _matches);

    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = _matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (_matches[i].distance <= max(2 * min_dist, 30.0))
        {
            good_matches.push_back(_matches[i]);
        }
    }

    cv::Mat img_match;
    cv::Mat img_goodmatch;
    drawMatches(_img_1, _keypoints_1, _img_2, _keypoints_2, _matches, img_match);
    _matches = good_matches;
    drawMatches(_img_1, _keypoints_1, _img_2, _keypoints_2, good_matches, img_goodmatch);
    imshow("All matches", img_match);
    imshow("Good matches", img_goodmatch);
}


void pose_estimation_2d2d(
        std::vector<cv::KeyPoint> keypoints_1,
        std::vector<cv::KeyPoint> keypoints_2,
        std::vector<cv::DMatch> matches,
        cv::Mat &R, cv::Mat &t)
{
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;

    for (auto &match : matches)
    {
        points1.push_back(keypoints_1[match.queryIdx].pt);
        points2.push_back(keypoints_2[match.trainIdx].pt);
    }

    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;


    cv::Point2d principal_point(325.1, 249.7);
    int focal_length = 521;
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point, cv::RANSAC);
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3, cv::noArray(), 2000, 0.99);
    cout << "homography_matrix is " << endl << homography_matrix << endl;

    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;

}


void triangulation(const vector<cv::KeyPoint> &keypoint_1,
                   const vector<cv::KeyPoint> &keypoint_2,
                   const std::vector<cv::DMatch> &matches,
                   const cv::Mat &R,
                   const cv::Mat &t,
                   vector<cv::Point3d> &points)
{
    cv::Mat T1 =
            (cv::Mat_<double>(3, 4) <<
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0);

    cv::Mat T2 =
            (cv::Mat_<double>(3, 4) <<
                    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<cv::Point2d> pts_1, pts_2;
    for (cv::DMatch m:matches)
    {
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for (int i = 0; i < pts_4d.cols; i++)
    {
        cv::Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0);
        cv::Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0));

        points.push_back(p);
    }

}

