//
// Created by kehan on 2019/9/30.
//


#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;


void find_feature_matches(cv::Mat &_img_1,
                          cv::Mat &_img_2,
                          vector<cv::KeyPoint> &_keypoints_1,
                          vector<cv::KeyPoint> &_keypoints_2,
                          vector<cv::DMatch> &_matches);

void pose_estimation_2d2d(
        std::vector<cv::KeyPoint> keypoints_1,
        std::vector<cv::KeyPoint> keypoints_2,
        std::vector<cv::DMatch> matches,
        cv::Mat &R, cv::Mat &t);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);



int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }


    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "Match cnt: " << matches.size() << endl;

    cv::Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);


    // E=t^R*scale
    cv::Mat t_x =
            (cv::Mat_<double>(3, 3) <<
                                    0, -t.at<double>(2, 0),
                    t.at<double>(1, 0),
                    t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                    -t.at<double>(1.0), t.at<double>(0, 0), 0);
    cout << "t^R=" << endl << t_x * R << endl;


    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (cv::DMatch m: matches)
    {
        cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        cv::Mat d = y2.t() * t_x * R * y1;

        cout << "epipolar constraint = " << d << endl;

    }

    cv::waitKey(0);
    return 0;
}


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

