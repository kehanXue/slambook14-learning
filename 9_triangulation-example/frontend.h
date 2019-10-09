//
// Created by kehan on 2019/10/7.
//

#ifndef TRIANGULATION_EXAMPLE_FRONTEND_H_
#define TRIANGULATION_EXAMPLE_FRONTEND_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

void find_feature_matches(cv::Mat &_img_1,
                          cv::Mat &_img_2,
                          vector<cv::KeyPoint> &_keypoints_1,
                          vector<cv::KeyPoint> &_keypoints_2,
                          vector<cv::DMatch> &_matches);

void pose_estimation_2d2d(
        std::vector<cv::KeyPoint> &_keypoints_1,
        std::vector<cv::KeyPoint> &_keypoints_2,
        std::vector<cv::DMatch> &_matches,
        cv::Mat &_R, cv::Mat &_t);

void triangulation(
        const vector<cv::KeyPoint> &_keypoint_1,
        const vector<cv::KeyPoint> &_keypoint_2,
        const std::vector<cv::DMatch> &_matches,
        const cv::Mat &_R, const cv::Mat &_t,
        vector<cv::Point3d> &_points
);

#endif //TRIANGULATION_EXAMPLE_FRONTEND_H_
