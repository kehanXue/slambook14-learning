//
// Created by kehan on 2019/9/30.
//

#include "frontend.h"


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
    cout << "R: " << R << endl;
    cout << "t: " << t << endl;

    vector<cv::Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);


    // E=t^R*scale
    // cv::Mat t_x =
    //         (cv::Mat_<double>(3, 3) <<
    //                 0, -t.at<double>(2, 0),
    //                 t.at<double>(1, 0),
    //                 t.at<double>(2, 0), 0, -t.at<double>(0, 0),
    //                 -t.at<double>(1.0), t.at<double>(0, 0), 0);
    // cout << "t^R=" << endl << t_x * R << endl;
    //
    //
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    // for (cv::DMatch m: matches)
    // {
    //     cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    //     cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
    //     cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    //     cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
    //     cv::Mat d = y2.t() * t_x * R * y1;
    //
    //     cout << "epipolar constraint = " << d << endl;
    //
    // }
    for (int i = 0; i < matches.size(); i++)
    {
        cv::Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::Point2d pt1_cam_3d(
                points[i].x / points[i].z,
                points[i].y / points[i].z);

        cout << "point in the first camera frame: " << pt1_cam << endl;
        cout << "point projected from 3D " << pt1_cam_3d << ", d=" << points[i].z << endl;

        cv::Point2d pt2_cam = pixel2cam(keypoints_2[matches[i].trainIdx].pt, K);
        cv::Mat pt2_trans =
                R * (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        pt2_trans /= pt2_trans.at<double>(2, 0);

        cout<<"point in the second camera frame: "<<pt2_cam<<endl;
        cout<<"point reprojected from second frame: "<<pt2_trans.t()<<endl;
        cout<<endl;
    }

    cv::waitKey(0);
    return 0;
}


