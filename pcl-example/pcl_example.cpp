//
// Created by kehan on 2019/9/29.
//

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Geometry>
#include <boost/format.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>


int main(int argc, char** argv)
{
    vector<cv::Mat> color_imgs, depth_imgs;
    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses;

    ifstream fin("./pose.txt");
    if (!fin)
    {
        cerr << "Please run this exe in the fold contains a pose.txt" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++)
    {
        boost::format fmt("./%s/%d.%s");
        color_imgs.emplace_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depth_imgs.emplace_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1));

        double data[7] = {0};
        for (auto &d:data)
        {
            fin >> d;
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.emplace_back(T);
    }

    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;

    cout << "Start convert image to pointscloud" << endl;

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloud::Ptr point_cloud(new PointCloud);
    for (int i = 0; i < 5; i++)
    {
        cout << "Converting... " << i << endl;
        cv::Mat color = color_imgs.at(i);
        cv::Mat depth = depth_imgs.at(i);

        Eigen::Isometry3d T = poses.at(i);

        for (int v = 0; v < color.rows; v++)
        {
            for (int u = 0; u < color.cols; u++)
            {
                unsigned int per_depth = depth.ptr<unsigned short>(v)[u];
                if (per_depth == 0)
                {
                    continue;
                }
                Eigen::Vector3d point;
                point[2] = double(per_depth) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d point_world = T * point;

                PointT p;
                p.x = point_world[0];
                p.y = point_world[1];
                p.z = point_world[2];
                p.b = color.data[v * color.step + u * color.channels()];
                p.g = color.data[v * color.step + u * color.channels() + 1];
                p.r = color.data[v * color.step + v * color.channels() + 2];
                point_cloud->points.emplace_back(p);
            }
        }
    }

    point_cloud->is_dense = false;
    cout << "The pointscloud has the points number: " << point_cloud->size() << endl;
    pcl::io::savePCDFileBinary("map.pcd", *point_cloud);

    return 0;
}