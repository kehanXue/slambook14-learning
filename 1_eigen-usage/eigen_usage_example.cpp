//
// Created by kehan on 2019/9/24.
//

#include <iostream>
#include <ctime>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>


#define MATRIX_SIZE 50


int main(int argc, char** argv)
{

    /*
     * Basic usages
     */
    Eigen::Matrix<float, 2, 3> matrix_23;

    Eigen::Vector3d vector_3d;
    Eigen::Matrix<float, 3, 1> vector_3f;

    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
    Eigen::MatrixXd matrix_xd;



    matrix_23 << 1, 2, 3, 4, 5, 6;
    std::cout << "matrix_23" << std::endl
              << matrix_23 << std::endl << std::endl;

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            std::cout << matrix_23(i, j) << " ";
        }
        std::cout << std::endl;
    }


    vector_3d << 3, 2, 1;
    vector_3f << 4, 5, 6;
    Eigen::Matrix<double, 2, 1> result_d = matrix_23.cast<double>() * vector_3d;
    std::cout << "result_d: " << result_d << std::endl;

    Eigen::Matrix<float, 2, 1> result_f = matrix_23 * vector_3f;
    std::cout << "result_f: " << result_f << std::endl;

    matrix_33 = Eigen::Matrix3d::Random();
    std::cout << matrix_33 << std::endl << std::endl;


    std::cout << matrix_33.transpose() << std::endl;
    std::cout << matrix_33.sum() << std::endl;
    std::cout << matrix_33.trace() << std::endl;
    std::cout << 10 * matrix_33 << std::endl;
    std::cout << matrix_33.inverse() << std::endl;
    std::cout << matrix_33.determinant() << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    std::cout << "Eigen values: " << eigen_solver.eigenvalues() << std::endl;
    std::cout << "Eigen vector: " << eigen_solver.eigenvectors() << std::endl;



    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::Matrix<double, MATRIX_SIZE, 1> vector_Nd;
    vector_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock();
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * vector_Nd;
    std::cout << "Time use in normal inverse is: " << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms"
              << std::endl;

    // Qr slover
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(vector_Nd);
    std::cout << "Time use in Qr decomposition is: " << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms"
              << std::endl;



    /*
     * Geometry model
     */
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    // std::cout << rotation_matrix << std::endl;
    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));
    std::cout.precision(3);
    std::cout << "rotation matrix: \n" << rotation_vector.matrix() << std::endl;

    rotation_matrix = rotation_vector.toRotationMatrix();

    Eigen::Vector3d vector_3d_to_be_rotation(1, 0, 0);
    Eigen::Vector3d vector_3d_rotated = rotation_vector * vector_3d_to_be_rotation;
    std::cout << "(1, 0, 0) after rotation: \n" << vector_3d_rotated << std::endl;
    vector_3d_rotated = rotation_matrix * vector_3d_to_be_rotation;
    std::cout << "(1, 0, 0) after rotation: \n" << vector_3d_rotated << std::endl;


    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); //TODO
    std::cout << "yaw, pitch, roll: \n" << euler_angles.transpose() << std::endl;


    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(rotation_vector);
    T.pretranslate(Eigen::Vector3d(1, 3, 4));
    std::cout << "Transform matrix: \n" << T.matrix() << std::endl;

    Eigen::Vector3d vector_3d_transformed = T * vector_3d_to_be_rotation;
    std::cout << "(1, 0, 0) transform: \n" << vector_3d_transformed.transpose() << std::endl;

    // Eigen::Affine3d, Eigen::Projective3d


    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    std::cout << "quaterniond: \n" << q.coeffs() << std::endl;

    q = Eigen::Quaterniond(rotation_matrix);
    std::cout << "quaterniond: \n" << q.coeffs() << std::endl;

    vector_3d_rotated = q * vector_3d_to_be_rotation;

    std::cout << "(1, 0, 0) after quat rotation: " << vector_3d_rotated.transpose() << std::endl;

    return 0;
}
