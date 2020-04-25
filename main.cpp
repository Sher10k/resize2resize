#include <iostream>
#include <string>
#include <vector>
#include <cmath>

// CV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Open3D
#include <Open3D/Open3D.h>
#include <Open3D/Geometry/Geometry.h>
//#include <Open3D/Geometry/Geometry3D.h>
//#include <Open3D/Geometry/PointCloud.h>
//#include <Open3D/Geometry/Octree.h>
//#include <Open3D/Visualization/Visualizer/Visualizer.h>

using namespace std;
using namespace cv;
using namespace open3d;
using namespace Eigen;

// type of interpolation
#define STRINGIFY(x) #x
#define STRINGIFYMACRO(y) STRINGIFY(y)
#define INTER_TYPE INTER_LINEAR
                    /// 1 INTER_NEAREST
                    /// 2 INTER_LINEAR
                    /// 3 INTER_CUBIC
                    /// 4 INTER_AREA
                    /// 5 INTER_LANCZOS4
                    /// 6 INTER_LINEAR_EXACT

#define RESIZE_COEF 8
#define L_SIZE 1
#define L_VIEW_SIZE 1
#define VIS_3D 0


void sobel( const Mat &input_img, Mat &output_img_X, Mat &output_img_Y )
{
	output_img_X = Mat::zeros( input_img.size(), CV_16S );
    output_img_Y = Mat::zeros( input_img.size(), CV_16S );
	int Fk[3][3] = { { 1,1,1 },
                     { 1,1,1 },
                     { 1,1,1 } }; // маска фильтра 3*3
	for (int i = 1; i < input_img.cols - 1; i++)
    {
		for (int j = 1; j < input_img.rows - 1; j++) 
        {
			for (int ii = -1; ii <= 1; ii++)
			{
                for (int jj = -1; jj <= 1; jj++) 
                {
					//uchar blurred = input_img.at<uchar>(j + jj, i + ii);
					Fk[ii + 1][jj + 1] = input_img.at< uchar >(j + jj, i + ii);
				}
            }
            output_img_X.at< short >(j, i) = short( (Fk[0][0] + 2 * Fk[0][1] + Fk[0][2]) - 
                                                    (Fk[2][0] + 2 * Fk[2][1] + Fk[2][2]) );
            output_img_Y.at< short >(j, i) = short( (Fk[0][0] + 2 * Fk[1][0] + Fk[2][0]) - 
                                                    (Fk[0][2] + 2 * Fk[1][2] + Fk[2][2]) );
		}
    }
}

int main()  // int argc, char *argv[]
{
    //std::cout << cv::getBuildInformation() << std::endl;
    
    
    string img_folder = "../img_train/";
                        //"../8_bit/";
    
    string img_name = "img_train_origin.jpg";
                      //"8_bit_input_img.png";
    
    // --- Version 2.0
    {
            // --- Load image
        Mat img_gray = imread( img_folder + img_name, IMREAD_GRAYSCALE );
        img_gray = img_gray / L_VIEW_SIZE;
        
            // --- Resize by interpolation
        Mat img_g_res;  // = Mat::zeros( img_gray.size(), CV_8U );
        cout << STRINGIFYMACRO(INTER_TYPE) << endl;
        resize( img_gray, img_g_res, img_gray.size() / RESIZE_COEF, 0, 0, cv::INTER_NEAREST );
        //img_g_res = img_g_res / L_VIEW_SIZE;
        
//        cout << "img_gray_resize: \n";
//        for ( int y = 0; y < img_g_res.rows; y++ )
//        {
//            for ( int x = 0; x < img_g_res.cols; x++ )
//            {
//                cout << +img_g_res.at< uint8_t >(y, x) << "  \t";
//            }
//            cout << "\n";
//        }
//        cout << "\n";
        
            // --- Sobel
        Mat img_sobel_X_1_8, img_sobel_Y_1_8;
        sobel( img_g_res, img_sobel_X_1_8, img_sobel_Y_1_8 );
//        cout << "img_sobel_X_resize: \n";
//        for ( int y = 0; y < img_sobel_X_1_8.rows; y++ )
//        {
//            for ( int x = 0; x < img_sobel_X_1_8.cols; x++ )
//            {
//                cout << +img_sobel_X_1_8.at< short >(y, x) << " \t";
//            }
//            cout << "\n";
//        }
//        cout << "\n";
//        cout << "img_sobel_Y_resize: \n";
//        for ( int y = 0; y < img_sobel_Y_1_8.rows; y++ )
//        {
//            for ( int x = 0; x < img_sobel_Y_1_8.cols; x++ )
//            {
//                cout << +img_sobel_Y_1_8.at< short >(y, x) << " \t";
//            }
//            cout << "\n";
//        }
//        cout << "\n";
        
        
            // --- Origin gray image
        //Mat img_gray;
        //resize( img_g_res, img_gray, img_g_res.size() * RESIZE_COEF, 0, 0, cv::INTER_NEAREST );
            // --- Save image_resize
        vector< int > compression_params;
        compression_params.push_back( IMWRITE_JPEG_QUALITY );
        compression_params.push_back( 100 );
        //imwrite( "../8_bit/8_bit_input_img.png", img_gray * L_VIEW_SIZE, compression_params );
        imwrite( img_folder + "input_img_resize.png", img_g_res * L_VIEW_SIZE, compression_params );
        //imshow( "img_gray_size/8", img*L_VIEW_SIZE );
        
        
            // --- --- Main algorithm --- --- //
        vector< Mat > img_gs_m;
        for ( size_t k = 0; k < 256; k++ )
            img_gs_m.push_back( Mat::zeros( img_g_res.size(), CV_64F ) );
        
            // --- Cutting into levels
        float progress = 0.0f;
        float step = 1.0f / img_g_res.total();
        int barWidth = 70;
        progress += step;
        cout << " --- Start cutting the image into levels" << endl;
        for ( int y = 0; y < img_g_res.rows; y++ )
            for ( int x = 0; x < img_g_res.cols; x++ )
            {
                img_gs_m.at( img_g_res.at< uint8_t >(y,x) ).at< double >(y,x) = 1.0;
                
                    // progress
                progress += step;
                cout << "[";
                int pos = int( float(barWidth) * progress );
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) cout << "=";
                    else if (i == pos) cout << ">";
                    else cout << " ";
                }
                cout << "] " << int(progress * 100.0f) << " %\r";
                cout.flush();
            }
        cout << "\n\n";
        
            // --- Save all layers
        for ( size_t k = 0; k < img_gs_m.size(); k++ )
        {
            Mat temp;
            img_gs_m.at(k) *= 255.0;
            img_gs_m.at(k).convertTo( temp, CV_8U );
            img_gs_m.at(k) /= 255.0;
            imwrite( img_folder + "img_layers_resize/img_gs_resize_layer_" + to_string(k) + ".png", 
                     temp * 255, 
                     compression_params );
        }
        
        // --- Save 0 level
        resize( img_gs_m.at(0), img_gs_m.at(0), img_gray.size(), 0, 0, INTER_TYPE );
        Mat temp;
        img_gs_m.at(0) *= 255.0;
        img_gs_m.at(0).convertTo( temp, CV_8U );
        img_gs_m.at(0) /= 255.0;
        imwrite( img_folder + "img_layers/img_gs_layer_" + to_string(0) + ".png", 
                 temp * 255, 
                 compression_params );
        
            // --- Resize all layers to the size of the original image & merge them
        Mat output_gray = Mat::zeros( img_gray.size(), CV_8U );
        progress = 0.0f;
        step = 1.0f / 256;
        barWidth = 70;
        progress += step;
        resize( img_sobel_X_1_8, img_sobel_X_1_8, img_gray.size(), 0, 0, INTER_TYPE );
        resize( img_sobel_Y_1_8, img_sobel_Y_1_8, img_gray.size(), 0, 0, INTER_TYPE );
        cout << " --- Start resize all layers to the size of the original image & merge them" << endl;
        for ( size_t k = 1; k < img_gs_m.size(); k++ )
        {
            resize( img_gs_m.at(k), img_gs_m.at(k), output_gray.size(), 0, 0, INTER_TYPE ); // INTER_NEAREST INTER_LINEAR INTER_CUBIC inter_type(2)
            Mat temp;
            img_gs_m.at(k) *= 255.0;
            img_gs_m.at(k).convertTo( temp, CV_8U );
            img_gs_m.at(k) /= 255.0;
            imwrite( img_folder + "img_layers/img_gs_layer_" + to_string(k) + ".png", 
                     temp * 255, 
                     compression_params );
            
//            for ( int y = 0; y < img_gs_m.at(k).rows; y++ )
//            {
//                for ( int x = 0; x < img_gs_m.at(k).cols; x++ )
//                {
//                    cout << img_gs_m.at(k).at< double >(y, x) << " ";
//                }
//                cout << "\n";
//            }
//            cout << "\n";
            
            for ( int y = 0; y < output_gray.rows; y++ )
                for ( int x = 0; x < output_gray.cols; x++ )
                {
                    if ( img_gs_m.at(k).at< double >(y,x) > 0.4 ) 
                    {
                        output_gray.at< uint8_t >(y,x) = uint8_t(k-1);
                        
                        if ( img_gs_m.at(k).at< double >(y,x) > 0.9 ) 
                        {
                            output_gray.at< uint8_t >(y,x) += 1;
                        }
                    }
                    
                }
                // Очищаем память иначе п*зд*ц
            img_gs_m.at(k).release();
            
                // progress
            progress += step;
            cout << "[";
            int pos = int( float(barWidth) * progress );
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) cout << "=";
                else if (i == pos) cout << ">";
                else cout << " ";
            }
            cout << "] " << int(progress * 100.0f) << " %\r";
            cout.flush();
        }
        cout << endl;
        
            // --- Save output image
        imwrite( img_folder + "output_img_" + STRINGIFYMACRO(INTER_TYPE) + ".png", output_gray * L_VIEW_SIZE, compression_params );
        
        
            // --- Convert input & output images to 3D point
        if (VIS_3D)
        {
            auto cloud_img_1_8 = make_shared< geometry::PointCloud >();
            for ( int y = 0; y < img_g_res.rows; y++ )
            {
                for ( int x = 0; x < img_g_res.cols; x++ )
                {
                    cloud_img_1_8->points_.push_back( Vector3d( x, 
                                                                y, 
                                                                img_g_res.at< uint8_t >(y, x) ) );
                    cloud_img_1_8->colors_.push_back( Vector3d( 0.0, // img_g_res.at< uint8_t >(y, x) * L_VIEW_SIZE / 255.0, 
                                                                0.0, 
                                                                1.0 ) );
                }
            }
            auto cloud_output_gray = make_shared< geometry::PointCloud >();
            for ( int y = 0; y < output_gray.rows; y++ )
            {
                for ( int x = 0; x < output_gray.cols; x++ )
                {
                    cloud_output_gray->points_.push_back( Vector3d( x, 
                                                                    y, 
                                                                    output_gray.at< uint8_t >(y, x) ) );
                    cloud_output_gray->colors_.push_back( Vector3d( 1.0, 
                                                                    0.0, 
                                                                    0.0 ) );
                }
            }
            auto cloud_input_gray = make_shared< geometry::PointCloud >();
            for ( int y = 0; y < img_gray.rows; y++ )
            {
                for ( int x = 0; x < img_gray.cols; x++ )
                {
                    cloud_input_gray->points_.push_back( Vector3d( x, 
                                                                   y, 
                                                                   img_gray.at< uint8_t >(y, x) ) );
                    cloud_input_gray->colors_.push_back( Vector3d( 0.0, 
                                                                   1.0, 
                                                                   0.0 ) );
                }
            }
            
                // --- Visualization
            visualization::Visualizer vis;
            vis.CreateVisualizerWindow( "Open3D", 1600, 900, 50, 50 );
                // --- Add Coordinate
            auto coord = geometry::TriangleMesh::CreateCoordinateFrame( 1.0, Vector3d( 0.0, 0.0, 0.0 ) );
            coord->ComputeVertexNormals();
            vis.AddGeometry( coord );
                // --- Add Point cloud
            vis.AddGeometry( cloud_img_1_8 );
            vis.AddGeometry( cloud_input_gray );
            vis.AddGeometry( cloud_output_gray );
                // --- Start visualization
            vis.Run();
        }
    }
    
    
    // --- Version 1.0
    {
//            // --- Load image
//        Mat img = imread( "../img_origin.JPG", IMREAD_COLOR );
//            // --- Conver to gray
//        Mat img_gray = Mat::zeros( img.size(), CV_8U );
//        cvtColor( img, img_gray, COLOR_BGR2GRAY );
//        //imshow( "img_gray", img_gray );
        
//            // --- Resize by interpolation
//        Mat img_g_res = Mat::zeros( img_gray.size(), CV_8U );
//        cout << STRINGIFYMACRO(INTER_TYPE) << endl;
//        resize( img_gray, img_g_res, img_gray.size() / RESIZE_COEF, 0, 0, INTER_TYPE );
//        //imshow( "img_gray_size/8", img_g_res );
        
//            // --- --- Main algorithm --- --- //
//        //vector< Mat > img_gs_m( 256, Mat::zeros( img_g_res.size(), CV_8U) ); --- error
//        vector< Mat > img_gs_m;
//        for ( size_t k = 0; k < 256; k++ )
//            img_gs_m.push_back( Mat::zeros( img_g_res.size(), CV_8U) );
            
//            // --- Cutting into levels
//        float progress = 0.0f;
//        float step = 1.0f / img_g_res.total();
//        int barWidth = 70;
//        progress += step;
//        cout << " --- Start cutting the image into levels" << endl;
//        for ( int y = 0; y < img_g_res.rows; y++ )
//            for ( int x = 0; x < img_g_res.cols; x++ )
//            {
//                img_gs_m.at( img_g_res.at< uint8_t >(y,x) ).at< uint8_t >(y,x) = 255;
                
//                    // progress
//                progress += step;
//                cout << "[";
//                int pos = int( float(barWidth) * progress );
//                for (int i = 0; i < barWidth; ++i) {
//                    if (i < pos) cout << "=";
//                    else if (i == pos) cout << ">";
//                    else cout << " ";
//                }
//                cout << "] " << int(progress * 100.0f) << " %\r";
//                cout.flush();
//            }
//        cout << endl;
//            // --- Save all layers
//        vector< int > compression_params;
//        compression_params.push_back( IMWRITE_JPEG_QUALITY );
//        compression_params.push_back( 100 );
//        for ( size_t k = 0; k < img_gs_m.size(); k++ )
//            imwrite( "../img_layers/img_gs_layer_" + to_string(k) + ".jpg", 
//                     img_gs_m.at(k), 
//                     compression_params );
        
//            // --- Resize all layers to the size of the original image & merge them
//        Mat output_gray = Mat::zeros( img_gray.size(), CV_8U );
//        progress = 0.0f;
//        step = 1.0f / 256;
//        barWidth = 70;
//        progress += step;
//        cout << " --- Start resize all layers to the size of the original image & merge them" << endl;
//        for ( size_t k = 0; k < img_gs_m.size(); k++ )
//        {
//            resize( img_gs_m.at(k), img_gs_m.at(k), output_gray.size(), 0, 0, INTER_TYPE );
//            for ( int y = 0; y < output_gray.rows; y++ )
//                for ( int x = 0; x < output_gray.cols; x++ )
//                    if ( img_gs_m.at( k ).at< uint8_t >(y,x) ) 
//                        output_gray.at< uint8_t >(y,x) = uint8_t(k);
//                // progress
//            progress += step;
//            cout << "[";
//            int pos = int( float(barWidth) * progress );
//            for (int i = 0; i < barWidth; ++i) {
//                if (i < pos) cout << "=";
//                else if (i == pos) cout << ">";
//                else cout << " ";
//            }
//            cout << "] " << int(progress * 100.0f) << " %\r";
//            cout.flush();
//        }
//        cout << endl;
//            // --- Save output image
//        imwrite( "../input_img.jpg", img_gray, compression_params );
//        imwrite( "../output_img.jpg", output_gray, compression_params );
        
//            // --- SKO 
//        double SKO = 0.0;
//        for ( int i = 0; i < int(output_gray.total()); i++ )
//        {
//            double delta = 1.0 * ( img_gray.at< uint8_t >(i) - output_gray.at< uint8_t >(i) );
//            SKO += delta * delta;
//        }
//        SKO /= output_gray.total();
//            // --- PSH
//        double PSH = 20.0 * log10( 255.0 / sqrt(SKO) );
//        cout << "SKO = " << SKO << endl;
//        cout << "PSH = " << PSH << endl;
    }
    
    
    waitKey(0);
    return 0;
}
