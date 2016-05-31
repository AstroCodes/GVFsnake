#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


//RNG rng(12345);
void find_moments( Mat output );

int main( )
{
    /// Load source image, convert it to gray and blur it
    Mat src, gray;;
    src = imread("U64_res.png", 1 );

   // cvtColor( src, gray, CV_BGR2GRAY );
   // blur( gray, gray, Size(3, 3) );

    Mat output;
    cv::inRange(src, cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0), output);

    cv::imshow("output", output);


    namedWindow( "Source", CV_WINDOW_AUTOSIZE );
    imshow( "Source", src );


    find_moments( output );

    waitKey(0);
    return(0);
}


void find_moments( Mat output)
{
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using canny
    Canny( output, canny_output, 100, 150, 3 );

    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Get the moments
    vector<Moments> mu(contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    {
        mu[i] = moments( contours[i], false );
    }

    ///  Get the mass centers:
    vector<Point2f> mc( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    {
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
    }


    /// Draw contours
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
        //Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, Scalar(255, 0, 0), 2, 8, hierarchy, 0, Point() );
        circle( drawing, mc[i], 4, Scalar(0, 0, 255), -1, 8, 0 );
    }

    /// Show in a window
    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );

    /// Calculate the area with the moments 00 and compare with the result of the OpenCV function
    printf("\t Info: Area and Contour Length \n");
    for( int i = 0; i< contours.size(); i++ )
        printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", i, mu[i].m00, contourArea(contours[i]), arcLength( contours[i], true ) );

}
