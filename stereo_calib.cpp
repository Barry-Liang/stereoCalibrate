/* This is sample from the OpenCV book. The copyright notice is below */

/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warranty, support or any guarantee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008

   AVAILABLE AT:
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130

   OPENCV WEBSITES:
     Homepage:      http://opencv.org
     Online docs:   http://docs.opencv.org
     Q&A forum:     http://answers.opencv.org
     GitHub:        https://github.com/opencv/opencv/
   ************************************************** */

#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

static int print_help()
{
    cout <<
            " Given a list of chessboard images, the number of corners (nx, ny)\n"
            " on the chessboards, and a flag: useCalibrated for \n"
            "   calibrated (0) or\n"
            "   uncalibrated \n"
            "     (1: use stereoCalibrate(), 2: compute fundamental\n"
            "         matrix separately) stereo. \n"
            " Calibrate the cameras and display the\n"
            " rectified results along with the computed disparity images.   \n" << endl;
    cout << "Usage:\n ./stereo_calib -w=<board_width default=9> -h=<board_height default=6> -s=<square_size default=1.0> <image list XML/YML file default=stereo_calib.xml>\n" << endl;
    return 0;
}

Mat matRotateClockWise180(Mat src)//顺时针180
{
    if (src.empty())
    {
        cout<< "RorateMat src is empty!";
    }
    flip(src, src, 0);// 翻转模式，flipCode == 0垂直翻转（沿X轴翻转），flipCode>0水平翻转（沿Y轴翻转），flipCode<0水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
    flip(src, src, 1);
    return src;
    //transpose(src, src);// 矩阵转置
}


static void
StereoCalib(bool isRGB, const vector<string>& imagelist, Size boardSize, float squareSize, bool displayCorners = false, bool useCalibrated=true, bool showRectified=true, const string& mode="ASYMMETRIC_CIRCLES_GRID")
{
    if( imagelist.size() % 2 != 0 )
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    const int maxScale = 2;
    // ARRAY AND VECTOR STORAGE:

    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size()/2;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;

    for( i = j = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            const string& filename = imagelist[i*2+k];
            Mat img = imread(filename, 0);
           // img=matRotateClockWise180(img);
            if(img.empty())
                break;
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale, INTER_LINEAR_EXACT);
                if (mode=="ASYMMETRIC_CIRCLES_GRID")
                    found = findCirclesGrid( timg, boardSize, corners, CALIB_CB_ASYMMETRIC_GRID );
                else
                    found = findChessboardCorners(timg, boardSize, corners,
                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

                if( found )
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }
            if( displayCorners )
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR_EXACT);
                imshow("corners", cimg1);
                char c = (char)waitKey(500);
                if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if( !found )
                break;
            if (mode=="CHESSBOARD")
            cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                      30, 0.01));
        }
        if( k == 2 )
        {
            goodImageList.push_back(imagelist[i*2]);
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if( nimages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);
    cout<<sizeof(imagePoints[0])<<endl;
    cout<<sizeof(imagePoints[0][1])<<endl;
    if (mode=="ASYMMETRIC_CIRCLES_GRID") {
        for (i = 0; i < nimages; i++) {
            for (j = 0; j < boardSize.height; j++)
                for (k = 0; k < boardSize.width; k++)
                    if (j%2==0)
                    {objectPoints[i].push_back(Point3f(j * squareSize, 2*k * squareSize, 0));}
            else{objectPoints[i].push_back(Point3f(j * squareSize, (2*k+1) * squareSize, 0));}

        }
    } else{
        for (i = 0; i < nimages; i++) {
            for (j = 0; j < boardSize.height; j++)
                for (k = 0; k < boardSize.width; k++)
                    objectPoints[i].push_back(Point3f(k * squareSize, j * squareSize, 0));}
    }
    cout << "Running stereo calibration ...\n";

    Mat cameraMatrix[2], distCoeffs[2];
    //cameraMatrix[0] = initCameraMatrix2D(objectPoints,imagePoints[0],imageSize,0);
    //cameraMatrix[1] = initCameraMatrix2D(objectPoints,imagePoints[1],imageSize,0);
    Mat R, T, E, F;
    Mat c1,c2,d1,d2;
    cv::FileStorage fs;
    fs.open("/home/liangxiao/XIMEA/package/samples/xiAPIplusOpenCV/seperate_singleCalibrate/out_camera_data_left.xml", cv::FileStorage::READ);
    fs ["camera_matrix_left"] >> c1;
    fs ["distortion_coefficients_left"] >> d1;
    fs.release();
    fs.open("/home/liangxiao/XIMEA/package/samples/xiAPIplusOpenCV/seperate_singleCalibrate/out_camera_data_right.xml", cv::FileStorage::READ);
    fs ["camera_matrix_right"] >> c2;
    fs ["distortion_coefficients_right"] >> d2;
    fs.release();
    cameraMatrix[0]=c1;
    cameraMatrix[1]=c2;
    distCoeffs[0]=d1;
    distCoeffs[1]=d2;

    //CALIB_USE_INTRINSIC_GUESS//CALIB_FIX_INTRINSIC     CALIB_SAME_FOCAL_LENGTH +

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,
                    CALIB_FIX_ASPECT_RATIO +
                    CALIB_ZERO_TANGENT_DIST +
                                         CALIB_USE_INTRINSIC_GUESS +
                    CALIB_SAME_FOCAL_LENGTH +
                    CALIB_RATIONAL_MODEL +
                    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    cout << "done with RMS error=" << rms << endl;

// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for( i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for( k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for( j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average epipolar err = " <<  err/npoints << endl;

    // save intrinsic parameters
     fs.open("/home/liangxiao/XIMEA/package/samples/xiAPIplusOpenCV/stereoCalibrate/intrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    fs.open("/home/liangxiao/XIMEA/package/samples/xiAPIplusOpenCV/stereoCalibrate/extrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

// COMPUTE AND DISPLAY RECTIFICATION
    if( !showRectified )
        return;

    Mat rmap[2][2];
// IF BY CALIBRATED (BOUGUET'S METHOD)
    if( useCalibrated )
    {
        // we already computed everything
    }
// OR ELSE HARTLEY'S METHOD
    else
 // use intrinsic parameters of each camera, but
 // compute the rectification transformation directly
 // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for( k = 0; k < 2; k++ )
        {
            for( i = 0; i < nimages; i++ )
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;
    double sf;
    int w, h;
    if( !isVerticalStereo )
    {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else
    {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }
    fs.open("/home/liangxiao/XIMEA/package/samples/xiAPIplusOpenCV/stereoCalibrate/final_data.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
       fs << "rmap00" << rmap[0][0]<<"rmap01" << rmap[0][1]<<"rmap10" << rmap[1][0]<<"rmap11" << rmap[1][1]<< "canvas" << canvas;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    for( i = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            Mat img = imread(goodImageList[i*2+k], 0), rimg, cimg;
           // img=matRotateClockWise180(img);
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            if( useCalibrated )
            {
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
            }
        }

        if( !isVerticalStereo )
            for( j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for( j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
        char c = (char)waitKey();
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }
    Mat canvasplus;
    sf = 600./MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width*sf);
    h = cvRound(imageSize.height*sf);
    canvasplus.create(h, w*2, CV_8UC3);
    Mat img11 = imread("/home/liangxiao/XIMEA/package/samples/xiAPIplusOpenCV/seperate_singleCalibrate/left01.jpg", 0);
    img11=matRotateClockWise180(img11);
    Mat img22 = imread("/home/liangxiao/XIMEA/package/samples/xiAPIplusOpenCV/seperate_singleCalibrate/right01.jpg", 0);
    img22=matRotateClockWise180(img22);
    Mat rimg11,rimg22,cimg11,cimg22;
    remap(img11, rimg11, rmap[0][0], rmap[0][1], INTER_LINEAR);
    remap(img22, rimg22, rmap[1][0], rmap[1][1], INTER_LINEAR);
    cvtColor(rimg11, cimg11, COLOR_GRAY2BGR);
    cvtColor(rimg22, cimg22, COLOR_GRAY2BGR);
    Mat canvasPart1 = canvasplus(Rect(w*0, 0, w, h)) ;
    Mat canvasPart2 = canvasplus(Rect(w*1, 0, w, h)) ;
    resize(cimg11, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
    resize(cimg22, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
    Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
              cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));
    rectangle(canvasPart1, vroi1, Scalar(0,0,255), 3, 8);
    Rect vroi2(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
              cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));
    rectangle(canvasPart2, vroi2, Scalar(0,0,255), 3, 8);
    for( j = 0; j < canvasplus.rows; j += 16 )
        line(canvasplus, Point(0, j), Point(canvasplus.cols, j), Scalar(0, 255, 0), 1, 8);
    while(1) {
        imshow("example_left", canvasplus);
        char c = (char) waitKey();
        if (c == 27 || c == 'q' || c == 'Q')
        {}
    }

}


static bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}

int main(int argc, char** argv)
{
    Size boardSize;
    string imagelistfn,mode;
    bool showRectified,isRGB;
    cv::CommandLineParser parser(argc, argv, "{RGB|0|}{w|6|}{h|9|}{s|24|}{nr||}{mode|CHESSBOARD|}{help||}{@input| /home/liangxiao/XIMEA/package/samples/xiAPIplusOpenCV/stereoCalibrate/stereo_calib_chessboard.xml|}");
    //cv::CommandLineParser parser(argc, argv, "{RGB|0|}{w|4|}{h|11|}{s|20|}{nr||}{mode|ASYMMETRIC_CIRCLES_GRID|}{help||}{@input| /home/liangxiao/XIMEA/package/samples/xiAPIplusOpenCV/stereoCalibrate/stereo_calib_circle.xml|}");

    if (parser.has("help"))
        return print_help();
    showRectified = !parser.has("nr");
    //imagelistfn = cv::samples::findFile(parser.get<string>("@input"));  // if this command has a problem, just assign the path to the imagelistfn.
    imagelistfn=parser.get<string>("@input");
    boardSize.width = parser.get<int>("w");
    boardSize.height = parser.get<int>("h");
    float squareSize = parser.get<float>("s");
    mode =parser.get<string>("mode");
    isRGB =parser.get<bool>("RGB");

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    vector<string> imagelist;
    bool ok = readStringList(imagelistfn, imagelist);
    if(!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        return print_help();
    }
    StereoCalib(isRGB,imagelist, boardSize, squareSize, true, true, showRectified,mode);
    return 0;
}
