/*
 * main.cpp
    Copyright (C) 2013  Timothy Sweet

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <iomanip>

#include <opencv2/opencv.hpp>

void removeOutliers(const cv::Mat magnitude, cv::Mat* real, cv::Mat* imaginary);
void getMagnitude(cv::Mat planes[], cv::Mat* magnitude);
void rearrangeQuadrants(cv::Mat* magnitude);
cv::Mat multiplyInTimeDomain(const cv::Mat& image, const cv::Mat& mask);

int main(int argc, char* argv[]) {
	//if OpenGL is enabled open an OpenGL window, otherwise just a regular window
	cv::namedWindow("magnitude",cv::WINDOW_AUTOSIZE);
	if(argc != 2) {
		std::cout<<"Usage: ./part1.cpp image"<<std::endl;
		return 0;
	}

	cv::Mat input_image = cv::imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);

	if(input_image.empty()) {
		char appended[100];
		appended[0] = '\0';
		strcat(appended,"../");
		strcat(appended,argv[1]);
		cv::Mat input_image = cv::imread(appended,CV_LOAD_IMAGE_GRAYSCALE);
		if(input_image.empty()) {
			std::cout<<argv[1]<<" was invalid"<<std::endl;
			std::cout<<" also tried: "<<appended<<std::endl;
			return 0;
		}
	}

	//the image dimensions must be a power of two
	cv::Mat padded_image;
	cv::Size padded_size(
			cv::getOptimalDFTSize(input_image.cols),
			cv::getOptimalDFTSize(input_image.rows));

	//pad the input image
	cv::copyMakeBorder(input_image, //input image
			padded_image, //output image
			0, //pad the top with..
			padded_size.height-input_image.rows, //pad the bottom with...
			0, //pad the left with...
			padded_size.width-input_image.cols, //pad the right with...
			cv::BORDER_CONSTANT, //make the border constant (as opposed to a copy of the data
			cv::Scalar::all(0)); //make the border black

	/*
	 * The DFT function expects a two-channel image, so let's make two planes
	 * and then merge them
	 */

	cv::Mat planes[] = {cv::Mat_<float>(padded_image),cv::Mat::zeros(padded_size,CV_32F)};

	//now make a single complex (two-channel) image
	cv::Mat complex_image;
	cv::merge(planes,2,complex_image);

	//get the dft of the image
	cv::dft(complex_image,complex_image);

	//generate the first mask
	cv::Mat first_mask(padded_size,complex_image.type(),cv::Scalar::all(0));

	//get roi in the center of the mask
	cv::Rect first_roi(first_mask.cols/2-1,first_mask.rows/2-1,3,3);
	std::cout<<"roi: "<<first_roi<<std::endl;
	cv::Mat first_mask_center(first_mask,first_roi);
	first_mask_center.at<cv::Vec2f>(0,0)[0] = -1;
	first_mask_center.at<cv::Vec2f>(1,0)[0] = -2;
	first_mask_center.at<cv::Vec2f>(2,0)[0] = -1;
	first_mask_center.at<cv::Vec2f>(0,2)[0] = 1;
	first_mask_center.at<cv::Vec2f>(1,2)[0] = 2;
	first_mask_center.at<cv::Vec2f>(2,2)[0] = 1;

	//transform the mask to the fourier domain
	cv::dft(first_mask,first_mask);

	//filter the image
	cv::Mat first_filtered_image = multiplyInTimeDomain(complex_image,first_mask);

	//generate the second mask
	cv::Mat second_mask(padded_size,complex_image.type(),cv::Scalar::all(0));

	//get roi in the center of the mask
	cv::Rect second_roi(second_mask.cols/2-1,second_mask.rows/2-1,3,3);
	std::cout<<"roi: "<<second_roi<<std::endl;
	cv::Mat second_mask_center(second_mask,second_roi);
	second_mask_center.at<cv::Vec2f>(0,0)[0] = -1;
	second_mask_center.at<cv::Vec2f>(0,1)[0] = -2;
	second_mask_center.at<cv::Vec2f>(0,2)[0] = -1;
	second_mask_center.at<cv::Vec2f>(2,0)[0] = 1;
	second_mask_center.at<cv::Vec2f>(2,1)[0] = 2;
	second_mask_center.at<cv::Vec2f>(2,2)[0] = 1;
	std::cout<<second_mask_center<<std::endl;

	//transform the mask to the fourier domain
	cv::dft(second_mask,second_mask);

	//filter the image
	cv::Mat second_filtered_image = multiplyInTimeDomain(complex_image,second_mask);

	//convert filtered image back to spatial domain
    cv::dft(second_filtered_image,second_filtered_image,cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    cv::dft(first_filtered_image,first_filtered_image,cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

	rearrangeQuadrants(&second_filtered_image);
	rearrangeQuadrants(&first_filtered_image);

    //normalize(first_filtered_image, first_filtered_image, 0, 1, CV_MINMAX);
    //normalize(second_filtered_image, second_filtered_image, 0, 1, CV_MINMAX);

    //compute magnitude

    cv::Mat result(first_filtered_image.rows,first_filtered_image.cols,CV_32FC1,cv::Scalar::all(0));
	for(int y=0;y<result.rows;y++) {
		for(int x=0;x<result.cols;x++) {
			float first = first_filtered_image.at<float>(y,x);
			float second = second_filtered_image.at<float>(y,x);
			result.at<float>(y,x) = sqrt(first*first+second*second);
		}
	}

    normalize(result, result, 0, 1, CV_MINMAX);
    cv::imshow("first filtered image",first_filtered_image);
    cv::imshow("second filtered image",second_filtered_image);
    cv::imshow("result",result);


    //show opencv sobel
    cv::Sobel(input_image,input_image,-1,1,0);
    imshow("opencv sobel",input_image);
   cv::waitKey(0);
    return 0;
}

void removeOutliers(const cv::Mat magnitude, cv::Mat* real, cv::Mat* imaginary) {
	//get pointer to the upper left +one row and +one col
	//so it doesn't process the border

	for(int y=1;y<magnitude.rows-2;y++) {
		for(int x=1;x<magnitude.cols-2;x++) {
			//get 4-neighbor max
			float left = magnitude.at<float>(y,x-1);
			float right = magnitude.at<float>(y,x+1);
			float up = magnitude.at<float>(y-1,x);
			float down = magnitude.at<float>(y+1,x);
			float center = magnitude.at<float>(y,x);

			//if the current pixel is greater than all its neighbors by more than
			//a factor of factor
			double factor = 1.4;
			if(center > left*factor && center > right*factor && center > up*factor && center > down*factor) {
				std::cout<<"remove "<<x<<","<<y<<std::endl;
				std::cout<<"left: "<<left<<" right: "<<right<<" up: "<<up<<" down: "<<down<<" center: "<<center<<std::endl;
				std::cout<<"real: "<<real->at<float>(y,x)<<" imaginary: "<<imaginary->at<float>(y,x)<<std::endl;
				real->at<float>(y,x) = 0;
				imaginary->at<float>(y,x) = 0;

				std::cout<<"2real: "<<real->at<float>(y,x)<<" imaginary: "<<imaginary->at<float>(y,x)<<std::endl;
			}
		}
	}
}

void getMagnitude(cv::Mat planes[], cv::Mat* magnitude) {
	cv::magnitude(planes[0],planes[1],*magnitude);

	//switch to logrithmic scale
	*magnitude += cv::Scalar::all(1);
	cv::log(*magnitude,*magnitude);

    /*
     * Move each pixel to between 0 and 1 since it's a float image
     */
    normalize(*magnitude,*magnitude, 0, 1, CV_MINMAX);
}
void rearrangeQuadrants(cv::Mat* magnitude) {
	// rearrange the image so that the bright stuff is in the middle
	int center_x = magnitude->cols/2, center_y = magnitude->rows/2;

	//get a ROI for each quadrant
	cv::Mat q0(*magnitude, cv::Rect(0, 0, center_x, center_y));   // Top-Left
	cv::Mat q1(*magnitude, cv::Rect(center_x, 0, center_x, center_y));  // Top-Right
	cv::Mat q2(*magnitude, cv::Rect(0, center_y, center_x, center_y));  // Bottom-Left
	cv::Mat q3(*magnitude, cv::Rect(center_x, center_y, center_x, center_y)); // Bottom-Right

	//by rearragning these ROIs it modifies the original image
	// swap top left and bottom right
	cv::Mat temp;
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);

	//swap top right and bottom left
	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);
}

cv::Mat multiplyInTimeDomain(const cv::Mat& image, const cv::Mat& mask) {
	cv::Mat result(image.rows,image.cols,CV_32FC2,cv::Scalar::all(0));

	//multiply each element as a complex number
	for(int y=0;y<image.rows;y++) {
		for(int x=0;x<image.cols;x++) {
			result.at<cv::Vec2f>(y,x)[0] =
					image.at<cv::Vec2f>(y,x)[0]*mask.at<cv::Vec2f>(y,x)[0]
				  - image.at<cv::Vec2f>(y,x)[1]*mask.at<cv::Vec2f>(y,x)[1];
			result.at<cv::Vec2f>(y,x)[1] =
					image.at<cv::Vec2f>(y,x)[0]*mask.at<cv::Vec2f>(y,x)[1]
				  + image.at<cv::Vec2f>(y,x)[1]*mask.at<cv::Vec2f>(y,x)[0];

		}
	}


	return result;
}
