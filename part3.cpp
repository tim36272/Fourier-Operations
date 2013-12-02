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
#include <complex>
#include <cmath>


void removeOutliers(const cv::Mat magnitude, cv::Mat* real, cv::Mat* imaginary);
void getMagnitude(cv::Mat planes[], cv::Mat* magnitude);
void rearrangeQuadrants(cv::Mat* magnitude);
cv::Mat multiplyInTimeDomain(const cv::Mat& image, const cv::Mat& mask);
cv::Mat divideInTimeDomain(const cv::Mat& image, const cv::Mat& mask);
cv::Mat invert(const cv::Mat& image);

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

	//generate a mask for creating motion blur
	cv::Mat motion_mask(complex_image.rows,complex_image.cols,CV_32FC2,cv::Scalar::all(0));
	for(int y=0;y<motion_mask.rows;y++) {
		for(int x=0;x<motion_mask.cols;x++) {
			double denom = 3.141592653*(x*0.005+y*0.005);
			std::complex<double> first(1./denom*std::sin(denom));
			std::complex<double> power(0,-denom);
			std::complex<double> second = std::exp(power);
			std::complex<double> result = first*second;
			if(std::abs(result.real()) > 0.1) {
				motion_mask.at<cv::Vec2f>(y,x)[0] = result.real();
				motion_mask.at<cv::Vec2f>(y,x)[1] = result.imag();
			}

		}
	}

	motion_mask.at<cv::Vec2f>(0,0)[0] = 0;
	motion_mask.at<cv::Vec2f>(0,0)[1] = 0;


	std::vector<cv::Mat> motion_planes;

	rearrangeQuadrants(&motion_mask);
	//rearrangeQuadrants(&complex_image);

	cv::split(motion_mask,motion_planes);

	cv::imshow("motion mask0",motion_planes[0]);
	cv::imshow("motion mask1",motion_planes[1]);


	cv::Mat image_with_motion = multiplyInTimeDomain(complex_image,motion_mask);
	//do the inversion stuff
	cv::Mat image_linear_unmotioned = divideInTimeDomain(image_with_motion,motion_mask);

	//do the weiner stuff
	cv::Mat image_weiner_unmotioned = divideInTimeDomain(image_with_motion,motion_mask);
	for(int y=0;y<image_weiner_unmotioned.rows;y++) {
		for(int x=0;x<image_weiner_unmotioned.cols;x++) {
			std::complex<double> val(image_weiner_unmotioned.at<cv::Vec2f>(y,x)[0],image_weiner_unmotioned.at<cv::Vec2f>(y,x)[1]);
			val = val*val;
			double abs_val = abs(val);
			image_weiner_unmotioned.at<cv::Vec2f>(y,x)[0] *=(abs_val*abs_val)/(abs_val*abs_val-1);
		}
	}

	cv::dft(image_with_motion,image_with_motion,cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
	cv::dft(image_linear_unmotioned,image_linear_unmotioned,cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
	cv::dft(image_weiner_unmotioned,image_weiner_unmotioned,cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

	//rearrangeQuadrants(&result);
    normalize(image_linear_unmotioned, image_linear_unmotioned, 0, 1, CV_MINMAX);
    normalize(image_with_motion, image_with_motion, 0, 1, CV_MINMAX);
    normalize(image_weiner_unmotioned, image_weiner_unmotioned, 0, 1, CV_MINMAX);


	//std::cout<<result<<std::endl;
	cv::imshow("image with motion",image_with_motion);
	cv::imshow("image unmotioned linear",image_linear_unmotioned);
	cv::imshow("image unmotioned weiner",image_weiner_unmotioned);

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

cv::Mat divideInTimeDomain(const cv::Mat& image, const cv::Mat& mask) {
	cv::Mat result(image.rows,image.cols,CV_32FC2,cv::Scalar::all(0));

	//multiply each element as a complex number
	for(int y=0;y<image.rows;y++) {
		for(int x=0;x<image.cols;x++) {
			std::complex<double> image_val(image.at<cv::Vec2f>(y,x)[0],image.at<cv::Vec2f>(y,x)[1]);
			std::complex<double> mask_val(mask.at<cv::Vec2f>(y,x)[0],mask.at<cv::Vec2f>(y,x)[1]);

			if(mask_val.real()!=0 && mask_val.imag()!=0) {
				std::complex<double> result_val = image_val / mask_val;
				result.at<cv::Vec2f>(y,x)[0] = result_val.real();
				result.at<cv::Vec2f>(y,x)[1] = result_val.imag();
			} else {
				result.at<cv::Vec2f>(y,x)[0] = image.at<cv::Vec2f>(y,x)[0];
				result.at<cv::Vec2f>(y,x)[1] = image.at<cv::Vec2f>(y,x)[1];
			}
		}
	}


	return result;
}

cv::Mat invert(const cv::Mat& image) {
	cv::Mat result(image.rows,image.cols,CV_32FC2,cv::Scalar::all(0));
	for(int y=0;y<image.rows;y++) {
		for(int x=0;x<image.cols;x++) {
			result.at<cv::Vec2f>(y,x)[0] = 1./image.at<cv::Vec2f>(y,x)[0];
			result.at<cv::Vec2f>(y,x)[1] = 1./image.at<cv::Vec2f>(y,x)[1];
		}
	}
	return result;
}
