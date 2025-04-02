#include "main_node.hpp"
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <vector>
#include <iostream>

// This represents the maximal motor rotation.
// 0.29 degrees per 1 impulse.
// Angles 300 - 360 are in invalid range.
#define MOTRO_ANGLES_TO_IMPULZES (1/0.29)
#define MOTRO_IMPULZES_TO_ANGLES 0.29
#define BASE_MOTOR_ID 1
#define CAMERA_MOTOR_ID 2
#define IP_BLOCK_NAME "nn_inference"

MiniprojectNode::MiniprojectNode()
	: rclcpp::Node("main_node")
{
	int status = XNn_inference_Initialize(&m_inference, IP_BLOCK_NAME);
	if (status != XST_SUCCESS) {
		RCLCPP_ERROR(this->get_logger(), "Could not initialize IP block.\n");
		throw std::runtime_error("Could not initialize IP block\n");
	}

	m_cameraSubscriber = this->create_subscription<sensor_msgs::msg::Image>(
		"/image_raw", 10, std::bind(&MiniprojectNode::getImageCallback, this, std::placeholders::_1));

	m_motorPositionPublsher = this->create_publisher<SetPosition>("/set_position", 10);
	m_timer = this->create_wall_timer(std::chrono::seconds(2), [this] () { this->publishBaseMotorRotation(BASE_MOTOR_ID); });
}

MiniprojectNode::~MiniprojectNode()
{
	XNn_inference_Release(&m_inference);
}

std::vector<float> MiniprojectNode::flatten(cv::Mat image)
{
	// Flatten the image
	cv::Mat flattenedImage = image.reshape(1, 1);

	std::vector<float> flattenedVector(IMG_SIZE, 0);
	std::transform(flattenedImage.begin<char>(), flattenedImage.end<char>(), flattenedVector.begin(), [](char c) -> float {return c/255.;});

	return flattenedVector;
}

void MiniprojectNode::getImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
	cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
	// Check for empty frame.
	if (cv_ptr->image.empty()) {
		return;
	}

	m_frame = cv_ptr->image;
}

bool MiniprojectNode::checkIfFits(const cv::Mat& frame)
{
	// Convert the frame to grayscale (if needed)
	cv::Mat grayFrame;
	cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

	// Apply Gaussian blur
	cv::GaussianBlur(grayFrame, grayFrame, cv::Size(5, 5), 0);

	// Use Canny edge detection to find edges in the blurred frame
	cv::Canny(grayFrame, grayFrame, 30, 100);

	// Find lines in the frame using HoughLinesP
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(grayFrame, lines, 1, CV_PI / 180, 50, 80, 10);

	// Iterate through the lines and check if any line is perpendicular to the top border
	if (!lines.empty()) {
		for (const auto& line : lines) {
			int x1, x2;
			std::tie(x1, std::ignore, x2, std::ignore) = std::tie(line[0], line[1], line[2], line[3]);

			// Check if the line is approximately vertical
			if (std::abs(x1 - x2) < 5) { // Adjust the threshold if needed
				return true; // Return true
			}
		}
	}

	return false; // Return false if not already true
}

MiniprojectNode::SetPosition MiniprojectNode::constructMsg(int id, int angle)
{
	SetPosition msg;
	msg.id = id;
	msg.position = angle;
	return msg;
}

void MiniprojectNode::findScrewRotation()
{
	for (int i = 0; i < 256; i++) {
		m_motorPositionPublsher->publish(constructMsg(CAMERA_MOTOR_ID, i*4));
		RCLCPP_DEBUG(this->get_logger(), std::string("Checking rotation: ") + std::to_string(i*4 * MOTRO_IMPULZES_TO_ANGLES));
		usleep(500);

		if (checkIfFits(m_frame)) {
			m_motorPositionPublsher->publish(constructMsg(CAMERA_MOTOR_ID, (i-1)*4));
			RCLCPP_INFO(this->get_logger(), std::string("Rotation found: ") + std::to_string((i-1)*4 * MOTRO_IMPULZES_TO_ANGLES));
			return;
		}
	}
}

void MiniprojectNode::publishBaseMotorRotation(int id)
{
	m_timer->cancel();

	for (int i = 5; i < 8; i++) {
		auto msg = constructMsg(id, i*80);
		RCLCPP_DEBUG(this->get_logger(), "Sending to motor: [ID]: %d [POSITION]: %d", msg.id, msg.position);
		m_motorPositionPublsher->publish(msg);
		sleep(1);

		auto screwType = detectScrewType(m_frame);
		if (screwType == ScrewType::Hexagonal || screwType == ScrewType::Nut) {
			findScrewRotation();
			sleep(1);
		}
		else {
			RCLCPP_WARN(this->get_logger(), "Incorrect type of screw");
		}
	}
}

MiniprojectNode::ScrewType MiniprojectNode::detectScrewType(cv::Mat frame)
{
	cv::resize(frame, frame, {32, 24}, 0, 0, cv::INTER_AREA);
	cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

	auto check = flatten(frame);

	while (!XNn_inference_IsReady(&m_inference)) {}
	XNn_inference_Write_Data_In_Words(&m_inference, 0, (word_type *)check.data(), IMG_SIZE);

	XNn_inference_Start(&m_inference);

	while (!XNn_inference_IsDone(&m_inference)) {}

	auto screwer =  static_cast<ScrewType>(XNn_inference_Get_Pred_out(&m_inference));
	RCLCPP_DEBUG(this->get_logger(), "Detected screw type: %s", getName(screwer).c_str());
	return screwer;
}

std::string MiniprojectNode::getName(ScrewType screwType)
{
	switch (screwType) {
		case ScrewType::Hexagonal:
			return "HEXAGONAL";
		case ScrewType::Nut:
			return "NUT";
		case ScrewType::Philips:
			return "PHILIPS";
		default:
			return "None";
	}
}

int main(int argc, char *argv[]) {
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<MiniprojectNode>());
	rclcpp::shutdown();
	return 0;
}

