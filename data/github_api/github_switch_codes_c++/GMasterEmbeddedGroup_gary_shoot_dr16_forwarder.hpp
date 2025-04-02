#pragma once
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "gary_msgs/msg/dr16_receiver.hpp"
#include "std_msgs/msg/float64.hpp"
#include "gary_msgs/msg/auto_aim.hpp"
#include "gary_msgs/msg/client_command.hpp"
#include "gary_msgs/srv/switch_cover.hpp"
#include "gary_msgs/srv/vision_mode_switch.hpp"
#include <string>
#include <chrono>
namespace gary_shoot{
    using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
    class DR16Forwarder : public rclcpp_lifecycle::LifecycleNode {

    public:
        explicit DR16Forwarder(const rclcpp::NodeOptions & options);


    private:

        CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
        CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous_state) override;
        CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
        CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
        CallbackReturn on_shutdown(const rclcpp_lifecycle::State & previous_state) override;
        CallbackReturn on_error(const rclcpp_lifecycle::State & previous_state) override;

        //callback group
        rclcpp::CallbackGroup::SharedPtr cb_group;

        std_msgs::msg::Float64 ShooterWheelOnMsg;
        rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64>::SharedPtr ShooterWheelOnPublisher;
        std_msgs::msg::Float64 TriggerWheelOnMsg;
        rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float64>::SharedPtr TriggerWheelOnPublisher;


        void rc_callback(gary_msgs::msg::DR16Receiver::SharedPtr msg);
        rclcpp::Subscription<gary_msgs::msg::DR16Receiver>::SharedPtr RemoteControlSubscription;

        void autoaim_callback(gary_msgs::msg::AutoAIM::SharedPtr msg);
        rclcpp::Subscription<gary_msgs::msg::AutoAIM>::SharedPtr autoaim_sub;

        void client_callback(gary_msgs::msg::ClientCommand::SharedPtr msg);
        rclcpp::Subscription<gary_msgs::msg::ClientCommand>::SharedPtr client_sub;

        rclcpp::Client<gary_msgs::srv::SwitchCover>::SharedPtr switch_cover_client;
        std::shared_future<gary_msgs::srv::SwitchCover::Response::SharedPtr> cover_resp;

        rclcpp::Client<gary_msgs::srv::VisionModeSwitch>::SharedPtr switch_vision_client;
        std::shared_future<gary_msgs::srv::VisionModeSwitch::Response::SharedPtr> vision_resp;


        void data_publisher();

        //timer
        rclcpp::TimerBase::SharedPtr timer_update;

        double update_freq;

        double shooter_wheel_pid_target;
        double trigger_wheel_pid_target;
        double trigger_wheel_pid_target_set;
        std::string remote_control_topic;
        std::string shooter_wheel_topic;
        std::string trigger_wheel_topic;
        std::string autoaim_topic;
        std::uint8_t prev_switch_state;
        std::uint8_t switch_state;
        std::uint8_t right_switch_state;
        bool shooter_on;
        bool trigger_on;
        bool use_auto_fire;
        bool cover_open;
        double freq_factor;

        std::chrono::time_point<std::chrono::steady_clock> stop_shoot_pressed_time_point;
    };
}
