/**
 * @file mainwindow.cpp
 * @authors Denys Pylypenko, Adam Veselý, Eliška Křeménková, Jaroslava Comová
 * 
 * @brief The main file for the calculator GUI
*/
#include "include/mainwindow.h"
#include "include/ui_mainwindow.h"

#include <QKeyEvent>
#include <QDesktopServices>
#include <QUrl>
#include <QObject>
#include <QtCore>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->bt_0, SIGNAL(released()), this, SLOT(bt_digit_pressed()));
    connect(ui->bt_1, SIGNAL(released()), this, SLOT(bt_digit_pressed()));
    connect(ui->bt_2, SIGNAL(released()), this, SLOT(bt_digit_pressed()));
    connect(ui->bt_3, SIGNAL(released()), this, SLOT(bt_digit_pressed()));
    connect(ui->bt_4, SIGNAL(released()), this, SLOT(bt_digit_pressed()));
    connect(ui->bt_5, SIGNAL(released()), this, SLOT(bt_digit_pressed()));
    connect(ui->bt_6, SIGNAL(released()), this, SLOT(bt_digit_pressed()));
    connect(ui->bt_7, SIGNAL(released()), this, SLOT(bt_digit_pressed()));
    connect(ui->bt_8, SIGNAL(released()), this, SLOT(bt_digit_pressed()));
    connect(ui->bt_9, SIGNAL(released()), this, SLOT(bt_digit_pressed()));
    connect(ui->bt_00, SIGNAL(released()), this, SLOT(bt_digit_pressed()));

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_bt_help_released(){
    QDesktopServices::openUrl(QUrl("https://kremenkova.github.io/help.html", QUrl::TolerantMode));
}

void MainWindow::bt_digit_pressed(){
    QPushButton * button = (QPushButton*)sender();
    write_number_on_display(button->text());
}

void MainWindow::on_bt_point_released(){
    if (check_errors(".")) return;    
    lb_long_string += ".";
    ui->lb_long->setText(lb_long_string);
    lb_number_string += ".";
    ui->lb_number->setText(lb_number_string);
}

void MainWindow::on_bt_sign_released(){
    if (!operation_pressed){ //if the last pressed button was number - change sign, otherwise do nothing
        if (negative){        
            lb_long_string.remove(lb_long_string.length() - lb_number_string.length() - 2, 2); //cuts the last occurrence of "(-"
            lb_long_string.chop(1); //cuts the last character, ")"
            double num = sign(lb_number_string.toDouble());
            lb_number_string = QString::number(num);            
            negative = false;   
        }
        else{
            lb_long_string.insert(lb_long_string.length() - lb_number_string.length(), "(-"); //insert "(-" before number
            lb_long_string.append(')'); //append ")" at the end of the string
            double num = sign(lb_number_string.toDouble());
            lb_number_string = QString::number(num);            
            negative = true;
        }
        ui->lb_long->setText(lb_long_string);
        ui->lb_number->setText(lb_number_string);
    } 
}

void MainWindow::on_bt_inverse_released(){
    if (!operation_pressed){ //if the last pressed button was number - invert, otherwise do nothing  
        if (negative){ 
            lb_long_string.remove(lb_long_string.length() - lb_number_string.length() - 1, lb_number_string.length()); //cuts the last number and replaces it with inverted one
            double num = inverse(lb_number_string.toDouble());
            lb_number_string = QString::number(num);
            lb_long_string.insert(lb_long_string.length() - 1, lb_number_string); //appends new inverted number at the end of the string
        }
        else{
            lb_long_string.chop(lb_number_string.length()); //cuts the last number and replaces it with inverted one
            double num = inverse(lb_number_string.toDouble());
            lb_number_string = QString::number(num);
            lb_long_string.append(lb_number_string); //appends new inverted number at the end of the string
        }
        ui->lb_long->setText(lb_long_string);
        ui->lb_number->setText(lb_number_string);
    }
}

void MainWindow::on_bt_plus_released(){
    operation_pressed = true;    
    if (check_errors("+")) return;
    lb_long_string += "+";
    ui->lb_long->setText(lb_long_string);
    evaluate(1);
    operation = "+";
}
void MainWindow::on_bt_minus_released(){
    operation_pressed = true;    
    if (check_errors("-")) return;
    lb_long_string += "-";
    ui->lb_long->setText(lb_long_string);
    evaluate(1);
    operation = "-";
}
void MainWindow::on_bt_multiply_released(){
    operation_pressed = true;    
    if (check_errors("*")) return;
    lb_long_string += "*";
    ui->lb_long->setText(lb_long_string);
    evaluate(1);
    operation = "*";
}
void MainWindow::on_bt_divide_released(){
    operation_pressed = true;    
    if (check_errors("/")) return;
    lb_long_string += "/";
    ui->lb_long->setText(lb_long_string);
    evaluate(1);
    operation = "/";
}
    
void MainWindow::on_bt_modulo_released(){
    operation_pressed = true;
    if (check_errors("%")) return;
    lb_long_string += "%";
    ui->lb_long->setText(lb_long_string);
    evaluate(1);
    operation = "%";
}
void MainWindow::on_bt_abs_released(){
    operation_pressed = true;
    if (check_errors("abs")) return;
    lb_long_string.prepend("|");
    lb_long_string.append("|");
    ui->lb_long->setText(lb_long_string);
    evaluate(1);
    operation = "abs";
    evaluate(0);
}
void MainWindow::on_bt_square_released(){
    operation_pressed = true;
    if (check_errors("^2")) return;
    lb_long_string += "^2";
    ui->lb_long->setText(lb_long_string);
    evaluate(1);
    operation = "^2"; 
    evaluate(0);       
}
void MainWindow::on_bt_exp_released(){
    operation_pressed = true;
    if (check_errors("^")) return;
    lb_long_string += "^"; 
    ui->lb_long->setText(lb_long_string);
    evaluate(1);
    operation = "^n";
}
void MainWindow::on_bt_square_root_released(){
    operation_pressed = true;
    if (check_errors("√")) return;
    lb_long_string.insert(lb_long_string.length() - lb_number_string.length(), "√");
    ui->lb_long->setText(lb_long_string);
    evaluate(TWO_OPERAND_TYPE);
    operation = "√";
    evaluate(ONE_OPERAND_TYPE);    
}
void MainWindow::on_bt_n_root_released(){
    operation_pressed = true;
    if (check_errors("ⁿ√")) return;
    lb_long_string += "^1/";
    ui->lb_long->setText(lb_long_string);
    evaluate(TWO_OPERAND_TYPE);
    operation = "ⁿ√";
}
void MainWindow::on_bt_factorial_released(){
    operation_pressed = true;
    if (check_errors("!")) return;
    lb_long_string += "!";
    ui->lb_long->setText(lb_long_string);
    evaluate(TWO_OPERAND_TYPE);
    operation = "!";
    evaluate(ONE_OPERAND_TYPE);
}

void MainWindow::on_bt_equal_released(){
    if (check_errors("=")) return;
    lb_long_string += "=";
    ui->lb_long->setText(lb_long_string);
    evaluate(TWO_OPERAND_TYPE);
    if (ui->lb_number->text() != "Error:"){
        lb_long_string += QString::number(result);
        ui->lb_long->setText(lb_long_string);
        store_result = result;
    }    
    RESET_VALUES
}

void MainWindow::on_bt_ans_released(){
    lb_long_string += "Ans";
    ui->lb_long->setText(lb_long_string);    
    lb_number_string = QString::number(store_result);
    ui->lb_number->setText(lb_number_string);
}

void MainWindow::on_bt_ac_released(){
    RESET_VALUES
    ui->lb_long->setText(lb_long_string);
    ui->lb_number->setText(lb_number_string);
}

void MainWindow::evaluate(bool operation_type){
    //for operations with two operands
    if (operation_type == 1){
        if (operation == ""){
            operand_1 = lb_number_string.toDouble();
            return;
        }
        else{
            operand_2 = lb_number_string.toDouble();
            //basic operations
            // +
            if (operation == "+") result = add(operand_1, operand_2);
            // -
            else if (operation == "-") result = subtract(operand_1, operand_2);
            // *
            else if (operation == "*") result = multiply(operand_1, operand_2);
            // /
            else if (operation == "/"){
                result = divide(operand_1, operand_2);
                //display error
                if (result == ERROR_DIVIDE_ZERO){
                    on_bt_ac_released();
                    ui->lb_number->setText("Error:");
                    ui->lb_long->setText("divide by zero");
                    return;
                }
            }
            //advanced operations
            // %
            else if (operation == "%") result = modulo(operand_1, operand_2);
            // ^n
            else if (operation == "^n"){
                result = exponent(operand_1, operand_2);
                //display error                
                if (result == ERROR_VALUE){
                    on_bt_ac_released();
                    ui->lb_number->setText("Error:");
                    ui->lb_long->setText("exponent must be natural value (n>=0), 0^0 is undefined");
                    return;
                }                
            } 
            // n root
            else if (operation == "ⁿ√"){
                result = root(operand_1, operand_2);
                //display error                
                if (result == ERROR_VALUE && (int)operand_2 % 2 == 0){ //if the root is even and the number is negative
                    on_bt_ac_released();
                    ui->lb_number->setText("Error:");
                    ui->lb_long->setText("root must be positive integer (n>0), number must be >= 0");
                    return;
                }
                else if (result == ERROR_VALUE && (int)operand_2 % 2 != 0){
                    on_bt_ac_released();
                    ui->lb_number->setText("Error:");
                    ui->lb_long->setText("root must be positive integer (n>0)");
                    return;
                }                
            }            
        }
    } 
    else{
        //advanced operations
        operand_1 = lb_number_string.toDouble();
        // |x|
        if (operation == "abs") result = absolute(operand_1);        
        // x^2
        else if (operation == "^2") result = exponent(operand_1, 2);
        // √
        else if (operation == "√"){
            result = root(operand_1, 2);
            if (result == ERROR_VALUE){
                on_bt_ac_released();
                ui->lb_number->setText("Error:");
                ui->lb_long->setText("number must be greater than 0");
                return;
            }
        }
        else if (operation == "!"){                        
            if (operand_1 >= MAX_FAKTORIAL){
                on_bt_ac_released();
                ui->lb_number->setText("Error:");
                ui->lb_long->setText("number is too big");
                return;
            }
            result = factorial(operand_1);
            if (result == ERROR_VALUE){
                on_bt_ac_released();
                ui->lb_number->setText("Error:");
                ui->lb_long->setText("number must be positive integer (n>0)");
                return;
            }
        }
    }
    //write on display
    lb_number_string = QString::number(result);
    ui->lb_number->setText(lb_number_string);
    operand_1 = result;
} //function evaluate - computes and execute the operations

bool MainWindow::check_errors(QString bt_check){
    //number starts with "."
    if (bt_check == "." && operation_pressed){
        ui->lb_long->setText("check_point_start");
        return true;
    }
    //two dots
    if (bt_check == "." && lb_number_string.last(1) == '.'){
        ui->lb_long->setText("check_two_point");
        return true;
    }
    //number ends with ".", dot after dot
    if ((operation_pressed || bt_check == "=") && lb_long_string.last(1) == '.'){
        ui->lb_long->setText("check_point_end");
        operation_pressed = false;
        return true;
    }
    //expresion cant start with an operation
    if (lb_long_string == ""){
        ui->lb_long->setText("check_op_start");
        return true;
    }
    //expresion cant end with an operation
    if (bt_check == "=" && (lb_long_string.last(1) == "+" || lb_long_string.last(1) == "-" || lb_long_string.last(1) == "*" || lb_long_string.last(1) == "/" || lb_long_string.last(1) == "%" || lb_long_string.last(2) == "^n" || lb_long_string.last(2) == "ⁿ√")){ 
        ui->lb_long->setText("check_op_end");
        return true;
    }
    //operations cant be entered after two operand operations operation
    if (!isdigit(lb_long_string[lb_long_string.length() - 1].toLatin1()) && (lb_long_string.last(1) == "+" || lb_long_string.last(1) == "-" || lb_long_string.last(1) == "*" || lb_long_string.last(1) == "/" || lb_long_string.last(1) == "%" || lb_long_string.last(2) == "^n" || lb_long_string.last(2) == "ⁿ√")){
        ui->lb_long->setText("check_two_op");
        return true;
    }    
    return false;
} //function check_errors - handles input errors

void MainWindow::write_number_on_display(QString number){
    //if true, clear lb_number for a new input
    if (operation_pressed){
        lb_number_string = "";
        operation_pressed = false;
    }
    lb_long_string += number;
    ui->lb_long->setText(lb_long_string);
    lb_number_string += number;
    ui->lb_number->setText(lb_number_string);
}

void MainWindow::keyPressEvent(QKeyEvent *event){
    switch (event->key()){
        case Qt::Key_H:
            on_bt_help_released();
            break;   
        case Qt::Key_0:
            write_number_on_display("0");
            break;
        case Qt::Key_1:
            write_number_on_display("1");
            break;
        case Qt::Key_2:
            write_number_on_display("2");
            break;
        case Qt::Key_3:
            write_number_on_display("3");
            break;
        case Qt::Key_4:
            write_number_on_display("4");
            break;
        case Qt::Key_5:
            write_number_on_display("5");
            break;
        case Qt::Key_6:
            write_number_on_display("6");
            break;
        case Qt::Key_7:
            write_number_on_display("7");
            break;
        case Qt::Key_8:
            write_number_on_display("8");
            break;
        case Qt::Key_9:
            write_number_on_display("9");
            break;
        case Qt::Key_Period:
            on_bt_point_released();
            break;
        case Qt::Key_Comma:
            on_bt_point_released();
            break;    
        case Qt::Key_G:
            on_bt_sign_released();
            break;
        case Qt::Key_I:
            on_bt_inverse_released();
            break;

        case Qt::Key_Plus:
            on_bt_plus_released();
            break;
        case Qt::Key_Minus:
            on_bt_minus_released();
            break;
        case Qt::Key_Asterisk:
            on_bt_multiply_released();
            break;
        case Qt::Key_Slash:
            on_bt_divide_released();
            break;

        case Qt::Key_Percent:
            on_bt_modulo_released();
            break;    
        case Qt::Key_B:
            on_bt_abs_released();
            break;
        case Qt::Key_S:
            on_bt_square_released();
            break;
        case Qt::Key_E:
            on_bt_exp_released();
            break;
        case Qt::Key_Q:
            on_bt_square_root_released();
            break;
        case Qt::Key_R:
            on_bt_n_root_released();
            break;
        case Qt::Key_Exclam:
            on_bt_factorial_released();
            break;
        case Qt::Key_Equal:
            on_bt_equal_released();
            break;
        case Qt::Key_Enter:
            on_bt_equal_released();
            break;
        case Qt::Key_Return:
            on_bt_equal_released();
            break;
        case Qt::Key_A:
            on_bt_ans_released();
            break;
        case Qt::Key_C:
            on_bt_ac_released();
            break;
        default:
            break;
    }
}

/*** End of file mainwindow.h ***/