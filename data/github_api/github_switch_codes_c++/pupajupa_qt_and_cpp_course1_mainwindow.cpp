#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "stack.h"
MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, ui(new Ui::MainWindow)
{
	ui->setupUi(this);
	operators = new Stack;
}

MainWindow::~MainWindow()
{
	delete ui;
}


void MainWindow::on_pushButton_clicked()
{
	QString str = ui->formula->text();
	for(int i = 0;i<str.size();i++)
	{
		if(!str[i].isDigit())
		{
			if(signer.see()==0)
			{
				signer.push(str[i]);
			}
			else if()
		}
		if (str[i]=='+' || str[i] == '-'||str[i]=='*'|| str[i]=='/')
		{
				if()
		}
	}
}

int MainWindow::pror(QChar c)
{
	int p;
	switch (c.toLatin1())
	{
	case '(': p = 0;
	case ')': p = 1;
	case '+':case '-': p = 5;
		break;
	case '*': case '/': p = 4;
		break;
	case '^': p = 2;
		break;
	default:
		break;
	}
	return p;
}

double MainWindow::value(const QChar &c)
{
	switch(c.toLatin1())
	{
	case 'a':
		return ui->aValue->value();
	case 'b':
		return ui->bValue->value();
	case 'c':
		return ui->cValue->value();
	case 'd':
		return ui->dValue->value();
	case 'e':
		return ui->eValue->value();
	}
}
