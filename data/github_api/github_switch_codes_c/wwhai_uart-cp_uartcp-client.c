// Copyright (C) 2025 wwhai
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "uartcp-client.h"

// 函数：设置串口参数
void set_serial(int fd, int baudrate, int databits, char parity, int stopbits)
{
    struct termios options;
    tcgetattr(fd, &options);
    // 设置输入输出波特率
    cfsetispeed(&options, baudrate);
    cfsetospeed(&options, baudrate);
    // 数据位
    options.c_cflag &= ~CSIZE;
    switch (databits)
    {
    case 8:
        options.c_cflag |= CS8;
        break;
        // 可添加其他数据位设置
    }
    // 校验位
    switch (parity)
    {
    case 'N':
        options.c_cflag &= ~PARENB;
        break;
        // 可添加其他校验位设置
    }
    // 停止位
    switch (stopbits)
    {
    case 1:
        options.c_cflag &= ~CSTOPB;
        break;
        // 可添加其他停止位设置
    }
    // 其他配置
    options.c_cflag |= CLOCAL | CREAD;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_oflag = 0;
    options.c_lflag = 0;
    tcsetattr(fd, TCSANOW, &options);
}

// 函数：显示进度条
void show_progress(int current, int total)
{
    int percent = (current * 100) / total;
    int bar_width = 50;
    int pos = (bar_width * percent) / 100;
    printf("[");
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < pos)
            printf("#");
        else
            printf(" ");
    }
    printf("] %d%%\r", percent);
    fflush(stdout);
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        fprintf(stderr, "Usage: %s <file> <port> <baudrate> <databits> <parity> <stopbits>\n", argv[0]);
        return 1;
    }
    char *file_path = argv[1];
    char *port = argv[2];
    int baudrate = atoi(argv[3]);
    int databits = atoi(argv[4]);
    char parity = argv[5][0];
    int stopbits = atoi(argv[6]);

    // 打开串口
    int serial_fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (serial_fd < 0)
    {
        perror("Open serial port failed");
        return 1;
    }
    set_serial(serial_fd, baudrate, databits, parity, stopbits);

    // 打开文件
    int file_fd = open(file_path, O_RDONLY);
    if (file_fd < 0)
    {
        perror("Open file failed");
        close(serial_fd);
        return 1;
    }
    struct stat st;
    fstat(file_fd, &st);
    int file_size = st.st_size;
    int sent_size = 0;
    char buffer[1024];
    ssize_t bytes_read, bytes_written;
    while ((bytes_read = read(file_fd, buffer, sizeof(buffer))) > 0)
    {
        bytes_written = write(serial_fd, buffer, bytes_read);
        if (bytes_written < 0)
        {
            perror("Write to serial failed");
            close(serial_fd);
            close(file_fd);
            return 1;
        }
        sent_size += bytes_written;
        show_progress(sent_size, file_size);
    }
    close(serial_fd);
    close(file_fd);
    printf("\nFile sent successfully!\n");
    return 0;
}