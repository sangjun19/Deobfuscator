	.file	"kswit_Arduino-VT100-I2C-LCD_termios_flatten.c"
	.text
	.globl	retval
	.bss
	.align 4
	.type	retval, @object
	.size	retval, 4
retval:
	.zero	4
	.globl	_TIG_IZ_K6yL_envp
	.align 8
	.type	_TIG_IZ_K6yL_envp, @object
	.size	_TIG_IZ_K6yL_envp, 8
_TIG_IZ_K6yL_envp:
	.zero	8
	.globl	_TIG_IZ_K6yL_argc
	.align 4
	.type	_TIG_IZ_K6yL_argc, @object
	.size	_TIG_IZ_K6yL_argc, 4
_TIG_IZ_K6yL_argc:
	.zero	4
	.globl	rfds
	.align 32
	.type	rfds, @object
	.size	rfds, 128
rfds:
	.zero	128
	.globl	STOP
	.align 4
	.type	STOP, @object
	.size	STOP, 4
STOP:
	.zero	4
	.globl	tty
	.align 32
	.type	tty, @object
	.size	tty, 60
tty:
	.zero	60
	.globl	_TIG_IZ_K6yL_argv
	.align 8
	.type	_TIG_IZ_K6yL_argv, @object
	.size	_TIG_IZ_K6yL_argv, 8
_TIG_IZ_K6yL_argv:
	.zero	8
	.globl	wait_flag
	.align 4
	.type	wait_flag, @object
	.size	wait_flag, 4
wait_flag:
	.zero	4
	.globl	read_buf
	.align 32
	.type	read_buf, @object
	.size	read_buf, 256
read_buf:
	.zero	256
	.globl	tv
	.align 16
	.type	tv, @object
	.size	tv, 16
tv:
	.zero	16
	.section	.rodata
.LC0:
	.string	"stop %d ,%d"
.LC1:
	.string	"input buff=0x%x \n"
.LC2:
	.string	"write %d,%d "
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, tty(%rip)
	movl	$0, 4+tty(%rip)
	movl	$0, 8+tty(%rip)
	movl	$0, 12+tty(%rip)
	movb	$0, 16+tty(%rip)
	movb	$0, 17+tty(%rip)
	movb	$0, 18+tty(%rip)
	movb	$0, 19+tty(%rip)
	movb	$0, 20+tty(%rip)
	movb	$0, 21+tty(%rip)
	movb	$0, 22+tty(%rip)
	movb	$0, 23+tty(%rip)
	movb	$0, 24+tty(%rip)
	movb	$0, 25+tty(%rip)
	movb	$0, 26+tty(%rip)
	movb	$0, 27+tty(%rip)
	movb	$0, 28+tty(%rip)
	movb	$0, 29+tty(%rip)
	movb	$0, 30+tty(%rip)
	movb	$0, 31+tty(%rip)
	movb	$0, 32+tty(%rip)
	movb	$0, 33+tty(%rip)
	movb	$0, 34+tty(%rip)
	movb	$0, 35+tty(%rip)
	movb	$0, 36+tty(%rip)
	movb	$0, 37+tty(%rip)
	movb	$0, 38+tty(%rip)
	movb	$0, 39+tty(%rip)
	movb	$0, 40+tty(%rip)
	movb	$0, 41+tty(%rip)
	movb	$0, 42+tty(%rip)
	movb	$0, 43+tty(%rip)
	movb	$0, 44+tty(%rip)
	movb	$0, 45+tty(%rip)
	movb	$0, 46+tty(%rip)
	movb	$0, 47+tty(%rip)
	movb	$0, 48+tty(%rip)
	movl	$0, 52+tty(%rip)
	movl	$0, 56+tty(%rip)
	nop
.L2:
	movl	$1, wait_flag(%rip)
	nop
.L3:
	movl	$0, retval(%rip)
	nop
.L4:
	movq	$0, tv(%rip)
	movq	$0, 8+tv(%rip)
	nop
.L5:
	movq	$0, rfds(%rip)
	movq	$0, 8+rfds(%rip)
	movq	$0, 16+rfds(%rip)
	movq	$0, 24+rfds(%rip)
	movq	$0, 32+rfds(%rip)
	movq	$0, 40+rfds(%rip)
	movq	$0, 48+rfds(%rip)
	movq	$0, 56+rfds(%rip)
	movq	$0, 64+rfds(%rip)
	movq	$0, 72+rfds(%rip)
	movq	$0, 80+rfds(%rip)
	movq	$0, 88+rfds(%rip)
	movq	$0, 96+rfds(%rip)
	movq	$0, 104+rfds(%rip)
	movq	$0, 112+rfds(%rip)
	movq	$0, 120+rfds(%rip)
	nop
.L6:
	movl	$0, STOP(%rip)
	nop
.L7:
	movl	$0, -72(%rbp)
	jmp	.L8
.L9:
	movl	-72(%rbp), %eax
	cltq
	leaq	read_buf(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -72(%rbp)
.L8:
	cmpl	$255, -72(%rbp)
	jle	.L9
	nop
.L10:
	movq	$0, _TIG_IZ_K6yL_envp(%rip)
	nop
.L11:
	movq	$0, _TIG_IZ_K6yL_argv(%rip)
	nop
.L12:
	movl	$0, _TIG_IZ_K6yL_argc(%rip)
	nop
	nop
.L13:
.L14:
#APP
# 207 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-K6yL--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_K6yL_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_K6yL_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_K6yL_envp(%rip)
	nop
	movq	$20, -32(%rbp)
.L56:
	cmpq	$33, -32(%rbp)
	ja	.L59
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L17(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L17(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L17:
	.long	.L39-.L17
	.long	.L59-.L17
	.long	.L38-.L17
	.long	.L59-.L17
	.long	.L37-.L17
	.long	.L36-.L17
	.long	.L35-.L17
	.long	.L34-.L17
	.long	.L59-.L17
	.long	.L33-.L17
	.long	.L32-.L17
	.long	.L31-.L17
	.long	.L30-.L17
	.long	.L29-.L17
	.long	.L28-.L17
	.long	.L27-.L17
	.long	.L59-.L17
	.long	.L26-.L17
	.long	.L25-.L17
	.long	.L24-.L17
	.long	.L23-.L17
	.long	.L59-.L17
	.long	.L59-.L17
	.long	.L22-.L17
	.long	.L21-.L17
	.long	.L59-.L17
	.long	.L20-.L17
	.long	.L59-.L17
	.long	.L59-.L17
	.long	.L19-.L17
	.long	.L59-.L17
	.long	.L59-.L17
	.long	.L18-.L17
	.long	.L16-.L17
	.text
.L25:
	cmpl	$15, -64(%rbp)
	ja	.L40
	movq	$6, -32(%rbp)
	jmp	.L42
.L40:
	movq	$10, -32(%rbp)
	jmp	.L42
.L37:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L57
	jmp	.L58
.L28:
	movzbl	-79(%rbp), %edx
	movl	-76(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, %edi
	call	sleep@PLT
	movl	$1, -76(%rbp)
	movl	$0, -60(%rbp)
	movl	-68(%rbp), %eax
	leaq	tty(%rip), %rdx
	movl	$2, %esi
	movl	%eax, %edi
	call	tcsetattr@PLT
	movq	$17, -32(%rbp)
	jmp	.L42
.L27:
	cmpl	$149, -56(%rbp)
	jg	.L44
	movq	$32, -32(%rbp)
	jmp	.L42
.L44:
	movq	$2, -32(%rbp)
	jmp	.L42
.L30:
	cmpl	$0, -48(%rbp)
	jle	.L46
	movq	$0, -32(%rbp)
	jmp	.L42
.L46:
	movq	$17, -32(%rbp)
	jmp	.L42
.L22:
	movl	$0, -60(%rbp)
	movl	$0, -56(%rbp)
	movq	$15, -32(%rbp)
	jmp	.L42
.L21:
	movzbl	-79(%rbp), %eax
	xorl	$1, %eax
	testb	%al, %al
	je	.L48
	movq	$29, -32(%rbp)
	jmp	.L42
.L48:
	movq	$14, -32(%rbp)
	jmp	.L42
.L20:
	movl	$1, %edi
	call	sleep@PLT
	movq	$5, -32(%rbp)
	jmp	.L42
.L31:
	cmpl	$0, -68(%rbp)
	jle	.L50
	movq	$9, -32(%rbp)
	jmp	.L42
.L50:
	movq	$7, -32(%rbp)
	jmp	.L42
.L33:
	leaq	rfds(%rip), %rax
	movq	%rax, -40(%rbp)
	movl	$0, -64(%rbp)
	movq	$18, -32(%rbp)
	jmp	.L42
.L29:
	movl	-76(%rbp), %eax
	testl	%eax, %eax
	jle	.L52
	movq	$33, -32(%rbp)
	jmp	.L42
.L52:
	movq	$24, -32(%rbp)
	jmp	.L42
.L24:
	movb	$97, -80(%rbp)
	movb	$68, -78(%rbp)
	movb	$17, -77(%rbp)
	movq	-96(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movl	$13, %esi
	movq	%rax, %rdi
	call	serial_open
	movl	%eax, -68(%rbp)
	movq	$11, -32(%rbp)
	jmp	.L42
.L18:
	movl	-68(%rbp), %eax
	movl	$256, %edx
	leaq	read_buf(%rip), %rcx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	%eax, -52(%rbp)
	leaq	-80(%rbp), %rcx
	movl	-68(%rbp), %eax
	movl	$1, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, -76(%rbp)
	movq	$13, -32(%rbp)
	jmp	.L42
.L26:
	addl	$1, -56(%rbp)
	movq	$15, -32(%rbp)
	jmp	.L42
.L35:
	movq	-40(%rbp), %rax
	movl	-64(%rbp), %edx
	movq	$0, (%rax,%rdx,8)
	addl	$1, -64(%rbp)
	movq	$18, -32(%rbp)
	jmp	.L42
.L36:
	movzbl	read_buf(%rip), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-60(%rbp), %eax
	movl	%eax, -44(%rbp)
	addl	$1, -60(%rbp)
	movl	-56(%rbp), %edx
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-68(%rbp), %eax
	movl	%eax, %edi
	call	tcdrain@PLT
	movq	$12, -32(%rbp)
	jmp	.L42
.L16:
	movzbl	read_buf(%rip), %eax
	cmpb	$19, %al
	jne	.L54
	movq	$26, -32(%rbp)
	jmp	.L42
.L54:
	movq	$5, -32(%rbp)
	jmp	.L42
.L32:
	movl	-68(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movslq	%eax, %rdx
	leaq	0(,%rdx,8), %rcx
	leaq	rfds(%rip), %rdx
	movq	(%rcx,%rdx), %rsi
	movl	-68(%rbp), %edx
	andl	$63, %edx
	movl	$1, %edi
	movl	%edx, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rdx
	movq	%rsi, %rcx
	orq	%rdx, %rcx
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	rfds(%rip), %rax
	movq	%rcx, (%rdx,%rax)
	movq	$5, tv(%rip)
	movq	$0, 8+tv(%rip)
	movl	$5, %edi
	call	sleep@PLT
	movl	-68(%rbp), %eax
	movl	$3, %esi
	movl	%eax, %edi
	call	tcflow@PLT
	movq	$23, -32(%rbp)
	jmp	.L42
.L39:
	movb	$0, -79(%rbp)
	movq	$17, -32(%rbp)
	jmp	.L42
.L34:
	movl	$1, %edi
	call	exit@PLT
.L19:
	movl	-68(%rbp), %eax
	movl	$0, %esi
	movl	%eax, %edi
	call	tcflow@PLT
	movq	$14, -32(%rbp)
	jmp	.L42
.L38:
	movl	-68(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$4, -32(%rbp)
	jmp	.L42
.L23:
	movq	$19, -32(%rbp)
	jmp	.L42
.L59:
	nop
.L42:
	jmp	.L56
.L58:
	call	__stack_chk_fail@PLT
.L57:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.globl	serial_open
	.type	serial_open, @function
serial_open:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	$3, -8(%rbp)
.L81:
	cmpq	$10, -8(%rbp)
	ja	.L82
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L63(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L63(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L63:
	.long	.L72-.L63
	.long	.L71-.L63
	.long	.L70-.L63
	.long	.L69-.L63
	.long	.L68-.L63
	.long	.L82-.L63
	.long	.L67-.L63
	.long	.L66-.L63
	.long	.L65-.L63
	.long	.L64-.L63
	.long	.L62-.L63
	.text
.L68:
	movl	-20(%rbp), %eax
	jmp	.L73
.L65:
	cmpl	$0, -20(%rbp)
	jns	.L74
	movq	$0, -8(%rbp)
	jmp	.L76
.L74:
	movq	$7, -8(%rbp)
	jmp	.L76
.L71:
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	tty(%rip), %rax
	movq	%rax, %rdi
	call	cfsetospeed@PLT
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	tty(%rip), %rax
	movq	%rax, %rdi
	call	cfsetispeed@PLT
	movl	8+tty(%rip), %eax
	andl	$-65, %eax
	movl	%eax, 8+tty(%rip)
	movl	8+tty(%rip), %eax
	andl	$-49, %eax
	movl	%eax, 8+tty(%rip)
	movl	8+tty(%rip), %eax
	orl	$48, %eax
	movl	%eax, 8+tty(%rip)
	movl	8+tty(%rip), %eax
	andl	$2147483647, %eax
	movl	%eax, 8+tty(%rip)
	movl	8+tty(%rip), %eax
	orl	$2176, %eax
	movl	%eax, 8+tty(%rip)
	movl	8+tty(%rip), %eax
	andb	$-4, %ah
	movl	%eax, 8+tty(%rip)
	movl	12+tty(%rip), %eax
	andl	$-3, %eax
	movl	%eax, 12+tty(%rip)
	movl	12+tty(%rip), %eax
	andl	$-9, %eax
	movl	%eax, 12+tty(%rip)
	movl	12+tty(%rip), %eax
	andl	$-17, %eax
	movl	%eax, 12+tty(%rip)
	movl	12+tty(%rip), %eax
	andl	$-65, %eax
	movl	%eax, 12+tty(%rip)
	movl	12+tty(%rip), %eax
	orl	$1, %eax
	movl	%eax, 12+tty(%rip)
	movl	tty(%rip), %eax
	andb	$-29, %ah
	movl	%eax, tty(%rip)
	movl	tty(%rip), %eax
	andl	$-492, %eax
	movl	%eax, tty(%rip)
	movl	4+tty(%rip), %eax
	andl	$-5, %eax
	movl	%eax, 4+tty(%rip)
	movl	8+tty(%rip), %eax
	andb	$-5, %ah
	movl	%eax, 8+tty(%rip)
	movb	$17, 25+tty(%rip)
	movb	$19, 26+tty(%rip)
	movb	$0, 22+tty(%rip)
	movb	$0, 23+tty(%rip)
	movl	-20(%rbp), %eax
	leaq	tty(%rip), %rdx
	movl	$0, %esi
	movl	%eax, %edi
	call	tcsetattr@PLT
	movl	%eax, -12(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L76
.L69:
	movq	-40(%rbp), %rax
	movl	$2306, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -20(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L76
.L64:
	movl	$-1, %eax
	jmp	.L73
.L67:
	movl	$-1, %eax
	jmp	.L73
.L62:
	cmpl	$0, -12(%rbp)
	je	.L77
	movq	$9, -8(%rbp)
	jmp	.L76
.L77:
	movq	$4, -8(%rbp)
	jmp	.L76
.L72:
	movl	$-1, %eax
	jmp	.L73
.L66:
	movl	-20(%rbp), %eax
	leaq	tty(%rip), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	tcgetattr@PLT
	movl	%eax, -16(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L76
.L70:
	cmpl	$0, -16(%rbp)
	jns	.L79
	movq	$6, -8(%rbp)
	jmp	.L76
.L79:
	movq	$1, -8(%rbp)
	jmp	.L76
.L82:
	nop
.L76:
	jmp	.L81
.L73:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	serial_open, .-serial_open
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
