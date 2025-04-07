	.file	"AkKsenia_unix_process_communication_b_flatten.c"
	.text
	.globl	_TIG_IZ_R88u_argv
	.bss
	.align 8
	.type	_TIG_IZ_R88u_argv, @object
	.size	_TIG_IZ_R88u_argv, 8
_TIG_IZ_R88u_argv:
	.zero	8
	.globl	_TIG_IZ_R88u_argc
	.align 4
	.type	_TIG_IZ_R88u_argc, @object
	.size	_TIG_IZ_R88u_argc, 4
_TIG_IZ_R88u_argc:
	.zero	4
	.globl	_TIG_IZ_R88u_envp
	.align 8
	.type	_TIG_IZ_R88u_envp, @object
	.size	_TIG_IZ_R88u_envp, 8
_TIG_IZ_R88u_envp:
	.zero	8
	.globl	pid
	.align 4
	.type	pid, @object
	.size	pid, 4
pid:
	.zero	4
	.section	.rodata
.LC0:
	.string	"|"
.LC1:
	.string	"dup2"
	.align 8
.LC2:
	.string	"Program B sent a signal to program A!"
.LC3:
	.string	"fork"
.LC4:
	.string	"execvp"
	.align 8
.LC5:
	.string	"Enter the programs and their parameters:"
.LC6:
	.string	" \n"
.LC7:
	.string	"pipe"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-294912(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$176, %rsp
	movl	%edi, -295060(%rbp)
	movq	%rsi, -295072(%rbp)
	movq	%rdx, -295080(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, pid(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_R88u_envp(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_R88u_argv(%rip)
	nop
.L4:
	movl	$0, _TIG_IZ_R88u_argc(%rip)
	nop
	nop
.L5:
.L6:
#APP
# 152 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-R88u--0
# 0 "" 2
#NO_APP
	movl	-295060(%rbp), %eax
	movl	%eax, _TIG_IZ_R88u_argc(%rip)
	movq	-295072(%rbp), %rax
	movq	%rax, _TIG_IZ_R88u_argv(%rip)
	movq	-295080(%rbp), %rax
	movq	%rax, _TIG_IZ_R88u_envp(%rip)
	nop
	movq	$76, -294952(%rbp)
.L103:
	cmpq	$78, -294952(%rbp)
	ja	.L105
	movq	-294952(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L9(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L9(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L9:
	.long	.L60-.L9
	.long	.L59-.L9
	.long	.L58-.L9
	.long	.L105-.L9
	.long	.L57-.L9
	.long	.L105-.L9
	.long	.L56-.L9
	.long	.L55-.L9
	.long	.L54-.L9
	.long	.L53-.L9
	.long	.L52-.L9
	.long	.L51-.L9
	.long	.L50-.L9
	.long	.L49-.L9
	.long	.L48-.L9
	.long	.L105-.L9
	.long	.L105-.L9
	.long	.L47-.L9
	.long	.L46-.L9
	.long	.L105-.L9
	.long	.L105-.L9
	.long	.L45-.L9
	.long	.L105-.L9
	.long	.L44-.L9
	.long	.L43-.L9
	.long	.L42-.L9
	.long	.L41-.L9
	.long	.L105-.L9
	.long	.L105-.L9
	.long	.L105-.L9
	.long	.L40-.L9
	.long	.L39-.L9
	.long	.L38-.L9
	.long	.L105-.L9
	.long	.L37-.L9
	.long	.L105-.L9
	.long	.L105-.L9
	.long	.L36-.L9
	.long	.L35-.L9
	.long	.L105-.L9
	.long	.L34-.L9
	.long	.L105-.L9
	.long	.L105-.L9
	.long	.L105-.L9
	.long	.L33-.L9
	.long	.L32-.L9
	.long	.L31-.L9
	.long	.L105-.L9
	.long	.L105-.L9
	.long	.L30-.L9
	.long	.L105-.L9
	.long	.L29-.L9
	.long	.L28-.L9
	.long	.L27-.L9
	.long	.L26-.L9
	.long	.L25-.L9
	.long	.L24-.L9
	.long	.L23-.L9
	.long	.L105-.L9
	.long	.L22-.L9
	.long	.L105-.L9
	.long	.L21-.L9
	.long	.L105-.L9
	.long	.L20-.L9
	.long	.L19-.L9
	.long	.L105-.L9
	.long	.L105-.L9
	.long	.L18-.L9
	.long	.L17-.L9
	.long	.L16-.L9
	.long	.L105-.L9
	.long	.L15-.L9
	.long	.L14-.L9
	.long	.L13-.L9
	.long	.L12-.L9
	.long	.L105-.L9
	.long	.L11-.L9
	.long	.L10-.L9
	.long	.L8-.L9
	.text
.L46:
	movq	-294976(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -295036(%rbp)
	movq	$24, -294952(%rbp)
	jmp	.L61
.L42:
	movl	-295024(%rbp), %eax
	cmpl	-295044(%rbp), %eax
	jg	.L62
	movq	$59, -294952(%rbp)
	jmp	.L61
.L62:
	movq	$40, -294952(%rbp)
	jmp	.L61
.L30:
	movl	$0, -295024(%rbp)
	movq	$25, -294952(%rbp)
	jmp	.L61
.L28:
	addl	$1, -295044(%rbp)
	movl	$0, -295040(%rbp)
	movq	$72, -294952(%rbp)
	jmp	.L61
.L57:
	movl	-295032(%rbp), %eax
	cltq
	imulq	-294960(%rbp), %rax
	movq	%rax, %rdx
	movq	-294968(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rdi
	call	pipe@PLT
	movl	%eax, -295028(%rbp)
	movq	$53, -294952(%rbp)
	jmp	.L61
.L40:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L48:
	movl	$0, -295032(%rbp)
	movq	$68, -294952(%rbp)
	jmp	.L61
.L24:
	movl	-295000(%rbp), %eax
	cmpl	-295044(%rbp), %eax
	jge	.L64
	movq	$46, -294952(%rbp)
	jmp	.L61
.L64:
	movq	$73, -294952(%rbp)
	jmp	.L61
.L39:
	movl	$0, %edi
	call	exit@PLT
.L50:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L16:
	movl	-295024(%rbp), %eax
	subl	$1, %eax
	cltq
	imulq	-294960(%rbp), %rax
	movq	%rax, %rdx
	movq	-294968(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	$0, %esi
	movl	%eax, %edi
	call	dup2@PLT
	movl	%eax, -295008(%rbp)
	movq	$9, -294952(%rbp)
	jmp	.L61
.L54:
	cmpl	$0, -295024(%rbp)
	jne	.L66
	movq	$51, -294952(%rbp)
	jmp	.L61
.L66:
	movq	$34, -294952(%rbp)
	jmp	.L61
.L32:
	call	getppid@PLT
	movl	%eax, -294980(%rbp)
	movl	-294980(%rbp), %eax
	movl	$10, %esi
	movl	%eax, %edi
	call	kill@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$23, -294952(%rbp)
	jmp	.L61
.L26:
	cmpl	$-1, -295004(%rbp)
	jne	.L68
	movq	$71, -294952(%rbp)
	jmp	.L61
.L68:
	movq	$6, -294952(%rbp)
	jmp	.L61
.L8:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L59:
	addl	$1, -295024(%rbp)
	movq	$25, -294952(%rbp)
	jmp	.L61
.L44:
	movl	-295044(%rbp), %eax
	cltq
	salq	$6, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -294936(%rbp)
	movq	$8, -294960(%rbp)
	movq	-294936(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %ecx
	movl	$0, %edx
	divq	%rcx
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L70:
	cmpq	%rdx, %rsp
	je	.L71
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L70
.L71:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L72
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L72:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -294968(%rbp)
	movq	$14, -294952(%rbp)
	jmp	.L61
.L10:
	addl	$1, -295032(%rbp)
	movq	$68, -294952(%rbp)
	jmp	.L61
.L43:
	cmpl	$0, -295036(%rbp)
	je	.L73
	movq	$0, -294952(%rbp)
	jmp	.L61
.L73:
	movq	$52, -294952(%rbp)
	jmp	.L61
.L45:
	cmpl	$-1, -295012(%rbp)
	jne	.L75
	movq	$10, -294952(%rbp)
	jmp	.L61
.L75:
	movq	$6, -294952(%rbp)
	jmp	.L61
.L11:
	movq	$63, -294952(%rbp)
	jmp	.L61
.L23:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L17:
	movl	-295032(%rbp), %eax
	cmpl	-295044(%rbp), %eax
	jge	.L77
	movq	$4, -294952(%rbp)
	jmp	.L61
.L77:
	movq	$49, -294952(%rbp)
	jmp	.L61
.L41:
	movl	$0, %edi
	call	wait@PLT
	addl	$1, -294988(%rbp)
	movq	$17, -294952(%rbp)
	jmp	.L61
.L51:
	cmpl	$-1, -294996(%rbp)
	jne	.L79
	movq	$57, -294952(%rbp)
	jmp	.L61
.L79:
	movq	$1, -294952(%rbp)
	jmp	.L61
.L53:
	cmpl	$-1, -295008(%rbp)
	jne	.L81
	movq	$30, -294952(%rbp)
	jmp	.L61
.L81:
	movq	$61, -294952(%rbp)
	jmp	.L61
.L49:
	cmpq	$0, -294976(%rbp)
	je	.L83
	movq	$18, -294952(%rbp)
	jmp	.L61
.L83:
	movq	$72, -294952(%rbp)
	jmp	.L61
.L20:
	call	getpid@PLT
	movl	%eax, pid(%rip)
	leaq	handle_sigterm(%rip), %rax
	movq	%rax, %rsi
	movl	$15, %edi
	call	signal@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	stdin(%rip), %rdx
	leaq	-32784(%rbp), %rax
	movl	$32768, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movl	$0, -295044(%rbp)
	movl	$0, -295040(%rbp)
	leaq	-32784(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -294944(%rbp)
	movq	-294944(%rbp), %rax
	movq	%rax, -294976(%rbp)
	movl	-295040(%rbp), %eax
	movslq	%eax, %rdx
	movl	-295044(%rbp), %eax
	cltq
	salq	$10, %rax
	addq	%rax, %rdx
	movq	-294976(%rbp), %rax
	movq	%rax, -294928(%rbp,%rdx,8)
	addl	$1, -295040(%rbp)
	movq	$72, -294952(%rbp)
	jmp	.L61
.L29:
	movl	-295024(%rbp), %eax
	cltq
	imulq	-294960(%rbp), %rax
	leaq	4(%rax), %rdx
	movq	-294968(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	$1, %esi
	movl	%eax, %edi
	call	dup2@PLT
	movl	%eax, -295016(%rbp)
	movq	$37, -294952(%rbp)
	jmp	.L61
.L38:
	movl	-294992(%rbp), %eax
	cmpl	-295044(%rbp), %eax
	jge	.L85
	movq	$44, -294952(%rbp)
	jmp	.L61
.L85:
	movq	$2, -294952(%rbp)
	jmp	.L61
.L47:
	movl	-294988(%rbp), %eax
	cmpl	-295044(%rbp), %eax
	jg	.L87
	movq	$26, -294952(%rbp)
	jmp	.L61
.L87:
	movq	$31, -294952(%rbp)
	jmp	.L61
.L34:
	movl	$0, -294992(%rbp)
	movq	$32, -294952(%rbp)
	jmp	.L61
.L18:
	cmpl	$0, -295044(%rbp)
	je	.L89
	movq	$8, -294952(%rbp)
	jmp	.L61
.L89:
	movq	$73, -294952(%rbp)
	jmp	.L61
.L25:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -294976(%rbp)
	movq	$13, -294952(%rbp)
	jmp	.L61
.L22:
	call	fork@PLT
	movl	%eax, -294984(%rbp)
	movl	-294984(%rbp), %eax
	movl	%eax, -295020(%rbp)
	movq	$64, -294952(%rbp)
	jmp	.L61
.L56:
	movl	$0, -295000(%rbp)
	movq	$56, -294952(%rbp)
	jmp	.L61
.L35:
	cmpl	$0, -295020(%rbp)
	jne	.L91
	movq	$67, -294952(%rbp)
	jmp	.L61
.L91:
	movq	$1, -294952(%rbp)
	jmp	.L61
.L21:
	movl	-295024(%rbp), %eax
	cltq
	imulq	-294960(%rbp), %rax
	leaq	4(%rax), %rdx
	movq	-294968(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	$1, %esi
	movl	%eax, %edi
	call	dup2@PLT
	movl	%eax, -295004(%rbp)
	movq	$54, -294952(%rbp)
	jmp	.L61
.L37:
	movl	-295024(%rbp), %eax
	cmpl	-295044(%rbp), %eax
	jne	.L93
	movq	$7, -294952(%rbp)
	jmp	.L61
.L93:
	movq	$69, -294952(%rbp)
	jmp	.L61
.L12:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L15:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L27:
	cmpl	$-1, -295028(%rbp)
	jne	.L95
	movq	$74, -294952(%rbp)
	jmp	.L61
.L95:
	movq	$77, -294952(%rbp)
	jmp	.L61
.L13:
	leaq	-294928(%rbp), %rdx
	movl	-295024(%rbp), %eax
	cltq
	salq	$13, %rax
	addq	%rax, %rdx
	movl	-295024(%rbp), %eax
	cltq
	salq	$13, %rax
	addq	%rbp, %rax
	subq	$294928, %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	execvp@PLT
	movl	%eax, -294996(%rbp)
	movq	$11, -294952(%rbp)
	jmp	.L61
.L33:
	movl	-294992(%rbp), %eax
	cltq
	imulq	-294960(%rbp), %rax
	movq	%rax, %rdx
	movq	-294968(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-294992(%rbp), %eax
	cltq
	imulq	-294960(%rbp), %rax
	leaq	4(%rax), %rdx
	movq	-294968(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	close@PLT
	addl	$1, -294992(%rbp)
	movq	$32, -294952(%rbp)
	jmp	.L61
.L14:
	cmpq	$0, -294976(%rbp)
	je	.L97
	movq	$55, -294952(%rbp)
	jmp	.L61
.L97:
	movq	$45, -294952(%rbp)
	jmp	.L61
.L36:
	cmpl	$-1, -295016(%rbp)
	jne	.L99
	movq	$12, -294952(%rbp)
	jmp	.L61
.L99:
	movq	$6, -294952(%rbp)
	jmp	.L61
.L19:
	cmpl	$-1, -295020(%rbp)
	jne	.L101
	movq	$78, -294952(%rbp)
	jmp	.L61
.L101:
	movq	$38, -294952(%rbp)
	jmp	.L61
.L52:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L60:
	movl	-295040(%rbp), %eax
	movslq	%eax, %rdx
	movl	-295044(%rbp), %eax
	cltq
	salq	$10, %rax
	addq	%rax, %rdx
	movq	-294976(%rbp), %rax
	movq	%rax, -294928(%rbp,%rdx,8)
	addl	$1, -295040(%rbp)
	movq	$72, -294952(%rbp)
	jmp	.L61
.L31:
	movl	-295000(%rbp), %eax
	cltq
	imulq	-294960(%rbp), %rax
	movq	%rax, %rdx
	movq	-294968(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-295000(%rbp), %eax
	cltq
	imulq	-294960(%rbp), %rax
	leaq	4(%rax), %rdx
	movq	-294968(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %edi
	call	close@PLT
	addl	$1, -295000(%rbp)
	movq	$56, -294952(%rbp)
	jmp	.L61
.L55:
	movl	-295024(%rbp), %eax
	subl	$1, %eax
	cltq
	imulq	-294960(%rbp), %rax
	movq	%rax, %rdx
	movq	-294968(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	$0, %esi
	movl	%eax, %edi
	call	dup2@PLT
	movl	%eax, -295012(%rbp)
	movq	$21, -294952(%rbp)
	jmp	.L61
.L58:
	movl	$0, -294988(%rbp)
	movq	$17, -294952(%rbp)
	jmp	.L61
.L105:
	nop
.L61:
	jmp	.L103
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.section	.rodata
.LC8:
	.string	"Program B was terminated..."
	.text
	.globl	handle_sigterm
	.type	handle_sigterm, @function
handle_sigterm:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L111:
	cmpq	$0, -8(%rbp)
	je	.L107
	cmpq	$2, -8(%rbp)
	je	.L108
	jmp	.L110
.L107:
	movq	$2, -8(%rbp)
	jmp	.L110
.L108:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	pid(%rip), %eax
	negl	%eax
	movl	$15, %esi
	movl	%eax, %edi
	call	kill@PLT
	movl	$0, %edi
	call	exit@PLT
.L110:
	jmp	.L111
	.cfi_endproc
.LFE7:
	.size	handle_sigterm, .-handle_sigterm
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
