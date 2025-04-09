	.file	"Edgarrosoir_dd_4_flatten.c"
	.text
	.globl	lesPavesY
	.bss
	.align 16
	.type	lesPavesY, @object
	.size	lesPavesY, 24
lesPavesY:
	.zero	24
	.globl	lesPavesX
	.align 16
	.type	lesPavesX, @object
	.size	lesPavesX, 24
lesPavesX:
	.zero	24
	.globl	_TIG_IZ_LnCg_envp
	.align 8
	.type	_TIG_IZ_LnCg_envp, @object
	.size	_TIG_IZ_LnCg_envp, 8
_TIG_IZ_LnCg_envp:
	.zero	8
	.globl	_TIG_IZ_LnCg_argc
	.align 4
	.type	_TIG_IZ_LnCg_argc, @object
	.size	_TIG_IZ_LnCg_argc, 4
_TIG_IZ_LnCg_argc:
	.zero	4
	.globl	_TIG_IZ_LnCg_argv
	.align 8
	.type	_TIG_IZ_LnCg_argv, @object
	.size	_TIG_IZ_LnCg_argv, 8
_TIG_IZ_LnCg_argv:
	.zero	8
	.globl	lesPommesY
	.align 32
	.type	lesPommesY, @object
	.size	lesPommesY, 40
lesPommesY:
	.zero	40
	.globl	lesPommesX
	.align 32
	.type	lesPommesX, @object
	.size	lesPommesX, 40
lesPommesX:
	.zero	40
	.section	.rodata
.LC0:
	.string	"\033[%d;%dH"
	.text
	.globl	gotoxy
	.type	gotoxy, @function
gotoxy:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	jmp	.L7
.L2:
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -8(%rbp)
	jmp	.L5
.L8:
	nop
.L5:
	jmp	.L6
.L7:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	gotoxy, .-gotoxy
	.globl	effacer
	.type	effacer, @function
effacer:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L15:
	cmpq	$2, -8(%rbp)
	je	.L16
	cmpq	$2, -8(%rbp)
	ja	.L17
	cmpq	$0, -8(%rbp)
	je	.L12
	cmpq	$1, -8(%rbp)
	jne	.L17
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	gotoxy
	movl	$32, %edi
	call	putchar@PLT
	movl	$1, %esi
	movl	$1, %edi
	call	gotoxy
	movq	$2, -8(%rbp)
	jmp	.L13
.L12:
	movq	$1, -8(%rbp)
	jmp	.L13
.L17:
	nop
.L13:
	jmp	.L15
.L16:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	effacer, .-effacer
	.globl	kbhit
	.type	kbhit, @function
kbhit:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -152(%rbp)
.L31:
	cmpq	$5, -152(%rbp)
	ja	.L34
	movq	-152(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L21(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L21(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L21:
	.long	.L26-.L21
	.long	.L25-.L21
	.long	.L24-.L21
	.long	.L23-.L21
	.long	.L22-.L21
	.long	.L20-.L21
	.text
.L22:
	cmpl	$-1, -160(%rbp)
	je	.L27
	movq	$3, -152(%rbp)
	jmp	.L29
.L27:
	movq	$5, -152(%rbp)
	jmp	.L29
.L25:
	movl	$1, %eax
	jmp	.L32
.L23:
	movq	stdin(%rip), %rdx
	movl	-160(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	ungetc@PLT
	movq	$1, -152(%rbp)
	jmp	.L29
.L20:
	movl	$0, %eax
	jmp	.L32
.L26:
	movq	$2, -152(%rbp)
	jmp	.L29
.L24:
	leaq	-144(%rbp), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	tcgetattr@PLT
	movq	-144(%rbp), %rax
	movq	-136(%rbp), %rdx
	movq	%rax, -80(%rbp)
	movq	%rdx, -72(%rbp)
	movq	-128(%rbp), %rax
	movq	-120(%rbp), %rdx
	movq	%rax, -64(%rbp)
	movq	%rdx, -56(%rbp)
	movq	-112(%rbp), %rax
	movq	-104(%rbp), %rdx
	movq	%rax, -48(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-96(%rbp), %rax
	movq	%rax, -32(%rbp)
	movl	-88(%rbp), %eax
	movl	%eax, -24(%rbp)
	movl	-68(%rbp), %eax
	andl	$-11, %eax
	movl	%eax, -68(%rbp)
	leaq	-80(%rbp), %rax
	movq	%rax, %rdx
	movl	$0, %esi
	movl	$0, %edi
	call	tcsetattr@PLT
	movl	$0, %edx
	movl	$3, %esi
	movl	$0, %edi
	movl	$0, %eax
	call	fcntl@PLT
	movl	%eax, -156(%rbp)
	movl	-156(%rbp), %eax
	orb	$8, %ah
	movl	%eax, %edx
	movl	$4, %esi
	movl	$0, %edi
	movl	$0, %eax
	call	fcntl@PLT
	call	getchar@PLT
	movl	%eax, -160(%rbp)
	leaq	-144(%rbp), %rax
	movq	%rax, %rdx
	movl	$0, %esi
	movl	$0, %edi
	call	tcsetattr@PLT
	movl	-156(%rbp), %eax
	movl	%eax, %edx
	movl	$4, %esi
	movl	$0, %edi
	movl	$0, %eax
	call	fcntl@PLT
	movq	$4, -152(%rbp)
	jmp	.L29
.L34:
	nop
.L29:
	jmp	.L31
.L32:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	call	__stack_chk_fail@PLT
.L33:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	kbhit, .-kbhit
	.globl	calculerDirection
	.type	calculerDirection, @function
calculerDirection:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -36(%rbp)
	movl	%esi, -40(%rbp)
	movl	%edx, -44(%rbp)
	movl	%ecx, -48(%rbp)
	movl	%r8d, %eax
	movq	%r9, -64(%rbp)
	movb	%al, -52(%rbp)
	movq	$45, -8(%rbp)
.L137:
	cmpq	$47, -8(%rbp)
	ja	.L138
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L38(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L38(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L38:
	.long	.L76-.L38
	.long	.L94-.L38
	.long	.L74-.L38
	.long	.L138-.L38
	.long	.L73-.L38
	.long	.L72-.L38
	.long	.L71-.L38
	.long	.L70-.L38
	.long	.L138-.L38
	.long	.L102-.L38
	.long	.L68-.L38
	.long	.L138-.L38
	.long	.L91-.L38
	.long	.L66-.L38
	.long	.L65-.L38
	.long	.L86-.L38
	.long	.L63-.L38
	.long	.L62-.L38
	.long	.L61-.L38
	.long	.L60-.L38
	.long	.L59-.L38
	.long	.L58-.L38
	.long	.L138-.L38
	.long	.L138-.L38
	.long	.L138-.L38
	.long	.L57-.L38
	.long	.L56-.L38
	.long	.L55-.L38
	.long	.L54-.L38
	.long	.L138-.L38
	.long	.L53-.L38
	.long	.L52-.L38
	.long	.L51-.L38
	.long	.L50-.L38
	.long	.L49-.L38
	.long	.L48-.L38
	.long	.L47-.L38
	.long	.L138-.L38
	.long	.L46-.L38
	.long	.L138-.L38
	.long	.L45-.L38
	.long	.L44-.L38
	.long	.L43-.L38
	.long	.L42-.L38
	.long	.L41-.L38
	.long	.L40-.L38
	.long	.L39-.L38
	.long	.L37-.L38
	.text
.L61:
	movl	-24(%rbp), %eax
	movl	%eax, %edx
	negl	%edx
	cmovns	%edx, %eax
	movl	%eax, -16(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %edx
	negl	%edx
	cmovns	%edx, %eax
	movl	%eax, -12(%rbp)
	movq	$25, -8(%rbp)
	jmp	.L77
.L57:
	movl	-16(%rbp), %eax
	cmpl	-12(%rbp), %eax
	jle	.L78
	movq	$21, -8(%rbp)
	jmp	.L77
.L78:
	movq	$13, -8(%rbp)
	jmp	.L77
.L73:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	subl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L80
	movq	$41, -8(%rbp)
	jmp	.L77
.L80:
	movq	$40, -8(%rbp)
	jmp	.L77
.L53:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rdx, %rax
	leaq	-41(%rax), %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L82
	movq	$35, -8(%rbp)
	jmp	.L77
.L82:
	movq	$19, -8(%rbp)
	jmp	.L77
.L65:
	cmpl	$0, -20(%rbp)
	jne	.L84
	movq	$2, -8(%rbp)
	jmp	.L77
.L84:
	movq	$19, -8(%rbp)
	jmp	.L77
.L64:
.L86:
	cmpl	$0, -24(%rbp)
	jns	.L87
	movq	$31, -8(%rbp)
	jmp	.L77
.L87:
	movq	$19, -8(%rbp)
	jmp	.L77
.L52:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rdx, %rax
	leaq	-41(%rax), %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L89
	movq	$30, -8(%rbp)
	jmp	.L77
.L89:
	movq	$19, -8(%rbp)
	jmp	.L77
.L67:
.L91:
	cmpl	$0, -20(%rbp)
	jns	.L92
	movq	$34, -8(%rbp)
	jmp	.L77
.L92:
	movq	$40, -8(%rbp)
	jmp	.L77
.L40:
	movq	$36, -8(%rbp)
	jmp	.L77
.L75:
.L94:
	cmpl	$0, -24(%rbp)
	jns	.L95
	movq	$7, -8(%rbp)
	jmp	.L77
.L95:
	movq	$40, -8(%rbp)
	jmp	.L77
.L63:
	movl	$115, %eax
	jmp	.L97
.L58:
	cmpl	$0, -24(%rbp)
	jle	.L98
	movq	$6, -8(%rbp)
	jmp	.L77
.L98:
	movq	$1, -8(%rbp)
	jmp	.L77
.L47:
	movl	-44(%rbp), %eax
	subl	-36(%rbp), %eax
	movl	%eax, -24(%rbp)
	movl	-48(%rbp), %eax
	subl	-40(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L77
.L56:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	addl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L100
	movq	$28, -8(%rbp)
	jmp	.L77
.L100:
	movq	$12, -8(%rbp)
	jmp	.L77
.L69:
.L102:
	cmpl	$0, -20(%rbp)
	jns	.L103
	movq	$20, -8(%rbp)
	jmp	.L77
.L103:
	movq	$18, -8(%rbp)
	jmp	.L77
.L66:
	cmpl	$0, -20(%rbp)
	jle	.L105
	movq	$38, -8(%rbp)
	jmp	.L77
.L105:
	movq	$12, -8(%rbp)
	jmp	.L77
.L60:
	cmpl	$0, -24(%rbp)
	jne	.L107
	movq	$43, -8(%rbp)
	jmp	.L77
.L107:
	movq	$18, -8(%rbp)
	jmp	.L77
.L51:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	addl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L109
	movq	$16, -8(%rbp)
	jmp	.L77
.L109:
	movq	$9, -8(%rbp)
	jmp	.L77
.L62:
	movl	-36(%rbp), %eax
	cltq
	leaq	1(%rax), %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L111
	movq	$5, -8(%rbp)
	jmp	.L77
.L111:
	movq	$15, -8(%rbp)
	jmp	.L77
.L45:
	movzbl	-52(%rbp), %eax
	jmp	.L97
.L71:
	movl	-36(%rbp), %eax
	cltq
	leaq	1(%rax), %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L113
	movq	$42, -8(%rbp)
	jmp	.L77
.L113:
	movq	$1, -8(%rbp)
	jmp	.L77
.L55:
	movl	$113, %eax
	jmp	.L97
.L46:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	addl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L115
	movq	$26, -8(%rbp)
	jmp	.L77
.L115:
	movq	$12, -8(%rbp)
	jmp	.L77
.L49:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	subl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L117
	movq	$4, -8(%rbp)
	jmp	.L77
.L117:
	movq	$40, -8(%rbp)
	jmp	.L77
.L54:
	movl	$115, %eax
	jmp	.L97
.L37:
	movl	-36(%rbp), %eax
	cltq
	leaq	1(%rax), %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L119
	movq	$17, -8(%rbp)
	jmp	.L77
.L119:
	movq	$15, -8(%rbp)
	jmp	.L77
.L41:
	movl	$100, %eax
	jmp	.L97
.L72:
	movl	$100, %eax
	jmp	.L97
.L50:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	subl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L121
	movq	$46, -8(%rbp)
	jmp	.L77
.L121:
	movq	$18, -8(%rbp)
	jmp	.L77
.L44:
	movl	$122, %eax
	jmp	.L97
.L68:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rdx, %rax
	leaq	-41(%rax), %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L123
	movq	$27, -8(%rbp)
	jmp	.L77
.L123:
	movq	$40, -8(%rbp)
	jmp	.L77
.L43:
	movl	-36(%rbp), %eax
	cltq
	leaq	1(%rax), %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L125
	movq	$44, -8(%rbp)
	jmp	.L77
.L125:
	movq	$1, -8(%rbp)
	jmp	.L77
.L76:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	addl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L127
	movq	$32, -8(%rbp)
	jmp	.L77
.L127:
	movq	$9, -8(%rbp)
	jmp	.L77
.L39:
	movl	$122, %eax
	jmp	.L97
.L70:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rdx, %rax
	leaq	-41(%rax), %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L129
	movq	$10, -8(%rbp)
	jmp	.L77
.L129:
	movq	$40, -8(%rbp)
	jmp	.L77
.L48:
	movl	$113, %eax
	jmp	.L97
.L42:
	cmpl	$0, -20(%rbp)
	jle	.L131
	movq	$0, -8(%rbp)
	jmp	.L77
.L131:
	movq	$9, -8(%rbp)
	jmp	.L77
.L74:
	cmpl	$0, -24(%rbp)
	jle	.L133
	movq	$47, -8(%rbp)
	jmp	.L77
.L133:
	movq	$15, -8(%rbp)
	jmp	.L77
.L59:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	addq	%rax, %rdx
	movl	-40(%rbp), %eax
	subl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L135
	movq	$33, -8(%rbp)
	jmp	.L77
.L135:
	movq	$18, -8(%rbp)
	jmp	.L77
.L138:
	nop
.L77:
	jmp	.L137
.L97:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	calculerDirection, .-calculerDirection
	.section	.rodata
.LC1:
	.string	"clear"
	.align 8
.LC2:
	.string	"Nombre de d\303\251placements : %d caract\303\250res.\n"
.LC4:
	.string	"Temps CPU = %.2f secondes.\n"
	.align 8
.LC5:
	.string	"Partie termin\303\251e. Pommes mang\303\251es : %d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-28672(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$400, %rsp
	movl	%edi, -29044(%rbp)
	movq	%rsi, -29056(%rbp)
	movq	%rdx, -29064(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$8, lesPommesY(%rip)
	movl	$39, 4+lesPommesY(%rip)
	movl	$2, 8+lesPommesY(%rip)
	movl	$2, 12+lesPommesY(%rip)
	movl	$5, 16+lesPommesY(%rip)
	movl	$39, 20+lesPommesY(%rip)
	movl	$33, 24+lesPommesY(%rip)
	movl	$38, 28+lesPommesY(%rip)
	movl	$35, 32+lesPommesY(%rip)
	movl	$2, 36+lesPommesY(%rip)
	nop
.L140:
	movl	$75, lesPommesX(%rip)
	movl	$75, 4+lesPommesX(%rip)
	movl	$78, 8+lesPommesX(%rip)
	movl	$2, 12+lesPommesX(%rip)
	movl	$8, 16+lesPommesX(%rip)
	movl	$78, 20+lesPommesX(%rip)
	movl	$74, 24+lesPommesX(%rip)
	movl	$2, 28+lesPommesX(%rip)
	movl	$72, 32+lesPommesX(%rip)
	movl	$5, 36+lesPommesX(%rip)
	nop
.L141:
	movl	$3, lesPavesY(%rip)
	movl	$3, 4+lesPavesY(%rip)
	movl	$34, 8+lesPavesY(%rip)
	movl	$34, 12+lesPavesY(%rip)
	movl	$21, 16+lesPavesY(%rip)
	movl	$15, 20+lesPavesY(%rip)
	nop
.L142:
	movl	$3, lesPavesX(%rip)
	movl	$74, 4+lesPavesX(%rip)
	movl	$3, 8+lesPavesX(%rip)
	movl	$74, 12+lesPavesX(%rip)
	movl	$38, 16+lesPavesX(%rip)
	movl	$38, 20+lesPavesX(%rip)
	nop
.L143:
	movq	$0, _TIG_IZ_LnCg_envp(%rip)
	nop
.L144:
	movq	$0, _TIG_IZ_LnCg_argv(%rip)
	nop
.L145:
	movl	$0, _TIG_IZ_LnCg_argc(%rip)
	nop
	nop
.L146:
.L147:
#APP
# 642 "Edgarrosoir_dd_4.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-LnCg--0
# 0 "" 2
#NO_APP
	movl	-29044(%rbp), %eax
	movl	%eax, _TIG_IZ_LnCg_argc(%rip)
	movq	-29056(%rbp), %rax
	movq	%rax, _TIG_IZ_LnCg_argv(%rip)
	movq	-29064(%rbp), %rax
	movq	%rax, _TIG_IZ_LnCg_envp(%rip)
	nop
	movq	$0, -28992(%rbp)
.L182:
	cmpq	$27, -28992(%rbp)
	ja	.L185
	movq	-28992(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L150(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L150(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L150:
	.long	.L167-.L150
	.long	.L185-.L150
	.long	.L185-.L150
	.long	.L185-.L150
	.long	.L166-.L150
	.long	.L165-.L150
	.long	.L164-.L150
	.long	.L185-.L150
	.long	.L163-.L150
	.long	.L185-.L150
	.long	.L162-.L150
	.long	.L161-.L150
	.long	.L160-.L150
	.long	.L159-.L150
	.long	.L185-.L150
	.long	.L158-.L150
	.long	.L157-.L150
	.long	.L185-.L150
	.long	.L156-.L150
	.long	.L185-.L150
	.long	.L155-.L150
	.long	.L154-.L150
	.long	.L153-.L150
	.long	.L152-.L150
	.long	.L151-.L150
	.long	.L185-.L150
	.long	.L185-.L150
	.long	.L149-.L150
	.text
.L156:
	movl	-29024(%rbp), %eax
	cmpl	%eax, -29012(%rbp)
	jge	.L168
	movq	$4, -28992(%rbp)
	jmp	.L170
.L168:
	movq	$11, -28992(%rbp)
	jmp	.L170
.L166:
	movl	$40, %eax
	subl	-29012(%rbp), %eax
	movl	%eax, %edx
	movl	-29012(%rbp), %eax
	cltq
	movl	%edx, -25616(%rbp,%rax,4)
	movl	-29012(%rbp), %eax
	cltq
	movl	$20, -12816(%rbp,%rax,4)
	addl	$1, -29012(%rbp)
	movq	$18, -28992(%rbp)
	jmp	.L170
.L158:
	movl	$200000, %edi
	call	usleep@PLT
	movq	$22, -28992(%rbp)
	jmp	.L170
.L160:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L183
	jmp	.L184
.L163:
	cmpb	$97, -29025(%rbp)
	jne	.L172
	movq	$6, -28992(%rbp)
	jmp	.L170
.L172:
	movq	$10, -28992(%rbp)
	jmp	.L170
.L152:
	addl	$1, -29020(%rbp)
	movq	$21, -28992(%rbp)
	jmp	.L170
.L157:
	call	getchar@PLT
	movl	%eax, -29004(%rbp)
	movl	-29004(%rbp), %eax
	movb	%al, -29025(%rbp)
	movq	$8, -28992(%rbp)
	jmp	.L170
.L151:
	movb	$100, -29028(%rbp)
	movb	$0, -29027(%rbp)
	movb	$0, -29026(%rbp)
	movl	$10, -29024(%rbp)
	movl	$0, -29020(%rbp)
	movl	$0, -29016(%rbp)
	call	clock@PLT
	movq	%rax, -28984(%rbp)
	movq	-28984(%rbp), %rax
	movq	%rax, -29000(%rbp)
	movl	$0, -29012(%rbp)
	movq	$18, -28992(%rbp)
	jmp	.L170
.L154:
	cmpl	$9, -29020(%rbp)
	jg	.L174
	movq	$5, -28992(%rbp)
	jmp	.L170
.L174:
	movq	$15, -28992(%rbp)
	jmp	.L170
.L161:
	leaq	-28944(%rbp), %rax
	movq	%rax, %rdi
	call	initPlateau
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	system@PLT
	leaq	-28944(%rbp), %rax
	movq	%rax, %rdi
	call	dessinerPlateau
	movl	-29020(%rbp), %edx
	leaq	-28944(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	ajouterPomme
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -28976(%rbp)
	movq	-28976(%rbp), %rax
	movl	%eax, %edi
	call	srand@PLT
	movq	$22, -28992(%rbp)
	jmp	.L170
.L159:
	cmpl	$0, -29008(%rbp)
	je	.L176
	movq	$16, -28992(%rbp)
	jmp	.L170
.L176:
	movq	$10, -28992(%rbp)
	jmp	.L170
.L164:
	movl	$42, %esi
	movl	$1, %edi
	call	gotoxy
	movl	-29016(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	clock@PLT
	movq	%rax, -28968(%rbp)
	movq	-28968(%rbp), %rax
	movq	%rax, -28960(%rbp)
	movq	-28960(%rbp), %rax
	subq	-29000(%rbp), %rax
	pxor	%xmm0, %xmm0
	cvtsi2sdq	%rax, %xmm0
	movsd	.LC3(%rip), %xmm1
	divsd	%xmm1, %xmm0
	movsd	%xmm0, -28952(%rbp)
	movq	-28952(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movl	-29020(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -28992(%rbp)
	jmp	.L170
.L149:
	call	kbhit
	movl	%eax, -29008(%rbp)
	movq	$13, -28992(%rbp)
	jmp	.L170
.L153:
	movzbl	-29027(%rbp), %eax
	xorl	$1, %eax
	movzbl	%al, %edx
	cmpl	$9, -29020(%rbp)
	setle	%al
	movzbl	%al, %eax
	andl	%edx, %eax
	testl	%eax, %eax
	je	.L178
	movq	$27, -28992(%rbp)
	jmp	.L170
.L178:
	movq	$6, -28992(%rbp)
	jmp	.L170
.L165:
	movl	-29020(%rbp), %edx
	leaq	-28944(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	ajouterPomme
	movq	$15, -28992(%rbp)
	jmp	.L170
.L162:
	movl	-29020(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	lesPommesY(%rip), %rax
	movl	(%rdx,%rax), %r8d
	movl	-29020(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	lesPommesX(%rip), %rax
	movl	(%rdx,%rax), %edi
	leaq	-29027(%rbp), %r9
	leaq	-28944(%rbp), %r10
	leaq	-29028(%rbp), %rcx
	leaq	-29024(%rbp), %rdx
	leaq	-12816(%rbp), %rsi
	leaq	-25616(%rbp), %rax
	subq	$8, %rsp
	pushq	%r8
	pushq	%rdi
	leaq	-29026(%rbp), %rdi
	pushq	%rdi
	movq	%r10, %r8
	movq	%rax, %rdi
	call	progresser
	addq	$32, %rsp
	addl	$1, -29016(%rbp)
	movq	$20, -28992(%rbp)
	jmp	.L170
.L167:
	movq	$24, -28992(%rbp)
	jmp	.L170
.L155:
	movzbl	-29026(%rbp), %eax
	testb	%al, %al
	je	.L180
	movq	$23, -28992(%rbp)
	jmp	.L170
.L180:
	movq	$15, -28992(%rbp)
	jmp	.L170
.L185:
	nop
.L170:
	jmp	.L182
.L184:
	call	__stack_chk_fail@PLT
.L183:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.globl	initPlateau
	.type	initPlateau, @function
initPlateau:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -56(%rbp)
	movq	$30, -8(%rbp)
.L227:
	cmpq	$37, -8(%rbp)
	ja	.L228
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L189(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L189(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L189:
	.long	.L210-.L189
	.long	.L209-.L189
	.long	.L228-.L189
	.long	.L208-.L189
	.long	.L207-.L189
	.long	.L228-.L189
	.long	.L206-.L189
	.long	.L205-.L189
	.long	.L228-.L189
	.long	.L204-.L189
	.long	.L203-.L189
	.long	.L228-.L189
	.long	.L202-.L189
	.long	.L228-.L189
	.long	.L228-.L189
	.long	.L201-.L189
	.long	.L228-.L189
	.long	.L200-.L189
	.long	.L228-.L189
	.long	.L199-.L189
	.long	.L198-.L189
	.long	.L228-.L189
	.long	.L228-.L189
	.long	.L228-.L189
	.long	.L197-.L189
	.long	.L196-.L189
	.long	.L195-.L189
	.long	.L194-.L189
	.long	.L228-.L189
	.long	.L228-.L189
	.long	.L193-.L189
	.long	.L192-.L189
	.long	.L228-.L189
	.long	.L228-.L189
	.long	.L228-.L189
	.long	.L191-.L189
	.long	.L190-.L189
	.long	.L229-.L189
	.text
.L196:
	movl	$1, -32(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L211
.L207:
	movl	$0, -12(%rbp)
	movq	$35, -8(%rbp)
	jmp	.L211
.L193:
	movl	$1, -36(%rbp)
	movq	$36, -8(%rbp)
	jmp	.L211
.L201:
	addl	$1, -36(%rbp)
	movq	$36, -8(%rbp)
	jmp	.L211
.L192:
	movb	$35, -38(%rbp)
	movq	-56(%rbp), %rax
	leaq	3280(%rax), %rcx
	movl	-24(%rbp), %eax
	cltq
	movzbl	-38(%rbp), %edx
	movb	%dl, (%rcx,%rax)
	movq	-56(%rbp), %rax
	leaq	41(%rax), %rcx
	movl	-24(%rbp), %eax
	cltq
	movzbl	-38(%rbp), %edx
	movb	%dl, (%rcx,%rax)
	addl	$1, -24(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L211
.L202:
	movl	-36(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	-32(%rbp), %eax
	cltq
	movb	$32, (%rdx,%rax)
	addl	$1, -32(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L211
.L209:
	movl	$1, -24(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L211
.L208:
	movl	$1, -28(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L211
.L197:
	addl	$1, -16(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L211
.L190:
	cmpl	$80, -36(%rbp)
	jg	.L212
	movq	$25, -8(%rbp)
	jmp	.L211
.L212:
	movq	$3, -8(%rbp)
	jmp	.L211
.L195:
	movb	$35, -37(%rbp)
	movl	-28(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-37(%rbp), %eax
	movb	%al, 40(%rdx)
	movl	-28(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movzbl	-37(%rbp), %eax
	movb	%al, 1(%rdx)
	addl	$1, -28(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L211
.L204:
	cmpl	$40, -32(%rbp)
	jg	.L214
	movq	$12, -8(%rbp)
	jmp	.L211
.L214:
	movq	$15, -8(%rbp)
	jmp	.L211
.L199:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	lesPavesX(%rip), %rax
	movl	(%rdx,%rax), %edx
	movl	-16(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	leaq	lesPavesY(%rip), %rax
	movl	(%rcx,%rax), %ecx
	movl	-12(%rbp), %eax
	addl	%ecx, %eax
	cltq
	movb	$35, (%rdx,%rax)
	addl	$1, -12(%rbp)
	movq	$35, -8(%rbp)
	jmp	.L211
.L200:
	movq	-56(%rbp), %rax
	addq	$1640, %rax
	movb	$32, 1(%rax)
	movq	-56(%rbp), %rax
	addq	$1640, %rax
	movb	$32, 40(%rax)
	movq	-56(%rbp), %rax
	addq	$41, %rax
	movb	$32, 20(%rax)
	movq	-56(%rbp), %rax
	addq	$3280, %rax
	movb	$32, 20(%rax)
	movl	$0, -20(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L211
.L206:
	cmpl	$80, -28(%rbp)
	jg	.L216
	movq	$26, -8(%rbp)
	jmp	.L211
.L216:
	movq	$1, -8(%rbp)
	jmp	.L211
.L194:
	addl	$1, -20(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L211
.L203:
	cmpl	$40, -24(%rbp)
	jg	.L219
	movq	$31, -8(%rbp)
	jmp	.L211
.L219:
	movq	$17, -8(%rbp)
	jmp	.L211
.L210:
	cmpl	$5, -20(%rbp)
	jg	.L221
	movq	$7, -8(%rbp)
	jmp	.L211
.L221:
	movq	$37, -8(%rbp)
	jmp	.L211
.L205:
	movl	$0, -16(%rbp)
	movq	$20, -8(%rbp)
	jmp	.L211
.L191:
	cmpl	$4, -12(%rbp)
	jg	.L223
	movq	$19, -8(%rbp)
	jmp	.L211
.L223:
	movq	$24, -8(%rbp)
	jmp	.L211
.L198:
	cmpl	$4, -16(%rbp)
	jg	.L225
	movq	$4, -8(%rbp)
	jmp	.L211
.L225:
	movq	$27, -8(%rbp)
	jmp	.L211
.L228:
	nop
.L211:
	jmp	.L227
.L229:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	initPlateau, .-initPlateau
	.globl	dessinerSerpent
	.type	dessinerSerpent, @function
dessinerSerpent:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movl	%edx, -36(%rbp)
	movq	$2, -8(%rbp)
.L242:
	cmpq	$7, -8(%rbp)
	ja	.L243
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L233(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L233(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L233:
	.long	.L237-.L233
	.long	.L243-.L233
	.long	.L236-.L233
	.long	.L243-.L233
	.long	.L244-.L233
	.long	.L243-.L233
	.long	.L234-.L233
	.long	.L232-.L233
	.text
.L234:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	$88, %edx
	movl	%ecx, %esi
	movl	%eax, %edi
	call	afficher
	addl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L239
.L237:
	movq	-32(%rbp), %rax
	movl	(%rax), %ecx
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	$79, %edx
	movl	%ecx, %esi
	movl	%eax, %edi
	call	afficher
	movq	$4, -8(%rbp)
	jmp	.L239
.L232:
	movl	-12(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jge	.L240
	movq	$6, -8(%rbp)
	jmp	.L239
.L240:
	movq	$0, -8(%rbp)
	jmp	.L239
.L236:
	movl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L239
.L243:
	nop
.L239:
	jmp	.L242
.L244:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	dessinerSerpent, .-dessinerSerpent
	.globl	ajouterPomme
	.type	ajouterPomme, @function
ajouterPomme:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$2, -8(%rbp)
.L251:
	cmpq	$2, -8(%rbp)
	je	.L246
	cmpq	$2, -8(%rbp)
	ja	.L252
	cmpq	$0, -8(%rbp)
	je	.L253
	cmpq	$1, -8(%rbp)
	jne	.L252
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	lesPommesX(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -16(%rbp)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	lesPommesY(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, -12(%rbp)
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rax, %rdx
	movl	-12(%rbp), %eax
	cltq
	movb	$54, (%rdx,%rax)
	movl	-12(%rbp), %ecx
	movl	-16(%rbp), %eax
	movl	$54, %edx
	movl	%ecx, %esi
	movl	%eax, %edi
	call	afficher
	movq	$0, -8(%rbp)
	jmp	.L249
.L246:
	movq	$1, -8(%rbp)
	jmp	.L249
.L252:
	nop
.L249:
	jmp	.L251
.L253:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	ajouterPomme, .-ajouterPomme
	.globl	progresser
	.type	progresser, @function
progresser:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%rcx, -64(%rbp)
	movq	%r8, -72(%rbp)
	movq	%r9, -80(%rbp)
	movq	$10, -8(%rbp)
.L399:
	movq	-8(%rbp), %rax
	subq	$3, %rax
	cmpq	$87, %rax
	ja	.L400
	leaq	0(,%rax,4), %rdx
	leaq	.L257(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L257(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L257:
	.long	.L324-.L257
	.long	.L323-.L257
	.long	.L322-.L257
	.long	.L321-.L257
	.long	.L400-.L257
	.long	.L400-.L257
	.long	.L320-.L257
	.long	.L319-.L257
	.long	.L318-.L257
	.long	.L317-.L257
	.long	.L316-.L257
	.long	.L400-.L257
	.long	.L315-.L257
	.long	.L314-.L257
	.long	.L313-.L257
	.long	.L312-.L257
	.long	.L311-.L257
	.long	.L310-.L257
	.long	.L309-.L257
	.long	.L308-.L257
	.long	.L307-.L257
	.long	.L306-.L257
	.long	.L305-.L257
	.long	.L304-.L257
	.long	.L303-.L257
	.long	.L302-.L257
	.long	.L301-.L257
	.long	.L300-.L257
	.long	.L299-.L257
	.long	.L400-.L257
	.long	.L298-.L257
	.long	.L297-.L257
	.long	.L400-.L257
	.long	.L296-.L257
	.long	.L400-.L257
	.long	.L295-.L257
	.long	.L400-.L257
	.long	.L400-.L257
	.long	.L294-.L257
	.long	.L293-.L257
	.long	.L292-.L257
	.long	.L400-.L257
	.long	.L291-.L257
	.long	.L290-.L257
	.long	.L289-.L257
	.long	.L288-.L257
	.long	.L287-.L257
	.long	.L286-.L257
	.long	.L400-.L257
	.long	.L285-.L257
	.long	.L284-.L257
	.long	.L283-.L257
	.long	.L400-.L257
	.long	.L282-.L257
	.long	.L281-.L257
	.long	.L280-.L257
	.long	.L279-.L257
	.long	.L278-.L257
	.long	.L277-.L257
	.long	.L276-.L257
	.long	.L275-.L257
	.long	.L400-.L257
	.long	.L274-.L257
	.long	.L273-.L257
	.long	.L272-.L257
	.long	.L271-.L257
	.long	.L270-.L257
	.long	.L400-.L257
	.long	.L269-.L257
	.long	.L268-.L257
	.long	.L267-.L257
	.long	.L400-.L257
	.long	.L266-.L257
	.long	.L400-.L257
	.long	.L265-.L257
	.long	.L264-.L257
	.long	.L263-.L257
	.long	.L400-.L257
	.long	.L400-.L257
	.long	.L400-.L257
	.long	.L400-.L257
	.long	.L262-.L257
	.long	.L261-.L257
	.long	.L400-.L257
	.long	.L260-.L257
	.long	.L259-.L257
	.long	.L258-.L257
	.long	.L401-.L257
	.text
.L312:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	subl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L325
	movq	$25, -8(%rbp)
	jmp	.L327
.L325:
	movq	$34, -8(%rbp)
	jmp	.L327
.L286:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$40, %eax
	jne	.L328
	movq	$61, -8(%rbp)
	jmp	.L327
.L328:
	movq	$56, -8(%rbp)
	jmp	.L327
.L305:
	movb	$122, -25(%rbp)
	movq	$49, -8(%rbp)
	jmp	.L327
.L287:
	movq	-64(%rbp), %rax
	movzbl	-25(%rbp), %edx
	movb	%dl, (%rax)
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -20(%rbp)
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -16(%rbp)
	movq	$78, -8(%rbp)
	jmp	.L327
.L285:
	movq	-40(%rbp), %rax
	movl	$0, (%rax)
	movq	$31, -8(%rbp)
	jmp	.L327
.L323:
	movq	$58, -8(%rbp)
	jmp	.L327
.L300:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cltq
	leaq	1(%rax), %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L330
	movq	$75, -8(%rbp)
	jmp	.L327
.L330:
	movq	$49, -8(%rbp)
	jmp	.L327
.L276:
	movb	$115, -25(%rbp)
	movq	$49, -8(%rbp)
	jmp	.L327
.L315:
	movq	$22, -8(%rbp)
	jmp	.L327
.L258:
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	effacer
	movq	-56(%rbp), %rax
	movl	(%rax), %eax
	subl	$1, %eax
	movl	%eax, -24(%rbp)
	movq	$43, -8(%rbp)
	jmp	.L327
.L282:
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$20, %eax
	jne	.L332
	movq	$79, -8(%rbp)
	jmp	.L327
.L332:
	movq	$53, -8(%rbp)
	jmp	.L327
.L263:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jne	.L334
	movq	$5, -8(%rbp)
	jmp	.L327
.L334:
	movq	$53, -8(%rbp)
	jmp	.L327
.L299:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jg	.L336
	movq	$20, -8(%rbp)
	jmp	.L327
.L336:
	movq	$27, -8(%rbp)
	jmp	.L327
.L317:
	movl	$1, -12(%rbp)
	movq	$59, -8(%rbp)
	jmp	.L327
.L270:
	movq	-48(%rbp), %rax
	movl	$40, (%rax)
	movq	$31, -8(%rbp)
	jmp	.L327
.L291:
	movb	$113, -25(%rbp)
	movq	$49, -8(%rbp)
	jmp	.L327
.L283:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	addl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L338
	movq	$62, -8(%rbp)
	jmp	.L327
.L338:
	movq	$49, -8(%rbp)
	jmp	.L327
.L264:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$122, %eax
	je	.L340
	cmpl	$122, %eax
	jg	.L341
	cmpl	$115, %eax
	je	.L342
	cmpl	$115, %eax
	jg	.L341
	cmpl	$100, %eax
	je	.L343
	cmpl	$113, %eax
	je	.L344
	jmp	.L341
.L343:
	movq	$67, -8(%rbp)
	jmp	.L345
.L344:
	movq	$9, -8(%rbp)
	jmp	.L345
.L342:
	movq	$13, -8(%rbp)
	jmp	.L345
.L340:
	movq	$85, -8(%rbp)
	jmp	.L345
.L341:
	movq	$4, -8(%rbp)
	nop
.L345:
	jmp	.L327
.L307:
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$40, %eax
	jle	.L346
	movq	$46, -8(%rbp)
	jmp	.L327
.L346:
	movq	$6, -8(%rbp)
	jmp	.L327
.L265:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	jne	.L348
	movq	$12, -8(%rbp)
	jmp	.L327
.L348:
	movq	$29, -8(%rbp)
	jmp	.L327
.L324:
	movq	-48(%rbp), %rax
	movl	$40, (%rax)
	movq	$23, -8(%rbp)
	jmp	.L327
.L314:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$122, %al
	jne	.L350
	movq	$48, -8(%rbp)
	jmp	.L327
.L350:
	movq	$66, -8(%rbp)
	jmp	.L327
.L306:
	subl	$1, -16(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L327
.L309:
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jg	.L352
	movq	$3, -8(%rbp)
	jmp	.L327
.L352:
	movq	$23, -8(%rbp)
	jmp	.L327
.L296:
	movq	-48(%rbp), %rax
	movl	$0, (%rax)
	movq	$31, -8(%rbp)
	jmp	.L327
.L281:
	movq	-56(%rbp), %rax
	movl	(%rax), %edx
	movq	-48(%rbp), %rcx
	movq	-40(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	dessinerSerpent
	movq	$90, -8(%rbp)
	jmp	.L327
.L271:
	movl	$1, -12(%rbp)
	movq	$59, -8(%rbp)
	jmp	.L327
.L261:
	subl	$1, -16(%rbp)
	movq	$58, -8(%rbp)
	jmp	.L327
.L304:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cltq
	movb	$32, (%rdx,%rax)
	movq	$57, -8(%rbp)
	jmp	.L327
.L318:
	movq	-40(%rbp), %rax
	movl	$1, (%rax)
	movq	$21, -8(%rbp)
	jmp	.L327
.L320:
	subl	$1, -20(%rbp)
	movq	$58, -8(%rbp)
	jmp	.L327
.L316:
	addl	$1, -16(%rbp)
	movq	$58, -8(%rbp)
	jmp	.L327
.L275:
	movq	16(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L354
	movq	$26, -8(%rbp)
	jmp	.L327
.L354:
	movq	$57, -8(%rbp)
	jmp	.L327
.L311:
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jne	.L356
	movq	$69, -8(%rbp)
	jmp	.L327
.L356:
	movq	$50, -8(%rbp)
	jmp	.L327
.L313:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %edi
	movq	-48(%rbp), %rax
	movl	(%rax), %esi
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movq	-72(%rbp), %r8
	movl	32(%rbp), %ecx
	movl	24(%rbp), %edx
	movq	%r8, %r9
	movl	%edi, %r8d
	movl	%eax, %edi
	call	calculerDirection
	movq	-64(%rbp), %rdx
	movb	%al, (%rdx)
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -20(%rbp)
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -16(%rbp)
	movq	$87, -8(%rbp)
	jmp	.L327
.L272:
	addl	$1, -20(%rbp)
	movq	$58, -8(%rbp)
	jmp	.L327
.L278:
	addl	$1, -16(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L327
.L279:
	cmpl	$0, -12(%rbp)
	setne	%dl
	movq	-80(%rbp), %rax
	movb	%dl, (%rax)
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$54, %al
	sete	%dl
	movq	16(%rbp), %rax
	movb	%dl, (%rax)
	movq	$63, -8(%rbp)
	jmp	.L327
.L321:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	jne	.L359
	movq	$68, -8(%rbp)
	jmp	.L327
.L359:
	movq	$77, -8(%rbp)
	jmp	.L327
.L303:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$80, %eax
	jle	.L361
	movq	$11, -8(%rbp)
	jmp	.L327
.L361:
	movq	$21, -8(%rbp)
	jmp	.L327
.L295:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cltq
	leaq	1(%rax), %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L363
	movq	$30, -8(%rbp)
	jmp	.L327
.L363:
	movq	$49, -8(%rbp)
	jmp	.L327
.L277:
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$41, %eax
	jne	.L365
	movq	$36, -8(%rbp)
	jmp	.L327
.L365:
	movq	$56, -8(%rbp)
	jmp	.L327
.L260:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	cmpl	$122, %eax
	je	.L367
	cmpl	$122, %eax
	jg	.L368
	cmpl	$115, %eax
	je	.L369
	cmpl	$115, %eax
	jg	.L368
	cmpl	$100, %eax
	je	.L370
	cmpl	$113, %eax
	je	.L371
	jmp	.L368
.L370:
	movq	$28, -8(%rbp)
	jmp	.L372
.L371:
	movq	$73, -8(%rbp)
	jmp	.L372
.L369:
	movq	$60, -8(%rbp)
	jmp	.L372
.L367:
	movq	$24, -8(%rbp)
	jmp	.L372
.L368:
	movq	$15, -8(%rbp)
	nop
.L372:
	jmp	.L327
.L280:
	movq	-40(%rbp), %rax
	movl	-20(%rbp), %edx
	movl	%edx, (%rax)
	movq	-48(%rbp), %rax
	movl	-16(%rbp), %edx
	movl	%edx, (%rax)
	movq	$88, -8(%rbp)
	jmp	.L327
.L262:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	movb	%al, -25(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L327
.L297:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	addl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L373
	movq	$54, -8(%rbp)
	jmp	.L327
.L373:
	movq	$49, -8(%rbp)
	jmp	.L327
.L266:
	movb	$100, -25(%rbp)
	movq	$49, -8(%rbp)
	jmp	.L327
.L288:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rdx, %rax
	leaq	-41(%rax), %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L375
	movq	$42, -8(%rbp)
	jmp	.L327
.L375:
	movq	$38, -8(%rbp)
	jmp	.L327
.L269:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$113, %al
	jne	.L377
	movq	$47, -8(%rbp)
	jmp	.L327
.L377:
	movq	$33, -8(%rbp)
	jmp	.L327
.L308:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movl	-16(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	jne	.L379
	movq	$84, -8(%rbp)
	jmp	.L327
.L379:
	movq	$41, -8(%rbp)
	jmp	.L327
.L302:
	addl	$1, -20(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L327
.L284:
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$20, %eax
	jne	.L381
	movq	$72, -8(%rbp)
	jmp	.L327
.L381:
	movq	$31, -8(%rbp)
	jmp	.L327
.L274:
	movl	-24(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	-24(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-40(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movl	-24(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	-24(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	subl	$1, -24(%rbp)
	movq	$43, -8(%rbp)
	jmp	.L327
.L289:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	subl	$1, %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$35, %al
	je	.L383
	movq	$18, -8(%rbp)
	jmp	.L327
.L383:
	movq	$34, -8(%rbp)
	jmp	.L327
.L267:
	subl	$1, -20(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L327
.L322:
	movq	-40(%rbp), %rax
	movl	$80, (%rax)
	movq	$31, -8(%rbp)
	jmp	.L327
.L268:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$81, %eax
	jne	.L385
	movq	$52, -8(%rbp)
	jmp	.L327
.L385:
	movq	$31, -8(%rbp)
	jmp	.L327
.L298:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$100, %al
	jne	.L387
	movq	$47, -8(%rbp)
	jmp	.L327
.L387:
	movq	$49, -8(%rbp)
	jmp	.L327
.L294:
	movl	-20(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movl	-16(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	jne	.L389
	movq	$84, -8(%rbp)
	jmp	.L327
.L389:
	movq	$58, -8(%rbp)
	jmp	.L327
.L319:
	movq	$89, -8(%rbp)
	jmp	.L327
.L293:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rdx, %rax
	leaq	-41(%rax), %rdx
	movq	-72(%rbp), %rax
	addq	%rax, %rdx
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	cmpb	$88, %al
	je	.L391
	movq	$45, -8(%rbp)
	jmp	.L327
.L391:
	movq	$38, -8(%rbp)
	jmp	.L327
.L290:
	movq	-48(%rbp), %rax
	movl	$1, (%rax)
	movq	$6, -8(%rbp)
	jmp	.L327
.L273:
	movq	-64(%rbp), %rax
	movzbl	(%rax), %eax
	cmpb	$115, %al
	jne	.L393
	movq	$48, -8(%rbp)
	jmp	.L327
.L393:
	movq	$71, -8(%rbp)
	jmp	.L327
.L259:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cmpl	$40, %eax
	jne	.L395
	movq	$19, -8(%rbp)
	jmp	.L327
.L395:
	movq	$50, -8(%rbp)
	jmp	.L327
.L301:
	movl	$0, -12(%rbp)
	movq	$59, -8(%rbp)
	jmp	.L327
.L292:
	cmpl	$0, -24(%rbp)
	jle	.L397
	movq	$65, -8(%rbp)
	jmp	.L327
.L397:
	movq	$17, -8(%rbp)
	jmp	.L327
.L310:
	movq	-40(%rbp), %rax
	movl	$80, (%rax)
	movq	$27, -8(%rbp)
	jmp	.L327
.L400:
	nop
.L327:
	jmp	.L399
.L401:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	progresser, .-progresser
	.globl	dessinerPlateau
	.type	dessinerPlateau, @function
dessinerPlateau:
.LFB15:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$6, -8(%rbp)
.L418:
	cmpq	$11, -8(%rbp)
	ja	.L419
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L405(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L405(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L405:
	.long	.L419-.L405
	.long	.L411-.L405
	.long	.L410-.L405
	.long	.L409-.L405
	.long	.L419-.L405
	.long	.L408-.L405
	.long	.L407-.L405
	.long	.L419-.L405
	.long	.L420-.L405
	.long	.L419-.L405
	.long	.L419-.L405
	.long	.L404-.L405
	.text
.L411:
	cmpl	$40, -16(%rbp)
	jg	.L413
	movq	$3, -8(%rbp)
	jmp	.L415
.L413:
	movq	$8, -8(%rbp)
	jmp	.L415
.L409:
	movl	$1, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L415
.L404:
	cmpl	$80, -12(%rbp)
	jg	.L416
	movq	$2, -8(%rbp)
	jmp	.L415
.L416:
	movq	$5, -8(%rbp)
	jmp	.L415
.L407:
	movl	$1, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L415
.L408:
	addl	$1, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L415
.L410:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	addq	%rax, %rdx
	movl	-16(%rbp), %eax
	cltq
	movzbl	(%rdx,%rax), %eax
	movsbl	%al, %edx
	movl	-16(%rbp), %ecx
	movl	-12(%rbp), %eax
	movl	%ecx, %esi
	movl	%eax, %edi
	call	afficher
	addl	$1, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L415
.L419:
	nop
.L415:
	jmp	.L418
.L420:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE15:
	.size	dessinerPlateau, .-dessinerPlateau
	.globl	afficher
	.type	afficher, @function
afficher:
.LFB16:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movl	%edx, %eax
	movb	%al, -28(%rbp)
	movq	$1, -8(%rbp)
.L427:
	cmpq	$2, -8(%rbp)
	je	.L422
	cmpq	$2, -8(%rbp)
	ja	.L428
	cmpq	$0, -8(%rbp)
	je	.L429
	cmpq	$1, -8(%rbp)
	jne	.L428
	movq	$2, -8(%rbp)
	jmp	.L425
.L422:
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	gotoxy
	movsbl	-28(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	movl	$1, %esi
	movl	$1, %edi
	call	gotoxy
	movq	$0, -8(%rbp)
	jmp	.L425
.L428:
	nop
.L425:
	jmp	.L427
.L429:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE16:
	.size	afficher, .-afficher
	.section	.rodata
	.align 8
.LC3:
	.long	0
	.long	1093567616
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
