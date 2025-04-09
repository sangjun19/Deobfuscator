	.file	"elaimizrahi_CS-138_h_flatten.c"
	.text
	.globl	_TIG_IZ_w4vc_envp
	.bss
	.align 8
	.type	_TIG_IZ_w4vc_envp, @object
	.size	_TIG_IZ_w4vc_envp, 8
_TIG_IZ_w4vc_envp:
	.zero	8
	.globl	_TIG_IZ_w4vc_argv
	.align 8
	.type	_TIG_IZ_w4vc_argv, @object
	.size	_TIG_IZ_w4vc_argv, 8
_TIG_IZ_w4vc_argv:
	.zero	8
	.globl	_TIG_IZ_w4vc_argc
	.align 4
	.type	_TIG_IZ_w4vc_argc, @object
	.size	_TIG_IZ_w4vc_argc, 4
_TIG_IZ_w4vc_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d"
	.text
	.globl	vlintegerRead
	.type	vlintegerRead, @function
vlintegerRead:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -32(%rbp)
.L14:
	cmpq	$9, -32(%rbp)
	ja	.L17
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L17-.L4
	.long	.L17-.L4
	.long	.L9-.L4
	.long	.L17-.L4
	.long	.L8-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L8:
	movl	$0, -40(%rbp)
	movq	$7, -32(%rbp)
	jmp	.L10
.L5:
	movl	-40(%rbp), %eax
	addl	$1, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -16(%rbp)
	movq	-56(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-56(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-40(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	-44(%rbp), %eax
	movl	%eax, (%rdx)
	addl	$1, -40(%rbp)
	movq	$7, -32(%rbp)
	jmp	.L10
.L3:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -24(%rbp)
	movq	-56(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-56(%rbp), %rax
	movl	-40(%rbp), %edx
	movl	%edx, (%rax)
	movq	$5, -32(%rbp)
	jmp	.L10
.L6:
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -36(%rbp)
	movq	$2, -32(%rbp)
	jmp	.L10
.L9:
	cmpl	$0, -36(%rbp)
	je	.L12
	movq	$8, -32(%rbp)
	jmp	.L10
.L12:
	movq	$9, -32(%rbp)
	jmp	.L10
.L17:
	nop
.L10:
	jmp	.L14
.L18:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L16
	call	__stack_chk_fail@PLT
.L16:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	vlintegerRead, .-vlintegerRead
	.section	.rodata
.LC1:
	.string	"length=%d\n"
	.text
	.globl	vlintegerPrint
	.type	vlintegerPrint, @function
vlintegerPrint:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L32:
	cmpq	$8, -8(%rbp)
	ja	.L33
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L22(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L22(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L22:
	.long	.L27-.L22
	.long	.L26-.L22
	.long	.L25-.L22
	.long	.L34-.L22
	.long	.L33-.L22
	.long	.L33-.L22
	.long	.L23-.L22
	.long	.L33-.L22
	.long	.L21-.L22
	.text
.L21:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L28
.L26:
	movq	$8, -8(%rbp)
	jmp	.L28
.L23:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-12(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L28
.L27:
	movl	$10, %edi
	call	putchar@PLT
	movq	$3, -8(%rbp)
	jmp	.L28
.L25:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L30
	movq	$6, -8(%rbp)
	jmp	.L28
.L30:
	movq	$0, -8(%rbp)
	jmp	.L28
.L33:
	nop
.L28:
	jmp	.L32
.L34:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	vlintegerPrint, .-vlintegerPrint
	.section	.rodata
.LC2:
	.string	"t2:%d\n"
.LC3:
	.string	"%d %d \n"
.LC4:
	.string	"t1:%d\n"
	.text
	.globl	vlintegerAdd
	.type	vlintegerAdd, @function
vlintegerAdd:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movq	%rdi, -152(%rbp)
	movq	%rsi, -160(%rbp)
	movq	$16, -64(%rbp)
.L125:
	cmpq	$81, -64(%rbp)
	ja	.L127
	movq	-64(%rbp), %rax
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
	.long	.L127-.L38
	.long	.L86-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L85-.L38
	.long	.L84-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L83-.L38
	.long	.L82-.L38
	.long	.L81-.L38
	.long	.L80-.L38
	.long	.L127-.L38
	.long	.L79-.L38
	.long	.L78-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L77-.L38
	.long	.L76-.L38
	.long	.L75-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L74-.L38
	.long	.L73-.L38
	.long	.L127-.L38
	.long	.L72-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L71-.L38
	.long	.L70-.L38
	.long	.L127-.L38
	.long	.L69-.L38
	.long	.L68-.L38
	.long	.L67-.L38
	.long	.L66-.L38
	.long	.L65-.L38
	.long	.L127-.L38
	.long	.L64-.L38
	.long	.L63-.L38
	.long	.L62-.L38
	.long	.L61-.L38
	.long	.L60-.L38
	.long	.L127-.L38
	.long	.L59-.L38
	.long	.L58-.L38
	.long	.L57-.L38
	.long	.L56-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L55-.L38
	.long	.L127-.L38
	.long	.L54-.L38
	.long	.L53-.L38
	.long	.L52-.L38
	.long	.L51-.L38
	.long	.L50-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L49-.L38
	.long	.L48-.L38
	.long	.L47-.L38
	.long	.L46-.L38
	.long	.L45-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L44-.L38
	.long	.L43-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L127-.L38
	.long	.L42-.L38
	.long	.L41-.L38
	.long	.L40-.L38
	.long	.L39-.L38
	.long	.L37-.L38
	.text
.L39:
	movl	-132(%rbp), %eax
	subl	-120(%rbp), %eax
	cmpl	%eax, -108(%rbp)
	jge	.L87
	movq	$24, -64(%rbp)
	jmp	.L89
.L87:
	movq	$65, -64(%rbp)
	jmp	.L89
.L73:
	movl	-132(%rbp), %eax
	addl	$1, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-72(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	-132(%rbp), %eax
	leal	1(%rax), %edx
	movq	-72(%rbp), %rax
	movl	%edx, (%rax)
	movl	-132(%rbp), %eax
	movl	%eax, -80(%rbp)
	movq	$43, -64(%rbp)
	jmp	.L89
.L56:
	movq	-152(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-160(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-84(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	addl	%eax, %edx
	movl	-128(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -124(%rbp)
	movl	$0, -128(%rbp)
	movq	$54, -64(%rbp)
	jmp	.L89
.L55:
	movl	$0, -92(%rbp)
	movq	$35, -64(%rbp)
	jmp	.L89
.L85:
	cmpl	$0, -112(%rbp)
	js	.L90
	movq	$13, -64(%rbp)
	jmp	.L89
.L90:
	movq	$81, -64(%rbp)
	jmp	.L89
.L79:
	movl	-132(%rbp), %eax
	subl	-120(%rbp), %eax
	cmpl	%eax, -96(%rbp)
	jge	.L92
	movq	$34, -64(%rbp)
	jmp	.L89
.L92:
	movq	$52, -64(%rbp)
	jmp	.L89
.L52:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	-124(%rbp), %eax
	movl	%eax, (%rdx)
	subl	$1, -84(%rbp)
	movq	$47, -64(%rbp)
	jmp	.L89
.L40:
	movq	-72(%rbp), %rax
	movl	-132(%rbp), %edx
	movl	%edx, (%rax)
	movl	-132(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -56(%rbp)
	movq	-72(%rbp), %rax
	movq	-56(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	-132(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -84(%rbp)
	movq	$47, -64(%rbp)
	jmp	.L89
.L71:
	movl	$0, -96(%rbp)
	movq	$15, -64(%rbp)
	jmp	.L89
.L81:
	movl	-132(%rbp), %eax
	subl	-120(%rbp), %eax
	cmpl	%eax, -116(%rbp)
	jge	.L94
	movq	$21, -64(%rbp)
	jmp	.L89
.L94:
	movq	$55, -64(%rbp)
	jmp	.L89
.L54:
	cmpl	$9, -124(%rbp)
	jle	.L96
	movq	$20, -64(%rbp)
	jmp	.L89
.L96:
	movq	$56, -64(%rbp)
	jmp	.L89
.L41:
	movl	$0, -88(%rbp)
	movq	$38, -64(%rbp)
	jmp	.L89
.L86:
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-112(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-112(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	movq	$48, -64(%rbp)
	jmp	.L89
.L37:
	addl	$1, -116(%rbp)
	movq	$12, -64(%rbp)
	jmp	.L89
.L42:
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-88(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -88(%rbp)
	movq	$38, -64(%rbp)
	jmp	.L89
.L78:
	movq	$71, -64(%rbp)
	jmp	.L89
.L74:
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-108(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -108(%rbp)
	movq	$80, -64(%rbp)
	jmp	.L89
.L75:
	movl	-132(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -112(%rbp)
	movq	$4, -64(%rbp)
	jmp	.L89
.L67:
	subl	$1, -100(%rbp)
	movq	$40, -64(%rbp)
	jmp	.L89
.L51:
	movq	-152(%rbp), %rax
	movl	(%rax), %edx
	movq	-160(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L98
	movq	$37, -64(%rbp)
	jmp	.L89
.L98:
	movq	$65, -64(%rbp)
	jmp	.L89
.L45:
	movq	-72(%rbp), %rax
	jmp	.L126
.L82:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movl	-128(%rbp), %edx
	movl	%edx, (%rax)
	movq	$44, -64(%rbp)
	jmp	.L89
.L80:
	cmpl	$0, -112(%rbp)
	jne	.L101
	movq	$66, -64(%rbp)
	jmp	.L89
.L101:
	movq	$64, -64(%rbp)
	jmp	.L89
.L77:
	addl	$1, -104(%rbp)
	movq	$27, -64(%rbp)
	jmp	.L89
.L70:
	movq	-160(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -132(%rbp)
	movq	-152(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -120(%rbp)
	movl	-132(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-152(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -16(%rbp)
	movq	-152(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	$0, -104(%rbp)
	movq	$27, -64(%rbp)
	jmp	.L89
.L64:
	cmpl	$0, -100(%rbp)
	js	.L103
	movq	$46, -64(%rbp)
	jmp	.L89
.L103:
	movq	$19, -64(%rbp)
	jmp	.L89
.L46:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-72(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-72(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-72(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	subl	$1, -80(%rbp)
	movq	$43, -64(%rbp)
	jmp	.L89
.L53:
	movl	$0, -108(%rbp)
	movq	$80, -64(%rbp)
	jmp	.L89
.L72:
	movl	-132(%rbp), %eax
	subl	-120(%rbp), %eax
	cmpl	%eax, -104(%rbp)
	jge	.L105
	movq	$58, -64(%rbp)
	jmp	.L89
.L105:
	movq	$31, -64(%rbp)
	jmp	.L89
.L65:
	movl	-88(%rbp), %eax
	cmpl	-132(%rbp), %eax
	jge	.L107
	movq	$77, -64(%rbp)
	jmp	.L89
.L107:
	movq	$79, -64(%rbp)
	jmp	.L89
.L50:
	movl	-132(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -100(%rbp)
	movq	$40, -64(%rbp)
	jmp	.L89
.L69:
	movq	-152(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-96(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -96(%rbp)
	movq	$15, -64(%rbp)
	jmp	.L89
.L57:
	subl	$1, -112(%rbp)
	movq	$4, -64(%rbp)
	jmp	.L89
.L44:
	movq	-152(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -132(%rbp)
	movl	$0, -76(%rbp)
	movl	$0, -128(%rbp)
	call	vlintegerCreate
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -72(%rbp)
	movq	$57, -64(%rbp)
	jmp	.L89
.L48:
	movq	-152(%rbp), %rax
	movl	(%rax), %edx
	movq	-160(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jge	.L109
	movq	$32, -64(%rbp)
	jmp	.L89
.L109:
	movq	$52, -64(%rbp)
	jmp	.L89
.L58:
	cmpl	$0, -84(%rbp)
	js	.L111
	movq	$49, -64(%rbp)
	jmp	.L89
.L111:
	movq	$10, -64(%rbp)
	jmp	.L89
.L60:
	movl	-132(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-152(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -40(%rbp)
	movq	-152(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	-132(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -32(%rbp)
	movq	-160(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$68, -64(%rbp)
	jmp	.L89
.L84:
	movq	-152(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -92(%rbp)
	movq	$35, -64(%rbp)
	jmp	.L89
.L43:
	movq	-152(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-100(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	$42, -64(%rbp)
	jmp	.L89
.L66:
	movq	-152(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -132(%rbp)
	movq	-160(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -120(%rbp)
	movl	-132(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-160(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -24(%rbp)
	movq	-160(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	$0, -116(%rbp)
	movq	$12, -64(%rbp)
	jmp	.L89
.L49:
	cmpl	$0, -112(%rbp)
	je	.L113
	movq	$1, -64(%rbp)
	jmp	.L89
.L113:
	movq	$48, -64(%rbp)
	jmp	.L89
.L63:
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-100(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-152(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-100(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	movq	$36, -64(%rbp)
	jmp	.L89
.L83:
	cmpl	$0, -128(%rbp)
	jle	.L115
	movq	$25, -64(%rbp)
	jmp	.L89
.L115:
	movq	$44, -64(%rbp)
	jmp	.L89
.L62:
	cmpl	$0, -100(%rbp)
	je	.L117
	movq	$41, -64(%rbp)
	jmp	.L89
.L117:
	movq	$36, -64(%rbp)
	jmp	.L89
.L59:
	cmpl	$0, -100(%rbp)
	jne	.L119
	movq	$72, -64(%rbp)
	jmp	.L89
.L119:
	movq	$42, -64(%rbp)
	jmp	.L89
.L47:
	movq	-160(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-112(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	$64, -64(%rbp)
	jmp	.L89
.L68:
	movl	-92(%rbp), %eax
	cmpl	-132(%rbp), %eax
	jge	.L121
	movq	$5, -64(%rbp)
	jmp	.L89
.L121:
	movq	$78, -64(%rbp)
	jmp	.L89
.L61:
	cmpl	$0, -80(%rbp)
	jle	.L123
	movq	$67, -64(%rbp)
	jmp	.L89
.L123:
	movq	$11, -64(%rbp)
	jmp	.L89
.L76:
	movl	-124(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1717986919, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$2, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -128(%rbp)
	movl	-124(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%edx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	movl	%eax, %ecx
	movl	%ecx, %eax
	sall	$2, %eax
	addl	%ecx, %eax
	addl	%eax, %eax
	subl	%eax, %edx
	movl	%edx, -124(%rbp)
	movq	$56, -64(%rbp)
	jmp	.L89
.L127:
	nop
.L89:
	jmp	.L125
.L126:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	vlintegerAdd, .-vlintegerAdd
	.section	.rodata
	.align 8
.LC5:
	.string	"Enter the digits separated by a space: "
.LC6:
	.string	"%c"
.LC7:
	.string	"addition"
.LC8:
	.string	"ARRARY: %d"
.LC9:
	.string	"main"
.LC10:
	.string	"elaimizrahi_CS-138_h.c"
.LC11:
	.string	"add->arr[0] !=0"
.LC12:
	.string	"int1->arr[0] !=0"
.LC13:
	.string	"mult->arr[0] !=0"
.LC14:
	.string	"int2->arr[0] !=0"
.LC15:
	.string	"multiplication"
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
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_w4vc_envp(%rip)
	nop
.L129:
	movq	$0, _TIG_IZ_w4vc_argv(%rip)
	nop
.L130:
	movl	$0, _TIG_IZ_w4vc_argc(%rip)
	nop
	nop
.L131:
.L132:
#APP
# 181 "elaimizrahi_CS-138_h.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-w4vc--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_w4vc_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_w4vc_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_w4vc_envp(%rip)
	nop
	movq	$8, -48(%rbp)
.L165:
	cmpq	$24, -48(%rbp)
	ja	.L168
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L135(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L135(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L135:
	.long	.L152-.L135
	.long	.L151-.L135
	.long	.L150-.L135
	.long	.L149-.L135
	.long	.L148-.L135
	.long	.L168-.L135
	.long	.L147-.L135
	.long	.L146-.L135
	.long	.L145-.L135
	.long	.L144-.L135
	.long	.L168-.L135
	.long	.L143-.L135
	.long	.L142-.L135
	.long	.L141-.L135
	.long	.L168-.L135
	.long	.L168-.L135
	.long	.L168-.L135
	.long	.L140-.L135
	.long	.L139-.L135
	.long	.L138-.L135
	.long	.L168-.L135
	.long	.L137-.L135
	.long	.L168-.L135
	.long	.L136-.L135
	.long	.L134-.L135
	.text
.L139:
	call	vlintegerCreate
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -80(%rbp)
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerRead
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$21, -48(%rbp)
	jmp	.L153
.L148:
	leaq	-85(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-72(%rbp), %rdx
	movq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	vlintegerAdd
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$23, -48(%rbp)
	jmp	.L153
.L142:
	movq	-80(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -84(%rbp)
	movq	$2, -48(%rbp)
	jmp	.L153
.L145:
	movq	$18, -48(%rbp)
	jmp	.L153
.L151:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L154
	movq	$4, -48(%rbp)
	jmp	.L153
.L154:
	movq	$6, -48(%rbp)
	jmp	.L153
.L136:
	movq	-64(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L156
	movq	$7, -48(%rbp)
	jmp	.L153
.L156:
	movq	$3, -48(%rbp)
	jmp	.L153
.L149:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rcx
	movl	$389, %edx
	leaq	.LC10(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L134:
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerDestroy
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerDestroy
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerDestroy
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerDestroy
	movq	$0, -48(%rbp)
	jmp	.L153
.L137:
	movq	-80(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L158
	movq	$11, -48(%rbp)
	jmp	.L153
.L158:
	movq	$13, -48(%rbp)
	jmp	.L153
.L143:
	leaq	-85(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	call	vlintegerCreate
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -72(%rbp)
	movl	$0, -84(%rbp)
	movq	$2, -48(%rbp)
	jmp	.L153
.L144:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerRead
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$1, -48(%rbp)
	jmp	.L153
.L141:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rcx
	movl	$369, %edx
	leaq	.LC10(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L138:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L160
	movq	$24, -48(%rbp)
	jmp	.L153
.L160:
	movq	$17, -48(%rbp)
	jmp	.L153
.L140:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rcx
	movl	$395, %edx
	leaq	.LC10(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L147:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rcx
	movl	$381, %edx
	leaq	.LC10(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L152:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L166
	jmp	.L167
.L146:
	leaq	-85(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-72(%rbp), %rdx
	movq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	vlintegerMult
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -56(%rbp)
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$19, -48(%rbp)
	jmp	.L153
.L150:
	movq	-80(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -84(%rbp)
	jge	.L163
	movq	$12, -48(%rbp)
	jmp	.L153
.L163:
	movq	$9, -48(%rbp)
	jmp	.L153
.L168:
	nop
.L153:
	jmp	.L165
.L167:
	call	__stack_chk_fail@PLT
.L166:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.globl	vlintegerDestroy
	.type	vlintegerDestroy, @function
vlintegerDestroy:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$3, -8(%rbp)
.L177:
	cmpq	$3, -8(%rbp)
	je	.L170
	cmpq	$3, -8(%rbp)
	ja	.L178
	cmpq	$0, -8(%rbp)
	je	.L179
	cmpq	$1, -8(%rbp)
	jne	.L178
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$0, -8(%rbp)
	jmp	.L173
.L170:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L174
	movq	$1, -8(%rbp)
	jmp	.L173
.L174:
	movq	$0, -8(%rbp)
	jmp	.L173
.L178:
	nop
.L173:
	jmp	.L177
.L179:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	vlintegerDestroy, .-vlintegerDestroy
	.section	.rodata
.LC16:
	.string	"mult3: %d\n"
.LC17:
	.string	"mult4: %d\n"
.LC18:
	.string	"hello!"
.LC19:
	.string	"mult: %d\n"
.LC20:
	.string	"i1: %d\n"
.LC21:
	.string	"i2: %d\n"
.LC22:
	.string	"i: %d\n"
.LC23:
	.string	"j: %d\n"
.LC24:
	.string	"mult2: %d\n"
	.text
	.globl	vlintegerMult
	.type	vlintegerMult, @function
vlintegerMult:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movq	%rdi, -104(%rbp)
	movq	%rsi, -112(%rbp)
	movq	$39, -32(%rbp)
.L225:
	cmpq	$48, -32(%rbp)
	ja	.L227
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L183(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L183(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L183:
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L206-.L183
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L205-.L183
	.long	.L227-.L183
	.long	.L204-.L183
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L203-.L183
	.long	.L227-.L183
	.long	.L202-.L183
	.long	.L227-.L183
	.long	.L201-.L183
	.long	.L227-.L183
	.long	.L200-.L183
	.long	.L199-.L183
	.long	.L227-.L183
	.long	.L198-.L183
	.long	.L197-.L183
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L196-.L183
	.long	.L195-.L183
	.long	.L194-.L183
	.long	.L193-.L183
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L192-.L183
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L191-.L183
	.long	.L190-.L183
	.long	.L189-.L183
	.long	.L227-.L183
	.long	.L227-.L183
	.long	.L188-.L183
	.long	.L187-.L183
	.long	.L186-.L183
	.long	.L185-.L183
	.long	.L227-.L183
	.long	.L184-.L183
	.long	.L182-.L183
	.text
.L199:
	movq	-104(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -60(%rbp)
	movq	$43, -32(%rbp)
	jmp	.L207
.L195:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-52(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-52(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	addl	$1, -52(%rbp)
	movq	$21, -32(%rbp)
	jmp	.L207
.L201:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -64(%rbp)
	jge	.L208
	movq	$37, -32(%rbp)
	jmp	.L207
.L208:
	movq	$18, -32(%rbp)
	jmp	.L207
.L192:
	movl	$0, -84(%rbp)
	movq	-112(%rbp), %rax
	movl	(%rax), %eax
	subl	$1, %eax
	movl	%eax, -68(%rbp)
	movq	$6, -32(%rbp)
	jmp	.L207
.L204:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jne	.L210
	movq	$2, -32(%rbp)
	jmp	.L207
.L210:
	movq	$17, -32(%rbp)
	jmp	.L207
.L185:
	movl	-76(%rbp), %eax
	subl	-84(%rbp), %eax
	movl	%eax, -80(%rbp)
	movl	$0, -64(%rbp)
	movq	$15, -32(%rbp)
	jmp	.L207
.L196:
	movq	-104(%rbp), %rax
	movl	(%rax), %edx
	movq	-112(%rbp), %rax
	movl	(%rax), %eax
	addl	%edx, %eax
	movl	%eax, -76(%rbp)
	call	vlintegerCreate
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	-76(%rbp), %eax
	cltq
	movl	$4, %esi
	movq	%rax, %rdi
	call	calloc@PLT
	movq	%rax, -8(%rbp)
	movq	-40(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-40(%rbp), %rax
	movl	-76(%rbp), %edx
	movl	%edx, (%rax)
	movl	$0, -72(%rbp)
	movq	$42, -32(%rbp)
	jmp	.L207
.L197:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -52(%rbp)
	jge	.L212
	movq	$25, -32(%rbp)
	jmp	.L207
.L212:
	movq	$38, -32(%rbp)
	jmp	.L207
.L194:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-56(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -56(%rbp)
	movq	$20, -32(%rbp)
	jmp	.L207
.L203:
	addl	$1, -84(%rbp)
	subl	$1, -68(%rbp)
	movq	$6, -32(%rbp)
	jmp	.L207
.L202:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-48(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -48(%rbp)
	movq	$48, -32(%rbp)
	jmp	.L207
.L200:
	movq	-40(%rbp), %rax
	jmp	.L226
.L205:
	cmpl	$0, -68(%rbp)
	js	.L215
	movq	$45, -32(%rbp)
	jmp	.L207
.L215:
	movq	$27, -32(%rbp)
	jmp	.L207
.L193:
	movl	$0, -56(%rbp)
	movq	$20, -32(%rbp)
	jmp	.L207
.L190:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	leal	-1(%rax), %edx
	movq	-40(%rbp), %rax
	movl	%edx, (%rax)
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	$0, -48(%rbp)
	movq	$48, -32(%rbp)
	jmp	.L207
.L182:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -48(%rbp)
	jge	.L217
	movq	$13, -32(%rbp)
	jmp	.L207
.L217:
	movq	$8, -32(%rbp)
	jmp	.L207
.L184:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-72(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -72(%rbp)
	movq	$42, -32(%rbp)
	jmp	.L207
.L186:
	movq	-112(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-68(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-104(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-60(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	imull	%eax, %edx
	movq	-40(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	addl	%edx, %eax
	movl	%eax, -44(%rbp)
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	(%rdx,%rax), %rsi
	movl	-44(%rbp), %ecx
	movslq	%ecx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%ecx, %edi
	sarl	$31, %edi
	subl	%edi, %eax
	movl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	movl	%edx, (%rsi)
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1717986919, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$2, %edx
	sarl	$31, %eax
	movl	%edx, %esi
	subl	%eax, %esi
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	addq	%rdx, %rax
	leal	(%rcx,%rsi), %edx
	movl	%edx, (%rax)
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-80(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-84(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-80(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-68(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-60(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	subl	$1, -80(%rbp)
	subl	$1, -60(%rbp)
	movq	$43, -32(%rbp)
	jmp	.L207
.L191:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-64(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -64(%rbp)
	movq	$15, -32(%rbp)
	jmp	.L207
.L188:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -72(%rbp)
	jge	.L219
	movq	$47, -32(%rbp)
	jmp	.L207
.L219:
	movq	$31, -32(%rbp)
	jmp	.L207
.L189:
	movq	$24, -32(%rbp)
	jmp	.L207
.L187:
	cmpl	$0, -60(%rbp)
	js	.L221
	movq	$44, -32(%rbp)
	jmp	.L207
.L221:
	movq	$11, -32(%rbp)
	jmp	.L207
.L206:
	movl	$1, -52(%rbp)
	movq	$21, -32(%rbp)
	jmp	.L207
.L198:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -56(%rbp)
	jge	.L223
	movq	$26, -32(%rbp)
	jmp	.L207
.L223:
	movq	$8, -32(%rbp)
	jmp	.L207
.L227:
	nop
.L207:
	jmp	.L225
.L226:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	vlintegerMult, .-vlintegerMult
	.globl	power
	.type	power, @function
power:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L240:
	cmpq	$7, -8(%rbp)
	ja	.L242
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L231(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L231(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L231:
	.long	.L235-.L231
	.long	.L234-.L231
	.long	.L242-.L231
	.long	.L233-.L231
	.long	.L242-.L231
	.long	.L232-.L231
	.long	.L242-.L231
	.long	.L230-.L231
	.text
.L234:
	movl	-16(%rbp), %eax
	jmp	.L241
.L233:
	movl	-12(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.L237
	movq	$5, -8(%rbp)
	jmp	.L239
.L237:
	movq	$1, -8(%rbp)
	jmp	.L239
.L232:
	movl	-16(%rbp), %eax
	imull	-20(%rbp), %eax
	movl	%eax, -16(%rbp)
	addl	$1, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L239
.L235:
	movq	$7, -8(%rbp)
	jmp	.L239
.L230:
	movl	$1, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L239
.L242:
	nop
.L239:
	jmp	.L240
.L241:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	power, .-power
	.globl	vlintegerCreate
	.type	vlintegerCreate, @function
vlintegerCreate:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$2, -16(%rbp)
.L249:
	cmpq	$2, -16(%rbp)
	je	.L244
	cmpq	$2, -16(%rbp)
	ja	.L251
	cmpq	$0, -16(%rbp)
	je	.L246
	cmpq	$1, -16(%rbp)
	jne	.L251
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-24(%rbp), %rax
	movl	$0, (%rax)
	movq	$0, -16(%rbp)
	jmp	.L247
.L246:
	movq	-24(%rbp), %rax
	jmp	.L250
.L244:
	movq	$1, -16(%rbp)
	jmp	.L247
.L251:
	nop
.L247:
	jmp	.L249
.L250:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	vlintegerCreate, .-vlintegerCreate
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
