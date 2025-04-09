	.file	"elaimizrahi_CS-138_j_flatten.c"
	.text
	.globl	_TIG_IZ_Defe_argc
	.bss
	.align 4
	.type	_TIG_IZ_Defe_argc, @object
	.size	_TIG_IZ_Defe_argc, 4
_TIG_IZ_Defe_argc:
	.zero	4
	.globl	_TIG_IZ_Defe_argv
	.align 8
	.type	_TIG_IZ_Defe_argv, @object
	.size	_TIG_IZ_Defe_argv, 8
_TIG_IZ_Defe_argv:
	.zero	8
	.globl	_TIG_IZ_Defe_envp
	.align 8
	.type	_TIG_IZ_Defe_envp, @object
	.size	_TIG_IZ_Defe_envp, 8
_TIG_IZ_Defe_envp:
	.zero	8
	.text
	.globl	vlintegerAdd
	.type	vlintegerAdd, @function
vlintegerAdd:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movq	%rdi, -136(%rbp)
	movq	%rsi, -144(%rbp)
	movq	$45, -64(%rbp)
.L81:
	cmpq	$67, -64(%rbp)
	ja	.L83
	movq	-64(%rbp), %rax
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
	.long	.L46-.L4
	.long	.L83-.L4
	.long	.L45-.L4
	.long	.L44-.L4
	.long	.L43-.L4
	.long	.L42-.L4
	.long	.L83-.L4
	.long	.L41-.L4
	.long	.L40-.L4
	.long	.L83-.L4
	.long	.L83-.L4
	.long	.L39-.L4
	.long	.L38-.L4
	.long	.L37-.L4
	.long	.L36-.L4
	.long	.L35-.L4
	.long	.L83-.L4
	.long	.L34-.L4
	.long	.L33-.L4
	.long	.L83-.L4
	.long	.L32-.L4
	.long	.L83-.L4
	.long	.L83-.L4
	.long	.L31-.L4
	.long	.L83-.L4
	.long	.L30-.L4
	.long	.L29-.L4
	.long	.L28-.L4
	.long	.L27-.L4
	.long	.L83-.L4
	.long	.L83-.L4
	.long	.L83-.L4
	.long	.L26-.L4
	.long	.L83-.L4
	.long	.L25-.L4
	.long	.L24-.L4
	.long	.L23-.L4
	.long	.L22-.L4
	.long	.L83-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L83-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L83-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L83-.L4
	.long	.L13-.L4
	.long	.L83-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L83-.L4
	.long	.L83-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L83-.L4
	.long	.L83-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L83-.L4
	.long	.L83-.L4
	.long	.L83-.L4
	.long	.L3-.L4
	.text
.L33:
	movq	-136(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -124(%rbp)
	movq	-144(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -112(%rbp)
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-144(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -40(%rbp)
	movq	-144(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	$0, -108(%rbp)
	movq	$46, -64(%rbp)
	jmp	.L47
.L13:
	cmpl	$0, -80(%rbp)
	jle	.L48
	movq	$23, -64(%rbp)
	jmp	.L47
.L48:
	movq	$25, -64(%rbp)
	jmp	.L47
.L30:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movl	-120(%rbp), %edx
	movl	%edx, (%rax)
	movq	$11, -64(%rbp)
	jmp	.L47
.L12:
	movl	-124(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -104(%rbp)
	movq	$34, -64(%rbp)
	jmp	.L47
.L43:
	movq	-144(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -124(%rbp)
	movq	-136(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -112(%rbp)
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-136(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -8(%rbp)
	movq	-136(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	$0, -96(%rbp)
	movq	$47, -64(%rbp)
	jmp	.L47
.L6:
	movq	-144(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-104(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	$63, -64(%rbp)
	jmp	.L47
.L36:
	cmpl	$0, -104(%rbp)
	jne	.L50
	movq	$62, -64(%rbp)
	jmp	.L47
.L50:
	movq	$63, -64(%rbp)
	jmp	.L47
.L35:
	movl	$0, -100(%rbp)
	movq	$40, -64(%rbp)
	jmp	.L47
.L10:
	movq	-136(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-144(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-84(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	addl	%eax, %edx
	movl	-120(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -116(%rbp)
	movl	$0, -120(%rbp)
	movq	$36, -64(%rbp)
	jmp	.L47
.L38:
	movl	-116(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$1717986919, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$2, %edx
	sarl	$31, %eax
	movl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, -120(%rbp)
	movl	-116(%rbp), %edx
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
	movl	%edx, -116(%rbp)
	movq	$48, -64(%rbp)
	jmp	.L47
.L40:
	cmpl	$0, -92(%rbp)
	jne	.L52
	movq	$43, -64(%rbp)
	jmp	.L47
.L52:
	movq	$7, -64(%rbp)
	jmp	.L47
.L17:
	movq	$26, -64(%rbp)
	jmp	.L47
.L31:
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
	movq	$50, -64(%rbp)
	jmp	.L47
.L44:
	movl	-124(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -88(%rbp)
	jge	.L54
	movq	$58, -64(%rbp)
	jmp	.L47
.L54:
	movq	$39, -64(%rbp)
	jmp	.L47
.L23:
	cmpl	$9, -116(%rbp)
	jle	.L56
	movq	$12, -64(%rbp)
	jmp	.L47
.L56:
	movq	$48, -64(%rbp)
	jmp	.L47
.L9:
	cmpl	$0, -92(%rbp)
	js	.L58
	movq	$8, -64(%rbp)
	jmp	.L47
.L58:
	movq	$27, -64(%rbp)
	jmp	.L47
.L29:
	movq	-136(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -124(%rbp)
	movl	$0, -76(%rbp)
	movl	$0, -120(%rbp)
	call	vlintegerCreate
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -72(%rbp)
	movq	$32, -64(%rbp)
	jmp	.L47
.L39:
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-136(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -24(%rbp)
	movq	-136(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	-124(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-144(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -16(%rbp)
	movq	-144(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$2, -64(%rbp)
	jmp	.L47
.L37:
	movl	-124(%rbp), %eax
	addl	$1, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -32(%rbp)
	movq	-72(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	-124(%rbp), %eax
	leal	1(%rax), %edx
	movq	-72(%rbp), %rax
	movl	%edx, (%rax)
	movl	-124(%rbp), %eax
	movl	%eax, -80(%rbp)
	movq	$50, -64(%rbp)
	jmp	.L47
.L5:
	cmpl	$0, -104(%rbp)
	je	.L60
	movq	$17, -64(%rbp)
	jmp	.L47
.L60:
	movq	$20, -64(%rbp)
	jmp	.L47
.L26:
	movq	-136(%rbp), %rax
	movl	(%rax), %edx
	movq	-144(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L62
	movq	$18, -64(%rbp)
	jmp	.L47
.L62:
	movq	$42, -64(%rbp)
	jmp	.L47
.L34:
	movq	-144(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-104(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-144(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-104(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	movq	$20, -64(%rbp)
	jmp	.L47
.L20:
	movl	-124(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -100(%rbp)
	jge	.L64
	movq	$37, -64(%rbp)
	jmp	.L47
.L64:
	movq	$42, -64(%rbp)
	jmp	.L47
.L3:
	movq	-144(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-136(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	movq	$28, -64(%rbp)
	jmp	.L47
.L28:
	addl	$1, -96(%rbp)
	movq	$47, -64(%rbp)
	jmp	.L47
.L7:
	movl	-124(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -92(%rbp)
	movq	$57, -64(%rbp)
	jmp	.L47
.L8:
	movq	-136(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-88(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -88(%rbp)
	movq	$3, -64(%rbp)
	jmp	.L47
.L25:
	cmpl	$0, -104(%rbp)
	js	.L66
	movq	$14, -64(%rbp)
	jmp	.L47
.L66:
	movq	$0, -64(%rbp)
	jmp	.L47
.L14:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-84(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rax, %rdx
	movl	-116(%rbp), %eax
	movl	%eax, (%rdx)
	subl	$1, -84(%rbp)
	movq	$5, -64(%rbp)
	jmp	.L47
.L27:
	subl	$1, -92(%rbp)
	movq	$57, -64(%rbp)
	jmp	.L47
.L11:
	cmpl	$0, -120(%rbp)
	jle	.L68
	movq	$13, -64(%rbp)
	jmp	.L47
.L68:
	movq	$11, -64(%rbp)
	jmp	.L47
.L15:
	movl	-124(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -96(%rbp)
	jge	.L70
	movq	$61, -64(%rbp)
	jmp	.L47
.L70:
	movq	$35, -64(%rbp)
	jmp	.L47
.L42:
	cmpl	$0, -84(%rbp)
	js	.L72
	movq	$56, -64(%rbp)
	jmp	.L47
.L72:
	movq	$53, -64(%rbp)
	jmp	.L47
.L22:
	movq	-144(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-100(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -100(%rbp)
	movq	$40, -64(%rbp)
	jmp	.L47
.L19:
	movq	-136(%rbp), %rax
	movl	(%rax), %edx
	movq	-144(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jge	.L74
	movq	$4, -64(%rbp)
	jmp	.L47
.L74:
	movq	$39, -64(%rbp)
	jmp	.L47
.L46:
	addl	$1, -108(%rbp)
	movq	$46, -64(%rbp)
	jmp	.L47
.L16:
	movl	-124(%rbp), %eax
	subl	-112(%rbp), %eax
	cmpl	%eax, -108(%rbp)
	jge	.L76
	movq	$52, -64(%rbp)
	jmp	.L47
.L76:
	movq	$15, -64(%rbp)
	jmp	.L47
.L21:
	movq	-72(%rbp), %rax
	movl	-124(%rbp), %edx
	movl	%edx, (%rax)
	movl	-124(%rbp), %eax
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
	movl	-124(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -84(%rbp)
	movq	$5, -64(%rbp)
	jmp	.L47
.L41:
	cmpl	$0, -92(%rbp)
	je	.L78
	movq	$67, -64(%rbp)
	jmp	.L47
.L78:
	movq	$28, -64(%rbp)
	jmp	.L47
.L24:
	movl	$0, -88(%rbp)
	movq	$3, -64(%rbp)
	jmp	.L47
.L18:
	movq	-136(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-92(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	movq	$7, -64(%rbp)
	jmp	.L47
.L45:
	movq	-72(%rbp), %rax
	jmp	.L82
.L32:
	subl	$1, -104(%rbp)
	movq	$34, -64(%rbp)
	jmp	.L47
.L83:
	nop
.L47:
	jmp	.L81
.L82:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	vlintegerAdd, .-vlintegerAdd
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"length=%d\n"
	.text
	.globl	vlintegerPrint
	.type	vlintegerPrint, @function
vlintegerPrint:
.LFB1:
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
.L97:
	cmpq	$8, -8(%rbp)
	ja	.L98
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L87(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L87(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L87:
	.long	.L98-.L87
	.long	.L92-.L87
	.long	.L98-.L87
	.long	.L91-.L87
	.long	.L99-.L87
	.long	.L89-.L87
	.long	.L88-.L87
	.long	.L98-.L87
	.long	.L86-.L87
	.text
.L86:
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
	movq	$3, -8(%rbp)
	jmp	.L94
.L92:
	movl	$10, %edi
	call	putchar@PLT
	movq	$4, -8(%rbp)
	jmp	.L94
.L91:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L95
	movq	$8, -8(%rbp)
	jmp	.L94
.L95:
	movq	$1, -8(%rbp)
	jmp	.L94
.L88:
	movq	$5, -8(%rbp)
	jmp	.L94
.L89:
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L94
.L98:
	nop
.L94:
	jmp	.L97
.L99:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	vlintegerPrint, .-vlintegerPrint
	.globl	vlintegerCreate
	.type	vlintegerCreate, @function
vlintegerCreate:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$1, -16(%rbp)
.L106:
	cmpq	$2, -16(%rbp)
	je	.L101
	cmpq	$2, -16(%rbp)
	ja	.L108
	cmpq	$0, -16(%rbp)
	je	.L103
	cmpq	$1, -16(%rbp)
	jne	.L108
	movq	$0, -16(%rbp)
	jmp	.L104
.L103:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-24(%rbp), %rax
	movl	$0, (%rax)
	movq	$2, -16(%rbp)
	jmp	.L104
.L101:
	movq	-24(%rbp), %rax
	jmp	.L107
.L108:
	nop
.L104:
	jmp	.L106
.L107:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	vlintegerCreate, .-vlintegerCreate
	.globl	vlintegerMult
	.type	vlintegerMult, @function
vlintegerMult:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -88(%rbp)
	movq	%rsi, -96(%rbp)
	movq	$23, -32(%rbp)
.L135:
	cmpq	$25, -32(%rbp)
	ja	.L137
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L112(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L112(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L112:
	.long	.L137-.L112
	.long	.L137-.L112
	.long	.L124-.L112
	.long	.L137-.L112
	.long	.L137-.L112
	.long	.L123-.L112
	.long	.L122-.L112
	.long	.L137-.L112
	.long	.L121-.L112
	.long	.L137-.L112
	.long	.L137-.L112
	.long	.L137-.L112
	.long	.L137-.L112
	.long	.L120-.L112
	.long	.L119-.L112
	.long	.L137-.L112
	.long	.L137-.L112
	.long	.L137-.L112
	.long	.L118-.L112
	.long	.L117-.L112
	.long	.L137-.L112
	.long	.L116-.L112
	.long	.L115-.L112
	.long	.L114-.L112
	.long	.L113-.L112
	.long	.L111-.L112
	.text
.L118:
	cmpl	$0, -52(%rbp)
	js	.L125
	movq	$21, -32(%rbp)
	jmp	.L127
.L125:
	movq	$24, -32(%rbp)
	jmp	.L127
.L111:
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
	movq	$14, -32(%rbp)
	jmp	.L127
.L119:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jne	.L128
	movq	$19, -32(%rbp)
	jmp	.L127
.L128:
	movq	$5, -32(%rbp)
	jmp	.L127
.L121:
	cmpl	$0, -56(%rbp)
	js	.L130
	movq	$2, -32(%rbp)
	jmp	.L127
.L130:
	movq	$14, -32(%rbp)
	jmp	.L127
.L114:
	movq	$6, -32(%rbp)
	jmp	.L127
.L113:
	addl	$1, -68(%rbp)
	subl	$1, -56(%rbp)
	movq	$8, -32(%rbp)
	jmp	.L127
.L116:
	movq	-96(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-56(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movq	-88(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-52(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	imull	%eax, %edx
	movq	-40(%rbp), %rax
	movq	8(%rax), %rcx
	movl	-64(%rbp), %eax
	cltq
	salq	$2, %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	addl	%edx, %eax
	movl	%eax, -44(%rbp)
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-64(%rbp), %eax
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
	movl	-64(%rbp), %eax
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
	movl	-64(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	addq	%rdx, %rax
	leal	(%rcx,%rsi), %edx
	movl	%edx, (%rax)
	subl	$1, -64(%rbp)
	subl	$1, -52(%rbp)
	movq	$18, -32(%rbp)
	jmp	.L127
.L120:
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-48(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	(%rdx,%rax), %rcx
	movq	-40(%rbp), %rax
	movq	8(%rax), %rdx
	movl	-48(%rbp), %eax
	cltq
	salq	$2, %rax
	subq	$4, %rax
	addq	%rax, %rdx
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	addl	$1, -48(%rbp)
	movq	$22, -32(%rbp)
	jmp	.L127
.L117:
	movl	$1, -48(%rbp)
	movq	$22, -32(%rbp)
	jmp	.L127
.L122:
	movq	-88(%rbp), %rax
	movl	(%rax), %edx
	movq	-96(%rbp), %rax
	movl	(%rax), %eax
	addl	%edx, %eax
	movl	%eax, -60(%rbp)
	call	vlintegerCreate
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	-60(%rbp), %eax
	cltq
	movl	$4, %esi
	movq	%rax, %rdi
	call	calloc@PLT
	movq	%rax, -8(%rbp)
	movq	-40(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-40(%rbp), %rax
	movl	-60(%rbp), %edx
	movl	%edx, (%rax)
	movl	$1, -68(%rbp)
	movq	-96(%rbp), %rax
	movl	(%rax), %eax
	subl	$1, %eax
	movl	%eax, -56(%rbp)
	movq	$8, -32(%rbp)
	jmp	.L127
.L115:
	movq	-40(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -48(%rbp)
	jge	.L132
	movq	$13, -32(%rbp)
	jmp	.L127
.L132:
	movq	$25, -32(%rbp)
	jmp	.L127
.L123:
	movq	-40(%rbp), %rax
	jmp	.L136
.L124:
	movl	-60(%rbp), %eax
	subl	-68(%rbp), %eax
	movl	%eax, -64(%rbp)
	movq	-88(%rbp), %rax
	movl	(%rax), %eax
	subl	$1, %eax
	movl	%eax, -52(%rbp)
	movq	$18, -32(%rbp)
	jmp	.L127
.L137:
	nop
.L127:
	jmp	.L135
.L136:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	vlintegerMult, .-vlintegerMult
	.section	.rodata
.LC2:
	.string	"%c"
.LC3:
	.string	"multiplication"
.LC4:
	.string	"main"
.LC5:
	.string	"elaimizrahi_CS-138_j.c"
.LC6:
	.string	"mult->arr[0] !=0"
.LC7:
	.string	"add->arr[0] !=0"
.LC8:
	.string	"addition"
	.align 8
.LC9:
	.string	"Enter the digits separated by a space: "
.LC10:
	.string	"int1->arr[0] !=0"
.LC11:
	.string	"int2->arr[0] !=0"
	.text
	.globl	main
	.type	main, @function
main:
.LFB7:
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
	movq	$0, _TIG_IZ_Defe_envp(%rip)
	nop
.L139:
	movq	$0, _TIG_IZ_Defe_argv(%rip)
	nop
.L140:
	movl	$0, _TIG_IZ_Defe_argc(%rip)
	nop
	nop
.L141:
.L142:
#APP
# 115 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Defe--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_Defe_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_Defe_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_Defe_envp(%rip)
	nop
	movq	$6, -48(%rbp)
.L170:
	cmpq	$18, -48(%rbp)
	ja	.L173
	movq	-48(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L145(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L145(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L145:
	.long	.L159-.L145
	.long	.L158-.L145
	.long	.L173-.L145
	.long	.L157-.L145
	.long	.L156-.L145
	.long	.L155-.L145
	.long	.L154-.L145
	.long	.L153-.L145
	.long	.L173-.L145
	.long	.L152-.L145
	.long	.L151-.L145
	.long	.L150-.L145
	.long	.L149-.L145
	.long	.L148-.L145
	.long	.L147-.L145
	.long	.L173-.L145
	.long	.L173-.L145
	.long	.L146-.L145
	.long	.L144-.L145
	.text
.L144:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L171
	jmp	.L172
.L156:
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
	movq	$18, -48(%rbp)
	jmp	.L161
.L147:
	leaq	-81(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-72(%rbp), %rdx
	movq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	vlintegerMult
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -56(%rbp)
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$11, -48(%rbp)
	jmp	.L161
.L149:
	movq	-80(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L162
	movq	$5, -48(%rbp)
	jmp	.L161
.L162:
	movq	$10, -48(%rbp)
	jmp	.L161
.L158:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rcx
	movl	$361, %edx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L157:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rcx
	movl	$356, %edx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L150:
	movq	-56(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L164
	movq	$4, -48(%rbp)
	jmp	.L161
.L164:
	movq	$1, -48(%rbp)
	jmp	.L161
.L152:
	movq	-72(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L166
	movq	$17, -48(%rbp)
	jmp	.L161
.L166:
	movq	$7, -48(%rbp)
	jmp	.L161
.L148:
	movq	-64(%rbp), %rax
	movq	8(%rax), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	je	.L168
	movq	$14, -48(%rbp)
	jmp	.L161
.L168:
	movq	$3, -48(%rbp)
	jmp	.L161
.L146:
	leaq	-81(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-72(%rbp), %rdx
	movq	-80(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	vlintegerAdd
	movq	%rax, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, -64(%rbp)
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$13, -48(%rbp)
	jmp	.L161
.L154:
	movq	$0, -48(%rbp)
	jmp	.L161
.L155:
	leaq	-81(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	call	vlintegerCreate
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -72(%rbp)
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerRead
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$9, -48(%rbp)
	jmp	.L161
.L151:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rcx
	movl	$342, %edx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L159:
	call	vlintegerCreate
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerRead
	movq	-80(%rbp), %rax
	movq	%rax, %rdi
	call	vlintegerPrint
	movq	$12, -48(%rbp)
	jmp	.L161
.L153:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rcx
	movl	$351, %edx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L173:
	nop
.L161:
	jmp	.L170
.L172:
	call	__stack_chk_fail@PLT
.L171:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
	.globl	vlintegerRead
	.type	vlintegerRead, @function
vlintegerRead:
.LFB10:
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
	movq	$1, -32(%rbp)
.L187:
	cmpq	$9, -32(%rbp)
	ja	.L190
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L177(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L177(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L177:
	.long	.L190-.L177
	.long	.L182-.L177
	.long	.L191-.L177
	.long	.L180-.L177
	.long	.L179-.L177
	.long	.L178-.L177
	.long	.L190-.L177
	.long	.L190-.L177
	.long	.L190-.L177
	.long	.L176-.L177
	.text
.L179:
	cmpl	$0, -36(%rbp)
	je	.L183
	movq	$5, -32(%rbp)
	jmp	.L185
.L183:
	movq	$9, -32(%rbp)
	jmp	.L185
.L182:
	movl	$0, -40(%rbp)
	movq	$3, -32(%rbp)
	jmp	.L185
.L180:
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	%eax, -36(%rbp)
	movq	$4, -32(%rbp)
	jmp	.L185
.L176:
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
	movq	$2, -32(%rbp)
	jmp	.L185
.L178:
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
	movq	$3, -32(%rbp)
	jmp	.L185
.L190:
	nop
.L185:
	jmp	.L187
.L191:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L189
	call	__stack_chk_fail@PLT
.L189:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	vlintegerRead, .-vlintegerRead
	.globl	vlintegerDestroy
	.type	vlintegerDestroy, @function
vlintegerDestroy:
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
	movq	$1, -8(%rbp)
.L201:
	cmpq	$3, -8(%rbp)
	je	.L193
	cmpq	$3, -8(%rbp)
	ja	.L202
	cmpq	$1, -8(%rbp)
	je	.L195
	cmpq	$2, -8(%rbp)
	je	.L203
	jmp	.L202
.L195:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L197
	movq	$3, -8(%rbp)
	jmp	.L199
.L197:
	movq	$2, -8(%rbp)
	jmp	.L199
.L193:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$2, -8(%rbp)
	jmp	.L199
.L202:
	nop
.L199:
	jmp	.L201
.L203:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	vlintegerDestroy, .-vlintegerDestroy
	.globl	power
	.type	power, @function
power:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$6, -8(%rbp)
.L216:
	cmpq	$7, -8(%rbp)
	ja	.L218
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L207(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L207(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L207:
	.long	.L211-.L207
	.long	.L210-.L207
	.long	.L209-.L207
	.long	.L218-.L207
	.long	.L218-.L207
	.long	.L218-.L207
	.long	.L208-.L207
	.long	.L206-.L207
	.text
.L210:
	movl	-16(%rbp), %eax
	imull	-20(%rbp), %eax
	movl	%eax, -16(%rbp)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L212
.L208:
	movq	$0, -8(%rbp)
	jmp	.L212
.L211:
	movl	$1, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L212
.L206:
	movl	-16(%rbp), %eax
	jmp	.L217
.L209:
	movl	-12(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jge	.L214
	movq	$1, -8(%rbp)
	jmp	.L212
.L214:
	movq	$7, -8(%rbp)
	jmp	.L212
.L218:
	nop
.L212:
	jmp	.L216
.L217:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	power, .-power
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
