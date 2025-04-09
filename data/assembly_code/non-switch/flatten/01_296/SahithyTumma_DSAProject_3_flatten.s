	.file	"SahithyTumma_DSAProject_3_flatten.c"
	.text
	.globl	time
	.bss
	.align 4
	.type	time, @object
	.size	time, 4
time:
	.zero	4
	.globl	_TIG_IZ_XpTM_argc
	.align 4
	.type	_TIG_IZ_XpTM_argc, @object
	.size	_TIG_IZ_XpTM_argc, 4
_TIG_IZ_XpTM_argc:
	.zero	4
	.globl	_TIG_IZ_XpTM_argv
	.align 8
	.type	_TIG_IZ_XpTM_argv, @object
	.size	_TIG_IZ_XpTM_argv, 8
_TIG_IZ_XpTM_argv:
	.zero	8
	.globl	_TIG_IZ_XpTM_envp
	.align 8
	.type	_TIG_IZ_XpTM_envp, @object
	.size	_TIG_IZ_XpTM_envp, 8
_TIG_IZ_XpTM_envp:
	.zero	8
	.globl	l
	.align 4
	.type	l, @object
	.size	l, 4
l:
	.zero	4
	.text
	.globl	distance
	.type	distance, @function
distance:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%rcx, -64(%rbp)
	movq	%r8, -72(%rbp)
	movq	%r9, -80(%rbp)
	movq	$4, -8(%rbp)
.L39:
	cmpq	$26, -8(%rbp)
	ja	.L40
	movq	-8(%rbp), %rax
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
	.long	.L41-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L40-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L40-.L4
	.long	.L40-.L4
	.long	.L15-.L4
	.long	.L40-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L40-.L4
	.long	.L8-.L4
	.long	.L40-.L4
	.long	.L40-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L40-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	movl	-16(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L23
	movq	$16, -8(%rbp)
	jmp	.L25
.L23:
	movq	$12, -8(%rbp)
	jmp	.L25
.L19:
	movq	$15, -8(%rbp)
	jmp	.L25
.L12:
	movl	$0, -16(%rbp)
	movq	$25, -8(%rbp)
	jmp	.L25
.L11:
	movl	-44(%rbp), %eax
	imull	%eax, %eax
	movl	%eax, -12(%rbp)
	movq	-56(%rbp), %rax
	movl	(%rax), %edx
	movq	-72(%rbp), %rax
	movl	%edx, (%rax)
	movq	-64(%rbp), %rax
	movl	(%rax), %edx
	movq	-80(%rbp), %rax
	movl	%edx, (%rax)
	movl	$0, -20(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L25
.L14:
	addl	$1, -20(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L25
.L21:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	36(%rax), %eax
	testl	%eax, %eax
	jle	.L26
	movq	$19, -8(%rbp)
	jmp	.L25
.L26:
	movq	$7, -8(%rbp)
	jmp	.L25
.L6:
	movq	-56(%rbp), %rax
	movl	(%rax), %ecx
	movl	-20(%rbp), %eax
	subl	%ecx, %eax
	movl	%eax, %edx
	movq	-56(%rbp), %rax
	movl	(%rax), %ecx
	movl	-20(%rbp), %eax
	subl	%ecx, %eax
	movl	%edx, %esi
	imull	%eax, %esi
	movq	-64(%rbp), %rax
	movl	(%rax), %ecx
	movl	-16(%rbp), %eax
	subl	%ecx, %eax
	movl	%eax, %edx
	movq	-64(%rbp), %rax
	movl	(%rax), %ecx
	movl	-16(%rbp), %eax
	subl	%ecx, %eax
	movl	%edx, %ecx
	imull	%eax, %ecx
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdi
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rdi, %rax
	leal	(%rsi,%rcx), %edx
	movl	%edx, 36(%rax)
	movq	$26, -8(%rbp)
	jmp	.L25
.L10:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	36(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jl	.L28
	movq	$1, -8(%rbp)
	jmp	.L25
.L28:
	movq	$7, -8(%rbp)
	jmp	.L25
.L3:
	addl	$1, -16(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L25
.L13:
	movl	-20(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L30
	movq	$14, -8(%rbp)
	jmp	.L25
.L30:
	movq	$0, -8(%rbp)
	jmp	.L25
.L8:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	36(%rax), %eax
	movl	%eax, -12(%rbp)
	movq	-56(%rbp), %rax
	movl	-20(%rbp), %edx
	movl	%edx, (%rax)
	movq	-64(%rbp), %rax
	movl	-16(%rbp), %edx
	movl	%edx, (%rax)
	movq	$7, -8(%rbp)
	jmp	.L25
.L9:
	movl	-20(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L32
	movq	$2, -8(%rbp)
	jmp	.L25
.L32:
	movq	$6, -8(%rbp)
	jmp	.L25
.L17:
	movl	$0, -20(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L25
.L7:
	movl	-16(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L34
	movq	$5, -8(%rbp)
	jmp	.L25
.L34:
	movq	$10, -8(%rbp)
	jmp	.L25
.L18:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-16(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	36(%rax), %eax
	testl	%eax, %eax
	jle	.L36
	movq	$23, -8(%rbp)
	jmp	.L25
.L36:
	movq	$26, -8(%rbp)
	jmp	.L25
.L15:
	addl	$1, -20(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L25
.L16:
	addl	$1, -16(%rbp)
	movq	$25, -8(%rbp)
	jmp	.L25
.L20:
	movl	$0, -16(%rbp)
	movq	$22, -8(%rbp)
	jmp	.L25
.L40:
	nop
.L25:
	jmp	.L39
.L41:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	distance, .-distance
	.section	.rodata
.LC0:
	.string	"move up"
.LC1:
	.string	"move down"
.LC2:
	.string	"move left"
.LC3:
	.string	"move right"
	.text
	.globl	print
	.type	print, @function
print:
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
	movl	%edx, -28(%rbp)
	movl	%ecx, -32(%rbp)
	movq	$24, -8(%rbp)
.L77:
	cmpq	$27, -8(%rbp)
	ja	.L78
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L45(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L45(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L45:
	.long	.L78-.L45
	.long	.L79-.L45
	.long	.L57-.L45
	.long	.L78-.L45
	.long	.L78-.L45
	.long	.L56-.L45
	.long	.L78-.L45
	.long	.L55-.L45
	.long	.L78-.L45
	.long	.L78-.L45
	.long	.L78-.L45
	.long	.L78-.L45
	.long	.L54-.L45
	.long	.L53-.L45
	.long	.L78-.L45
	.long	.L52-.L45
	.long	.L78-.L45
	.long	.L51-.L45
	.long	.L50-.L45
	.long	.L49-.L45
	.long	.L78-.L45
	.long	.L48-.L45
	.long	.L78-.L45
	.long	.L78-.L45
	.long	.L47-.L45
	.long	.L78-.L45
	.long	.L46-.L45
	.long	.L44-.L45
	.text
.L50:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	time(%rip), %eax
	subl	$1, %eax
	movl	%eax, time(%rip)
	addl	$1, -24(%rbp)
	movq	$27, -8(%rbp)
	jmp	.L59
.L52:
	movl	time(%rip), %eax
	testl	%eax, %eax
	jle	.L60
	movq	$18, -8(%rbp)
	jmp	.L59
.L60:
	movq	$13, -8(%rbp)
	jmp	.L59
.L54:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	time(%rip), %eax
	subl	$1, %eax
	movl	%eax, time(%rip)
	subl	$1, -24(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L59
.L47:
	movq	$17, -8(%rbp)
	jmp	.L59
.L48:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	time(%rip), %eax
	subl	$1, %eax
	movl	%eax, time(%rip)
	subl	$1, -20(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L59
.L46:
	movl	time(%rip), %eax
	testl	%eax, %eax
	jle	.L63
	movq	$7, -8(%rbp)
	jmp	.L59
.L63:
	movq	$27, -8(%rbp)
	jmp	.L59
.L53:
	movl	-20(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jle	.L65
	movq	$5, -8(%rbp)
	jmp	.L59
.L65:
	movq	$19, -8(%rbp)
	jmp	.L59
.L49:
	movl	-24(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jle	.L67
	movq	$2, -8(%rbp)
	jmp	.L59
.L67:
	movq	$1, -8(%rbp)
	jmp	.L59
.L51:
	movl	-20(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L69
	movq	$26, -8(%rbp)
	jmp	.L59
.L69:
	movq	$27, -8(%rbp)
	jmp	.L59
.L44:
	movl	-24(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jge	.L71
	movq	$15, -8(%rbp)
	jmp	.L59
.L71:
	movq	$13, -8(%rbp)
	jmp	.L59
.L56:
	movl	time(%rip), %eax
	testl	%eax, %eax
	jle	.L73
	movq	$21, -8(%rbp)
	jmp	.L59
.L73:
	movq	$19, -8(%rbp)
	jmp	.L59
.L55:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	time(%rip), %eax
	subl	$1, %eax
	movl	%eax, time(%rip)
	addl	$1, -20(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L59
.L57:
	movl	time(%rip), %eax
	testl	%eax, %eax
	jle	.L75
	movq	$12, -8(%rbp)
	jmp	.L59
.L75:
	movq	$1, -8(%rbp)
	jmp	.L59
.L78:
	nop
.L59:
	jmp	.L77
.L79:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	print, .-print
	.section	.rodata
.LC4:
	.string	"%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
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
	movl	$0, time(%rip)
	nop
.L81:
	movl	$0, l(%rip)
	nop
.L82:
	movq	$0, _TIG_IZ_XpTM_envp(%rip)
	nop
.L83:
	movq	$0, _TIG_IZ_XpTM_argv(%rip)
	nop
.L84:
	movl	$0, _TIG_IZ_XpTM_argc(%rip)
	nop
	nop
.L85:
.L86:
#APP
# 115 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-XpTM--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_XpTM_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_XpTM_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_XpTM_envp(%rip)
	nop
	movq	$20, -32(%rbp)
.L134:
	cmpq	$43, -32(%rbp)
	ja	.L137
	movq	-32(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L89(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L89(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L89:
	.long	.L113-.L89
	.long	.L112-.L89
	.long	.L111-.L89
	.long	.L137-.L89
	.long	.L137-.L89
	.long	.L110-.L89
	.long	.L137-.L89
	.long	.L109-.L89
	.long	.L137-.L89
	.long	.L137-.L89
	.long	.L108-.L89
	.long	.L107-.L89
	.long	.L106-.L89
	.long	.L105-.L89
	.long	.L104-.L89
	.long	.L137-.L89
	.long	.L103-.L89
	.long	.L102-.L89
	.long	.L101-.L89
	.long	.L100-.L89
	.long	.L99-.L89
	.long	.L137-.L89
	.long	.L98-.L89
	.long	.L137-.L89
	.long	.L137-.L89
	.long	.L97-.L89
	.long	.L137-.L89
	.long	.L137-.L89
	.long	.L96-.L89
	.long	.L137-.L89
	.long	.L137-.L89
	.long	.L137-.L89
	.long	.L137-.L89
	.long	.L95-.L89
	.long	.L94-.L89
	.long	.L93-.L89
	.long	.L92-.L89
	.long	.L91-.L89
	.long	.L137-.L89
	.long	.L90-.L89
	.long	.L137-.L89
	.long	.L137-.L89
	.long	.L137-.L89
	.long	.L88-.L89
	.text
.L101:
	movl	-80(%rbp), %eax
	cmpl	%eax, -48(%rbp)
	jge	.L114
	movq	$25, -32(%rbp)
	jmp	.L116
.L114:
	movq	$28, -32(%rbp)
	jmp	.L116
.L97:
	movl	$0, -44(%rbp)
	movq	$37, -32(%rbp)
	jmp	.L116
.L104:
	movl	time(%rip), %eax
	testl	%eax, %eax
	jle	.L117
	movq	$13, -32(%rbp)
	jmp	.L116
.L117:
	movq	$35, -32(%rbp)
	jmp	.L116
.L106:
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 28(%rax)
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 32(%rax)
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	l(%rip), %eax
	subl	$1, %eax
	movl	%eax, l(%rip)
	movq	$11, -32(%rbp)
	jmp	.L116
.L112:
	movl	-80(%rbp), %eax
	cmpl	%eax, -48(%rbp)
	jge	.L119
	movq	$36, -32(%rbp)
	jmp	.L116
.L119:
	movq	$0, -32(%rbp)
	jmp	.L116
.L103:
	movl	time(%rip), %eax
	testl	%eax, %eax
	jle	.L121
	movq	$39, -32(%rbp)
	jmp	.L116
.L121:
	movq	$35, -32(%rbp)
	jmp	.L116
.L92:
	movl	-80(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	%rax, (%rdx)
	addl	$1, -48(%rbp)
	movq	$1, -32(%rbp)
	jmp	.L116
.L107:
	addl	$1, -48(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L116
.L105:
	movl	-80(%rbp), %esi
	leaq	-52(%rbp), %r8
	leaq	-56(%rbp), %rdi
	leaq	-60(%rbp), %rcx
	leaq	-64(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%r8, %r9
	movq	%rdi, %r8
	movq	%rax, %rdi
	call	distance
	movq	$5, -32(%rbp)
	jmp	.L116
.L100:
	movl	l(%rip), %eax
	testl	%eax, %eax
	jle	.L123
	movq	$14, -32(%rbp)
	jmp	.L116
.L123:
	movq	$35, -32(%rbp)
	jmp	.L116
.L102:
	leaq	-84(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-80(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-76(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-80(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movl	$0, -48(%rbp)
	movq	$1, -32(%rbp)
	jmp	.L116
.L94:
	addl	$1, -48(%rbp)
	movq	$18, -32(%rbp)
	jmp	.L116
.L98:
	leaq	-72(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-68(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	addq	$8, %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	addq	$12, %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	addq	$16, %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	addq	$20, %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-72(%rbp), %edx
	movl	-72(%rbp), %eax
	movl	%edx, %esi
	imull	%eax, %esi
	movl	-68(%rbp), %edx
	movl	-68(%rbp), %eax
	movl	%edx, %ecx
	imull	%eax, %ecx
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdi
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rdi, %rax
	leal	(%rsi,%rcx), %edx
	movl	%edx, 36(%rax)
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	(%rcx,%rax), %rdx
	movl	-72(%rbp), %eax
	movl	%eax, (%rdx)
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	(%rcx,%rax), %rdx
	movl	-68(%rbp), %eax
	movl	%eax, 4(%rdx)
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$1, 24(%rax)
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	20(%rax), %ecx
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	8(%rax), %eax
	movl	%ecx, %esi
	imull	%eax, %esi
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	12(%rax), %ecx
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdi
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	(%rdi,%rax), %rdx
	movl	%esi, %eax
	imull	%ecx, %eax
	movl	%eax, 28(%rdx)
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	16(%rax), %ecx
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	12(%rax), %eax
	movl	%ecx, %esi
	imull	%eax, %esi
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	8(%rax), %ecx
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdi
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	(%rdi,%rax), %rdx
	movl	%esi, %eax
	imull	%ecx, %eax
	movl	%eax, 32(%rdx)
	movq	$10, -32(%rbp)
	jmp	.L116
.L96:
	movl	$0, -48(%rbp)
	movq	$33, -32(%rbp)
	jmp	.L116
.L110:
	movl	time(%rip), %eax
	testl	%eax, %eax
	jle	.L125
	movq	$43, -32(%rbp)
	jmp	.L116
.L125:
	movq	$35, -32(%rbp)
	jmp	.L116
.L95:
	movl	-76(%rbp), %eax
	cmpl	%eax, -48(%rbp)
	jge	.L127
	movq	$22, -32(%rbp)
	jmp	.L116
.L127:
	movq	$2, -32(%rbp)
	jmp	.L116
.L91:
	movl	-80(%rbp), %eax
	cmpl	%eax, -44(%rbp)
	jge	.L129
	movq	$7, -32(%rbp)
	jmp	.L116
.L129:
	movq	$34, -32(%rbp)
	jmp	.L116
.L108:
	movl	-72(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-68(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	36(%rax), %eax
	testl	%eax, %eax
	jne	.L131
	movq	$12, -32(%rbp)
	jmp	.L116
.L131:
	movq	$11, -32(%rbp)
	jmp	.L116
.L113:
	movl	-76(%rbp), %eax
	movl	%eax, l(%rip)
	movl	-84(%rbp), %eax
	movl	%eax, time(%rip)
	movl	$0, -48(%rbp)
	movq	$18, -32(%rbp)
	jmp	.L116
.L90:
	movl	-80(%rbp), %ecx
	movl	-60(%rbp), %edx
	movl	-64(%rbp), %esi
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	calculation
	movq	$19, -32(%rbp)
	jmp	.L116
.L109:
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	(%rcx,%rax), %rdx
	movl	-48(%rbp), %eax
	movl	%eax, (%rdx)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	(%rcx,%rax), %rdx
	movl	-44(%rbp), %eax
	movl	%eax, 4(%rdx)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 8(%rax)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 12(%rax)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 16(%rax)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 20(%rax)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 28(%rax)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 32(%rax)
	movl	-48(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-44(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	addl	$1, -44(%rbp)
	movq	$37, -32(%rbp)
	jmp	.L116
.L93:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L135
	jmp	.L136
.L88:
	movl	-60(%rbp), %ecx
	movl	-64(%rbp), %edx
	movl	-52(%rbp), %esi
	movl	-56(%rbp), %eax
	movl	%eax, %edi
	call	print
	movq	$16, -32(%rbp)
	jmp	.L116
.L111:
	movl	$0, -64(%rbp)
	movl	$0, -60(%rbp)
	movq	$19, -32(%rbp)
	jmp	.L116
.L99:
	movq	$17, -32(%rbp)
	jmp	.L116
.L137:
	nop
.L116:
	jmp	.L134
.L136:
	call	__stack_chk_fail@PLT
.L135:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC5:
	.string	"cut left"
.LC6:
	.string	"cut down"
.LC7:
	.string	"cut up"
.LC8:
	.string	"cut right"
	.text
	.globl	calculation
	.type	calculation, @function
calculation:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movl	%ecx, -36(%rbp)
	movq	$16, -8(%rbp)
.L213:
	cmpq	$39, -8(%rbp)
	ja	.L214
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L141(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L141(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L141:
	.long	.L174-.L141
	.long	.L173-.L141
	.long	.L172-.L141
	.long	.L214-.L141
	.long	.L171-.L141
	.long	.L214-.L141
	.long	.L215-.L141
	.long	.L215-.L141
	.long	.L168-.L141
	.long	.L215-.L141
	.long	.L166-.L141
	.long	.L165-.L141
	.long	.L164-.L141
	.long	.L163-.L141
	.long	.L162-.L141
	.long	.L161-.L141
	.long	.L160-.L141
	.long	.L215-.L141
	.long	.L214-.L141
	.long	.L214-.L141
	.long	.L158-.L141
	.long	.L157-.L141
	.long	.L156-.L141
	.long	.L215-.L141
	.long	.L215-.L141
	.long	.L153-.L141
	.long	.L152-.L141
	.long	.L214-.L141
	.long	.L215-.L141
	.long	.L150-.L141
	.long	.L215-.L141
	.long	.L148-.L141
	.long	.L147-.L141
	.long	.L215-.L141
	.long	.L145-.L141
	.long	.L144-.L141
	.long	.L143-.L141
	.long	.L214-.L141
	.long	.L142-.L141
	.long	.L140-.L141
	.text
.L153:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L175
	movq	$31, -8(%rbp)
	jmp	.L177
.L175:
	movq	$10, -8(%rbp)
	jmp	.L177
.L171:
	addl	$1, -12(%rbp)
	movq	$26, -8(%rbp)
	jmp	.L177
.L162:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jle	.L179
	movq	$8, -8(%rbp)
	jmp	.L177
.L179:
	movq	$4, -8(%rbp)
	jmp	.L177
.L161:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L181
	movq	$14, -8(%rbp)
	jmp	.L177
.L181:
	movq	$4, -8(%rbp)
	jmp	.L177
.L148:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jle	.L183
	movq	$34, -8(%rbp)
	jmp	.L177
.L183:
	movq	$10, -8(%rbp)
	jmp	.L177
.L164:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	12(%rax), %edx
	movl	time(%rip), %eax
	cmpl	%eax, %edx
	jg	.L185
	movq	$38, -8(%rbp)
	jmp	.L177
.L185:
	movq	$6, -8(%rbp)
	jmp	.L177
.L168:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	12(%rax), %edx
	movl	time(%rip), %eax
	cmpl	%eax, %edx
	jg	.L187
	movq	$13, -8(%rbp)
	jmp	.L177
.L187:
	movq	$28, -8(%rbp)
	jmp	.L177
.L173:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jle	.L189
	movq	$0, -8(%rbp)
	jmp	.L177
.L189:
	movq	$32, -8(%rbp)
	jmp	.L177
.L160:
	movl	$1, -12(%rbp)
	movq	$26, -8(%rbp)
	jmp	.L177
.L157:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	time(%rip), %ecx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	12(%rax), %edx
	movl	%ecx, %eax
	subl	%edx, %eax
	movl	%eax, time(%rip)
	movl	l(%rip), %eax
	subl	$2, %eax
	movl	%eax, l(%rip)
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	movl	%eax, %esi
	movl	-36(%rbp), %ecx
	movl	-32(%rbp), %edx
	movq	-24(%rbp), %rax
	movl	%ecx, %r8d
	movl	$3, %ecx
	movq	%rax, %rdi
	call	calc
	movq	$9, -8(%rbp)
	jmp	.L177
.L143:
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L191
	movq	$2, -8(%rbp)
	jmp	.L177
.L191:
	movq	$11, -8(%rbp)
	jmp	.L177
.L152:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	8(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L193
	movq	$29, -8(%rbp)
	jmp	.L177
.L193:
	movq	$12, -8(%rbp)
	jmp	.L177
.L165:
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	testl	%eax, %eax
	js	.L195
	movq	$15, -8(%rbp)
	jmp	.L177
.L195:
	movq	$4, -8(%rbp)
	jmp	.L177
.L163:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	time(%rip), %ecx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	12(%rax), %edx
	movl	%ecx, %eax
	subl	%edx, %eax
	movl	%eax, time(%rip)
	movl	l(%rip), %eax
	subl	$2, %eax
	movl	%eax, l(%rip)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movl	%eax, %edx
	movl	-36(%rbp), %ecx
	movl	-28(%rbp), %esi
	movq	-24(%rbp), %rax
	movl	%ecx, %r8d
	movl	$4, %ecx
	movq	%rax, %rdi
	call	calc
	movq	$7, -8(%rbp)
	jmp	.L177
.L147:
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cmpl	%eax, -36(%rbp)
	jle	.L197
	movq	$25, -8(%rbp)
	jmp	.L177
.L197:
	movq	$10, -8(%rbp)
	jmp	.L177
.L142:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	l(%rip), %eax
	subl	$1, %eax
	movl	%eax, l(%rip)
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	time(%rip), %ecx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	12(%rax), %edx
	movl	%ecx, %eax
	subl	%edx, %eax
	movl	%eax, time(%rip)
	movq	$6, -8(%rbp)
	jmp	.L177
.L145:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	12(%rax), %edx
	movl	time(%rip), %eax
	cmpl	%eax, %edx
	jg	.L199
	movq	$20, -8(%rbp)
	jmp	.L177
.L199:
	movq	$17, -8(%rbp)
	jmp	.L177
.L156:
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L201
	movq	$1, -8(%rbp)
	jmp	.L177
.L201:
	movq	$32, -8(%rbp)
	jmp	.L177
.L166:
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	testl	%eax, %eax
	js	.L203
	movq	$36, -8(%rbp)
	jmp	.L177
.L203:
	movq	$11, -8(%rbp)
	jmp	.L177
.L174:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	12(%rax), %edx
	movl	time(%rip), %eax
	cmpl	%eax, %edx
	jg	.L205
	movq	$39, -8(%rbp)
	jmp	.L177
.L205:
	movq	$33, -8(%rbp)
	jmp	.L177
.L140:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	time(%rip), %ecx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	12(%rax), %edx
	movl	%ecx, %eax
	subl	%edx, %eax
	movl	%eax, time(%rip)
	movl	l(%rip), %eax
	subl	$2, %eax
	movl	%eax, l(%rip)
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	leal	(%rdx,%rax), %esi
	movl	-36(%rbp), %ecx
	movl	-32(%rbp), %edx
	movq	-24(%rbp), %rax
	movl	%ecx, %r8d
	movl	$1, %ecx
	movq	%rax, %rdi
	call	calc
	movq	$24, -8(%rbp)
	jmp	.L177
.L144:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	12(%rax), %edx
	movl	time(%rip), %eax
	cmpl	%eax, %edx
	jg	.L207
	movq	$21, -8(%rbp)
	jmp	.L177
.L207:
	movq	$23, -8(%rbp)
	jmp	.L177
.L150:
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cmpl	%eax, -36(%rbp)
	jle	.L209
	movq	$22, -8(%rbp)
	jmp	.L177
.L209:
	movq	$32, -8(%rbp)
	jmp	.L177
.L172:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jle	.L211
	movq	$35, -8(%rbp)
	jmp	.L177
.L211:
	movq	$11, -8(%rbp)
	jmp	.L177
.L158:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	time(%rip), %ecx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	12(%rax), %edx
	movl	%ecx, %eax
	subl	%edx, %eax
	movl	%eax, time(%rip)
	movl	l(%rip), %eax
	subl	$2, %eax
	movl	%eax, l(%rip)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%eax, %edx
	movl	-36(%rbp), %ecx
	movl	-28(%rbp), %esi
	movq	-24(%rbp), %rax
	movl	%ecx, %r8d
	movl	$2, %ecx
	movq	%rax, %rdi
	call	calc
	movq	$30, -8(%rbp)
	jmp	.L177
.L214:
	nop
.L177:
	jmp	.L213
.L215:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	calculation, .-calculation
	.globl	calc
	.type	calc, @function
calc:
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
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movl	%ecx, -36(%rbp)
	movl	%r8d, -40(%rbp)
	movq	$0, -8(%rbp)
.L303:
	cmpq	$43, -8(%rbp)
	ja	.L304
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L219(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L219(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L219:
	.long	.L250-.L219
	.long	.L249-.L219
	.long	.L248-.L219
	.long	.L304-.L219
	.long	.L247-.L219
	.long	.L246-.L219
	.long	.L304-.L219
	.long	.L304-.L219
	.long	.L245-.L219
	.long	.L304-.L219
	.long	.L244-.L219
	.long	.L243-.L219
	.long	.L242-.L219
	.long	.L241-.L219
	.long	.L240-.L219
	.long	.L304-.L219
	.long	.L304-.L219
	.long	.L304-.L219
	.long	.L239-.L219
	.long	.L238-.L219
	.long	.L237-.L219
	.long	.L236-.L219
	.long	.L304-.L219
	.long	.L235-.L219
	.long	.L234-.L219
	.long	.L304-.L219
	.long	.L233-.L219
	.long	.L232-.L219
	.long	.L231-.L219
	.long	.L230-.L219
	.long	.L229-.L219
	.long	.L228-.L219
	.long	.L227-.L219
	.long	.L226-.L219
	.long	.L304-.L219
	.long	.L225-.L219
	.long	.L224-.L219
	.long	.L304-.L219
	.long	.L305-.L219
	.long	.L304-.L219
	.long	.L222-.L219
	.long	.L221-.L219
	.long	.L220-.L219
	.long	.L218-.L219
	.text
.L239:
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L251
	movq	$27, -8(%rbp)
	jmp	.L253
.L251:
	movq	$11, -8(%rbp)
	jmp	.L253
.L247:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L254
	movq	$36, -8(%rbp)
	jmp	.L253
.L254:
	movq	$24, -8(%rbp)
	jmp	.L253
.L229:
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L256
	movq	$2, -8(%rbp)
	jmp	.L253
.L256:
	movq	$8, -8(%rbp)
	jmp	.L253
.L240:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	8(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L258
	movq	$35, -8(%rbp)
	jmp	.L253
.L258:
	movq	$38, -8(%rbp)
	jmp	.L253
.L228:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jle	.L260
	movq	$42, -8(%rbp)
	jmp	.L253
.L260:
	movq	$10, -8(%rbp)
	jmp	.L253
.L242:
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	l(%rip), %eax
	subl	$1, %eax
	movl	%eax, l(%rip)
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	leal	(%rdx,%rax), %esi
	movl	-40(%rbp), %ecx
	movl	-32(%rbp), %edx
	movq	-24(%rbp), %rax
	movl	%ecx, %r8d
	movl	$1, %ecx
	movq	%rax, %rdi
	call	calc
	movq	$10, -8(%rbp)
	jmp	.L253
.L245:
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L262
	movq	$21, -8(%rbp)
	jmp	.L253
.L262:
	movq	$10, -8(%rbp)
	jmp	.L253
.L249:
	cmpl	$3, -36(%rbp)
	jne	.L264
	movq	$32, -8(%rbp)
	jmp	.L253
.L264:
	movq	$26, -8(%rbp)
	jmp	.L253
.L235:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jg	.L266
	movq	$38, -8(%rbp)
	jmp	.L253
.L266:
	movq	$5, -8(%rbp)
	jmp	.L253
.L234:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L268
	movq	$28, -8(%rbp)
	jmp	.L253
.L268:
	movq	$10, -8(%rbp)
	jmp	.L253
.L236:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jle	.L270
	movq	$12, -8(%rbp)
	jmp	.L253
.L270:
	movq	$10, -8(%rbp)
	jmp	.L253
.L224:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jg	.L272
	movq	$38, -8(%rbp)
	jmp	.L253
.L272:
	movq	$24, -8(%rbp)
	jmp	.L253
.L233:
	cmpl	$4, -36(%rbp)
	jne	.L274
	movq	$43, -8(%rbp)
	jmp	.L253
.L274:
	movq	$10, -8(%rbp)
	jmp	.L253
.L243:
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L276
	movq	$41, -8(%rbp)
	jmp	.L253
.L276:
	movq	$10, -8(%rbp)
	jmp	.L253
.L241:
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cmpl	%eax, -40(%rbp)
	jle	.L278
	movq	$30, -8(%rbp)
	jmp	.L253
.L278:
	movq	$29, -8(%rbp)
	jmp	.L253
.L238:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	l(%rip), %eax
	subl	$1, %eax
	movl	%eax, l(%rip)
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%eax, %edx
	movl	-40(%rbp), %ecx
	movl	-28(%rbp), %esi
	movq	-24(%rbp), %rax
	movl	%ecx, %r8d
	movl	$2, %ecx
	movq	%rax, %rdi
	call	calc
	movq	$10, -8(%rbp)
	jmp	.L253
.L227:
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	testl	%eax, %eax
	jle	.L280
	movq	$18, -8(%rbp)
	jmp	.L253
.L280:
	movq	$26, -8(%rbp)
	jmp	.L253
.L222:
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cmpl	%eax, -40(%rbp)
	jle	.L282
	movq	$4, -8(%rbp)
	jmp	.L253
.L282:
	movq	$1, -8(%rbp)
	jmp	.L253
.L232:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jg	.L284
	movq	$38, -8(%rbp)
	jmp	.L253
.L284:
	movq	$11, -8(%rbp)
	jmp	.L253
.L231:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jle	.L287
	movq	$19, -8(%rbp)
	jmp	.L253
.L287:
	movq	$10, -8(%rbp)
	jmp	.L253
.L246:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L289
	movq	$31, -8(%rbp)
	jmp	.L253
.L289:
	movq	$10, -8(%rbp)
	jmp	.L253
.L226:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	24(%rax), %eax
	cmpl	$1, %eax
	jne	.L291
	movq	$23, -8(%rbp)
	jmp	.L253
.L291:
	movq	$5, -8(%rbp)
	jmp	.L253
.L221:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jle	.L293
	movq	$20, -8(%rbp)
	jmp	.L253
.L293:
	movq	$10, -8(%rbp)
	jmp	.L253
.L244:
	addl	$1, -12(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L253
.L220:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	l(%rip), %eax
	subl	$1, %eax
	movl	%eax, l(%rip)
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	movl	%eax, %edx
	movl	-40(%rbp), %ecx
	movl	-28(%rbp), %esi
	movq	-24(%rbp), %rax
	movl	%ecx, %r8d
	movl	$4, %ecx
	movq	%rax, %rdi
	call	calc
	movq	$10, -8(%rbp)
	jmp	.L253
.L250:
	movl	$1, -12(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L253
.L225:
	cmpl	$1, -36(%rbp)
	jne	.L295
	movq	$13, -8(%rbp)
	jmp	.L253
.L295:
	movq	$29, -8(%rbp)
	jmp	.L253
.L230:
	cmpl	$2, -36(%rbp)
	jne	.L297
	movq	$40, -8(%rbp)
	jmp	.L253
.L297:
	movq	$1, -8(%rbp)
	jmp	.L253
.L218:
	movl	-32(%rbp), %eax
	subl	-12(%rbp), %eax
	testl	%eax, %eax
	jle	.L299
	movq	$33, -8(%rbp)
	jmp	.L253
.L299:
	movq	$10, -8(%rbp)
	jmp	.L253
.L248:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	32(%rax), %ecx
	movl	-28(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rsi
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rsi, %rax
	movl	32(%rax), %eax
	cmpl	%eax, %ecx
	jg	.L301
	movq	$38, -8(%rbp)
	jmp	.L253
.L301:
	movq	$8, -8(%rbp)
	jmp	.L253
.L237:
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 24(%rax)
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rcx
	movl	-32(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rcx, %rax
	movl	$0, 36(%rax)
	movl	l(%rip), %eax
	subl	$1, %eax
	movl	%eax, l(%rip)
	movl	-28(%rbp), %eax
	subl	-12(%rbp), %eax
	movl	%eax, %esi
	movl	-40(%rbp), %ecx
	movl	-32(%rbp), %edx
	movq	-24(%rbp), %rax
	movl	%ecx, %r8d
	movl	$3, %ecx
	movq	%rax, %rdi
	call	calc
	movq	$10, -8(%rbp)
	jmp	.L253
.L304:
	nop
.L253:
	jmp	.L303
.L305:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	calc, .-calc
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
