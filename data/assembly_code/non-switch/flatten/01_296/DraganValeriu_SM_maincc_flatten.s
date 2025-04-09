	.file	"DraganValeriu_SM_maincc_flatten.c"
	.text
	.globl	_TIG_IZ_MA9Y_envp
	.bss
	.align 8
	.type	_TIG_IZ_MA9Y_envp, @object
	.size	_TIG_IZ_MA9Y_envp, 8
_TIG_IZ_MA9Y_envp:
	.zero	8
	.globl	n
	.align 4
	.type	n, @object
	.size	n, 4
n:
	.zero	4
	.globl	_TIG_IZ_MA9Y_argc
	.align 4
	.type	_TIG_IZ_MA9Y_argc, @object
	.size	_TIG_IZ_MA9Y_argc, 4
_TIG_IZ_MA9Y_argc:
	.zero	4
	.globl	_TIG_IZ_MA9Y_argv
	.align 8
	.type	_TIG_IZ_MA9Y_argv, @object
	.size	_TIG_IZ_MA9Y_argv, 8
_TIG_IZ_MA9Y_argv:
	.zero	8
	.text
	.globl	interclasare
	.type	interclasare, @function
interclasare:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movl	%esi, -60(%rbp)
	movq	%rdx, -72(%rbp)
	movl	%ecx, -64(%rbp)
	movq	$16, -16(%rbp)
.L29:
	cmpq	$23, -16(%rbp)
	ja	.L31
	movq	-16(%rbp), %rax
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
	.long	.L16-.L4
	.long	.L31-.L4
	.long	.L15-.L4
	.long	.L31-.L4
	.long	.L31-.L4
	.long	.L31-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L31-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L31-.L4
	.long	.L31-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L31-.L4
	.long	.L8-.L4
	.long	.L31-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L31-.L4
	.long	.L31-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L7:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movl	-32(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-24(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -36(%rbp)
	addl	$1, -32(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L17
.L9:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-72(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jge	.L18
	movq	$10, -16(%rbp)
	jmp	.L17
.L18:
	movq	$7, -16(%rbp)
	jmp	.L17
.L3:
	movl	-40(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jge	.L20
	movq	$9, -16(%rbp)
	jmp	.L17
.L20:
	movq	$13, -16(%rbp)
	jmp	.L17
.L8:
	movq	$19, -16(%rbp)
	jmp	.L17
.L12:
	movl	-36(%rbp), %eax
	cmpl	-64(%rbp), %eax
	jge	.L22
	movq	$14, -16(%rbp)
	jmp	.L17
.L22:
	movq	$13, -16(%rbp)
	jmp	.L17
.L10:
	movl	-40(%rbp), %eax
	cmpl	-60(%rbp), %eax
	jge	.L24
	movq	$2, -16(%rbp)
	jmp	.L17
.L24:
	movq	$6, -16(%rbp)
	jmp	.L17
.L6:
	movl	-60(%rbp), %edx
	movl	-64(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movl	$0, -40(%rbp)
	movl	$0, -36(%rbp)
	movl	$0, -32(%rbp)
	movq	$23, -16(%rbp)
	jmp	.L17
.L14:
	movl	-36(%rbp), %eax
	cmpl	-64(%rbp), %eax
	jge	.L26
	movq	$18, -16(%rbp)
	jmp	.L17
.L26:
	movq	$22, -16(%rbp)
	jmp	.L17
.L5:
	movq	-24(%rbp), %rax
	jmp	.L30
.L11:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	-32(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-24(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -40(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L17
.L16:
	addl	$1, -32(%rbp)
	movq	$23, -16(%rbp)
	jmp	.L17
.L13:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movl	-32(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-24(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -36(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L17
.L15:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	-32(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-24(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -40(%rbp)
	addl	$1, -32(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L17
.L31:
	nop
.L17:
	jmp	.L29
.L30:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	interclasare, .-interclasare
	.section	.rodata
.LC0:
	.string	"%d "
	.text
	.globl	printArray
	.type	printArray, @function
printArray:
.LFB4:
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
	movq	$4, -8(%rbp)
.L44:
	cmpq	$7, -8(%rbp)
	ja	.L45
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L35(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L35(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L35:
	.long	.L45-.L35
	.long	.L39-.L35
	.long	.L45-.L35
	.long	.L38-.L35
	.long	.L37-.L35
	.long	.L45-.L35
	.long	.L36-.L35
	.long	.L46-.L35
	.text
.L37:
	movl	$0, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L40
.L39:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L41
	movq	$6, -8(%rbp)
	jmp	.L40
.L41:
	movq	$3, -8(%rbp)
	jmp	.L40
.L38:
	movl	$10, %edi
	call	putchar@PLT
	movq	$7, -8(%rbp)
	jmp	.L40
.L36:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L40
.L45:
	nop
.L40:
	jmp	.L44
.L46:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	printArray, .-printArray
	.section	.rodata
.LC1:
	.string	" #"
.LC2:
	.string	"%d"
.LC3:
	.string	" -"
.LC4:
	.string	"  "
	.text
	.globl	print_hist
	.type	print_hist, @function
print_hist:
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
	movq	$26, -8(%rbp)
.L96:
	cmpq	$40, -8(%rbp)
	ja	.L97
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L50(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L50(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L50:
	.long	.L75-.L50
	.long	.L98-.L50
	.long	.L97-.L50
	.long	.L73-.L50
	.long	.L97-.L50
	.long	.L72-.L50
	.long	.L71-.L50
	.long	.L97-.L50
	.long	.L70-.L50
	.long	.L69-.L50
	.long	.L68-.L50
	.long	.L67-.L50
	.long	.L66-.L50
	.long	.L97-.L50
	.long	.L97-.L50
	.long	.L65-.L50
	.long	.L64-.L50
	.long	.L97-.L50
	.long	.L97-.L50
	.long	.L63-.L50
	.long	.L97-.L50
	.long	.L62-.L50
	.long	.L97-.L50
	.long	.L97-.L50
	.long	.L61-.L50
	.long	.L60-.L50
	.long	.L59-.L50
	.long	.L58-.L50
	.long	.L57-.L50
	.long	.L97-.L50
	.long	.L56-.L50
	.long	.L55-.L50
	.long	.L97-.L50
	.long	.L54-.L50
	.long	.L53-.L50
	.long	.L52-.L50
	.long	.L97-.L50
	.long	.L51-.L50
	.long	.L97-.L50
	.long	.L97-.L50
	.long	.L49-.L50
	.text
.L60:
	movl	$43, %edi
	call	putchar@PLT
	movl	$0, -16(%rbp)
	movq	$31, -8(%rbp)
	jmp	.L76
.L56:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -8(%rbp)
	jmp	.L76
.L65:
	movl	$0, -12(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L76
.L55:
	movl	-16(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L77
	movq	$28, -8(%rbp)
	jmp	.L76
.L77:
	movq	$1, -8(%rbp)
	jmp	.L76
.L66:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	$9, %eax
	subl	-20(%rbp), %eax
	cmpl	%eax, %edx
	jl	.L79
	movq	$30, -8(%rbp)
	jmp	.L76
.L79:
	movq	$35, -8(%rbp)
	jmp	.L76
.L70:
	addl	$1, -16(%rbp)
	movq	$40, -8(%rbp)
	jmp	.L76
.L73:
	movl	$0, -20(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L76
.L64:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movslq	%eax, %rdx
	imulq	$1717986919, %rdx, %rdx
	shrq	$32, %rdx
	sarl	$2, %edx
	sarl	$31, %eax
	movl	%edx, %ecx
	subl	%eax, %ecx
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	-12(%rbp), %edx
	addl	%ecx, %edx
	movl	%edx, (%rax)
	addl	$1, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L76
.L61:
	movl	$9, %eax
	subl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -16(%rbp)
	movq	$40, -8(%rbp)
	jmp	.L76
.L62:
	movl	$1, -12(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L76
.L59:
	movl	$0, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L76
.L67:
	movl	-16(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L82
	movq	$27, -8(%rbp)
	jmp	.L76
.L82:
	movq	$3, -8(%rbp)
	jmp	.L76
.L69:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$1, %eax
	jle	.L84
	movq	$12, -8(%rbp)
	jmp	.L76
.L84:
	movq	$0, -8(%rbp)
	jmp	.L76
.L63:
	cmpl	$9, -20(%rbp)
	jg	.L86
	movq	$24, -8(%rbp)
	jmp	.L76
.L86:
	movq	$25, -8(%rbp)
	jmp	.L76
.L49:
	cmpl	$15, -16(%rbp)
	jg	.L88
	movq	$9, -8(%rbp)
	jmp	.L76
.L88:
	movq	$34, -8(%rbp)
	jmp	.L76
.L71:
	cmpl	$15, -20(%rbp)
	jg	.L90
	movq	$33, -8(%rbp)
	jmp	.L76
.L90:
	movq	$37, -8(%rbp)
	jmp	.L76
.L58:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %ecx
	movslq	%ecx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	$2, %eax
	movl	%ecx, %esi
	sarl	$31, %esi
	subl	%esi, %eax
	movl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	subl	%eax, %ecx
	movl	%ecx, %edx
	testl	%edx, %edx
	jle	.L92
	movq	$5, -8(%rbp)
	jmp	.L76
.L92:
	movq	$10, -8(%rbp)
	jmp	.L76
.L53:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -20(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L76
.L57:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -16(%rbp)
	movq	$31, -8(%rbp)
	jmp	.L76
.L72:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$9, %eax
	jle	.L94
	movq	$21, -8(%rbp)
	jmp	.L76
.L94:
	movq	$15, -8(%rbp)
	jmp	.L76
.L54:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -20(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L76
.L51:
	movl	$10, %edi
	call	putchar@PLT
	movl	$0, -20(%rbp)
	movq	$19, -8(%rbp)
	jmp	.L76
.L68:
	movl	$0, -12(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L76
.L75:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -8(%rbp)
	jmp	.L76
.L52:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -8(%rbp)
	jmp	.L76
.L97:
	nop
.L76:
	jmp	.L96
.L98:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	print_hist, .-print_hist
	.globl	histogram
	.type	histogram, @function
histogram:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movl	%edx, -68(%rbp)
	movq	$1, -8(%rbp)
.L115:
	cmpq	$13, -8(%rbp)
	ja	.L116
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L102(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L102(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L102:
	.long	.L116-.L102
	.long	.L108-.L102
	.long	.L107-.L102
	.long	.L116-.L102
	.long	.L116-.L102
	.long	.L116-.L102
	.long	.L117-.L102
	.long	.L116-.L102
	.long	.L116-.L102
	.long	.L116-.L102
	.long	.L105-.L102
	.long	.L104-.L102
	.long	.L103-.L102
	.long	.L101-.L102
	.text
.L103:
	movl	n(%rip), %eax
	cmpl	%eax, -36(%rbp)
	jge	.L109
	movq	$11, -8(%rbp)
	jmp	.L111
.L109:
	movq	$10, -8(%rbp)
	jmp	.L111
.L108:
	movl	$0, -36(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L111
.L104:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	leal	15(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$4, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	addl	$1, %edx
	movl	%edx, (%rax)
	addl	$1, -36(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L111
.L101:
	cmpl	$15, -28(%rbp)
	jg	.L112
	movq	$2, -8(%rbp)
	jmp	.L111
.L112:
	movq	$6, -8(%rbp)
	jmp	.L111
.L105:
	call	rand@PLT
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	cltd
	shrl	$28, %edx
	addl	%edx, %eax
	andl	$15, %eax
	subl	%edx, %eax
	movslq	%eax, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rdx), %edx
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-56(%rbp), %rax
	addq	%rcx, %rax
	addl	$1000, %edx
	movl	%edx, (%rax)
	call	rand@PLT
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	cltd
	shrl	$28, %edx
	addl	%edx, %eax
	andl	$15, %eax
	subl	%edx, %eax
	movslq	%eax, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rdx), %edx
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-56(%rbp), %rax
	addq	%rcx, %rax
	addl	$100, %edx
	movl	%edx, (%rax)
	call	rand@PLT
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	cltd
	shrl	$28, %edx
	addl	%edx, %eax
	andl	$15, %eax
	subl	%edx, %eax
	movslq	%eax, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rdx), %edx
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-56(%rbp), %rax
	addq	%rcx, %rax
	addl	$100, %edx
	movl	%edx, (%rax)
	call	rand@PLT
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	cltd
	shrl	$28, %edx
	addl	%edx, %eax
	andl	$15, %eax
	subl	%edx, %eax
	movslq	%eax, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rdx), %edx
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-56(%rbp), %rax
	addq	%rcx, %rax
	addl	$100, %edx
	movl	%edx, (%rax)
	movl	$0, -32(%rbp)
	movl	$0, -28(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L111
.L107:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	imull	$1000, %eax, %eax
	movl	n(%rip), %edx
	leal	400(%rdx), %edi
	cltd
	idivl	%edi
	movl	%eax, %edx
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-56(%rbp), %rax
	addq	%rcx, %rax
	movslq	%edx, %rcx
	imulq	$1717986919, %rcx, %rcx
	shrq	$32, %rcx
	sarl	$2, %ecx
	movl	%edx, %esi
	sarl	$31, %esi
	subl	%esi, %ecx
	movl	%ecx, %edx
	movl	%edx, (%rax)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	addl	%eax, -32(%rbp)
	addl	$1, -28(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L111
.L116:
	nop
.L111:
	jmp	.L115
.L117:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	histogram, .-histogram
	.globl	main
	.type	main, @function
main:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r13
	pushq	%r12
	subq	$176, %rsp
	.cfi_offset 13, -24
	.cfi_offset 12, -32
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%rdx, -184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movl	$1061, n(%rip)
	nop
.L119:
	movq	$0, _TIG_IZ_MA9Y_envp(%rip)
	nop
.L120:
	movq	$0, _TIG_IZ_MA9Y_argv(%rip)
	nop
.L121:
	movl	$0, _TIG_IZ_MA9Y_argc(%rip)
	nop
	nop
.L122:
.L123:
#APP
# 139 "DraganValeriu_SM_maincc.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-MA9Y--0
# 0 "" 2
#NO_APP
	movl	-164(%rbp), %eax
	movl	%eax, _TIG_IZ_MA9Y_argc(%rip)
	movq	-176(%rbp), %rax
	movq	%rax, _TIG_IZ_MA9Y_argv(%rip)
	movq	-184(%rbp), %rax
	movq	%rax, _TIG_IZ_MA9Y_envp(%rip)
	nop
	movq	$20, -128(%rbp)
.L151:
	cmpq	$26, -128(%rbp)
	ja	.L154
	movq	-128(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L126(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L126(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L126:
	.long	.L154-.L126
	.long	.L139-.L126
	.long	.L138-.L126
	.long	.L137-.L126
	.long	.L136-.L126
	.long	.L154-.L126
	.long	.L154-.L126
	.long	.L154-.L126
	.long	.L154-.L126
	.long	.L135-.L126
	.long	.L134-.L126
	.long	.L133-.L126
	.long	.L154-.L126
	.long	.L132-.L126
	.long	.L131-.L126
	.long	.L130-.L126
	.long	.L154-.L126
	.long	.L154-.L126
	.long	.L129-.L126
	.long	.L154-.L126
	.long	.L128-.L126
	.long	.L154-.L126
	.long	.L154-.L126
	.long	.L127-.L126
	.long	.L154-.L126
	.long	.L154-.L126
	.long	.L125-.L126
	.text
.L129:
	movl	-144(%rbp), %eax
	cltq
	movl	-96(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -144(%rbp)
	movq	$11, -128(%rbp)
	jmp	.L140
.L136:
	cmpl	$15, -148(%rbp)
	jbe	.L141
	movq	$10, -128(%rbp)
	jmp	.L140
.L141:
	movq	$13, -128(%rbp)
	jmp	.L140
.L131:
	call	rand@PLT
	movl	%eax, -140(%rbp)
	movl	-152(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-136(%rbp), %rax
	leaq	(%rdx,%rax), %rsi
	movl	-140(%rbp), %edx
	movslq	%edx, %rax
	imulq	$-2139062143, %rax, %rax
	shrq	$32, %rax
	addl	%edx, %eax
	sarl	$7, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	sall	$8, %ecx
	subl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	movl	%eax, (%rsi)
	addl	$1, -152(%rbp)
	movq	$3, -128(%rbp)
	jmp	.L140
.L130:
	movl	$0, -96(%rbp)
	movl	$1, -148(%rbp)
	movq	$4, -128(%rbp)
	jmp	.L140
.L139:
	movl	$10, %edi
	call	putchar@PLT
	leaq	-96(%rbp), %rax
	movl	$16, %esi
	movq	%rax, %rdi
	call	print_hist
	movq	$26, -128(%rbp)
	jmp	.L140
.L127:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -120(%rbp)
	movq	-120(%rbp), %rax
	movl	%eax, %edi
	call	srand@PLT
	movq	$9, -128(%rbp)
	jmp	.L140
.L137:
	movl	n(%rip), %eax
	cmpl	%eax, -152(%rbp)
	jge	.L143
	movq	$14, -128(%rbp)
	jmp	.L140
.L143:
	movq	$15, -128(%rbp)
	jmp	.L140
.L125:
	movl	$0, %eax
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L152
	jmp	.L153
.L133:
	cmpl	$15, -144(%rbp)
	jg	.L146
	movq	$18, -128(%rbp)
	jmp	.L140
.L146:
	movq	$1, -128(%rbp)
	jmp	.L140
.L135:
	movl	n(%rip), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -104(%rbp)
	movq	-104(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %esi
	movl	$0, %edx
	divq	%rsi
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L148:
	cmpq	%rdx, %rsp
	je	.L149
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L148
.L149:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L150
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L150:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -136(%rbp)
	movq	$2, -128(%rbp)
	jmp	.L140
.L132:
	movl	-148(%rbp), %eax
	movl	$0, -96(%rbp,%rax,4)
	addl	$1, -148(%rbp)
	movq	$4, -128(%rbp)
	jmp	.L140
.L134:
	movl	n(%rip), %eax
	movslq	%eax, %rdx
	subq	$1, %rdx
	movq	%rdx, -112(%rbp)
	cltq
	movq	%rax, %r12
	movl	$0, %r13d
	movq	-136(%rbp), %rcx
	leaq	-96(%rbp), %rax
	movl	$16, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	histogram
	movl	$0, -144(%rbp)
	movq	$11, -128(%rbp)
	jmp	.L140
.L138:
	movl	$0, -152(%rbp)
	movq	$3, -128(%rbp)
	jmp	.L140
.L128:
	movq	$23, -128(%rbp)
	jmp	.L140
.L154:
	nop
.L140:
	jmp	.L151
.L153:
	call	__stack_chk_fail@PLT
.L152:
	leaq	-16(%rbp), %rsp
	popq	%r12
	popq	%r13
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	main, .-main
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
