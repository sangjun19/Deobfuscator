	.file	"Diegoav87_resaltador_code3_flatten.c"
	.text
	.globl	_TIG_IZ_4zD3_argc
	.bss
	.align 4
	.type	_TIG_IZ_4zD3_argc, @object
	.size	_TIG_IZ_4zD3_argc, 4
_TIG_IZ_4zD3_argc:
	.zero	4
	.globl	_TIG_IZ_4zD3_envp
	.align 8
	.type	_TIG_IZ_4zD3_envp, @object
	.size	_TIG_IZ_4zD3_envp, 8
_TIG_IZ_4zD3_envp:
	.zero	8
	.globl	_TIG_IZ_4zD3_argv
	.align 8
	.type	_TIG_IZ_4zD3_argv, @object
	.size	_TIG_IZ_4zD3_argv, 8
_TIG_IZ_4zD3_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%4d"
	.text
	.globl	printMultiplicationTable
	.type	printMultiplicationTable, @function
printMultiplicationTable:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$3, -8(%rbp)
.L17:
	cmpq	$12, -8(%rbp)
	ja	.L18
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
	.long	.L10-.L4
	.long	.L18-.L4
	.long	.L18-.L4
	.long	.L9-.L4
	.long	.L18-.L4
	.long	.L8-.L4
	.long	.L18-.L4
	.long	.L18-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L18-.L4
	.long	.L19-.L4
	.long	.L3-.L4
	.text
.L3:
	movl	-12(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jg	.L11
	movq	$5, -8(%rbp)
	jmp	.L13
.L11:
	movq	$9, -8(%rbp)
	jmp	.L13
.L7:
	movl	$1, -12(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L13
.L9:
	movl	$1, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L13
.L6:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L13
.L8:
	movl	-16(%rbp), %eax
	imull	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$12, -8(%rbp)
	jmp	.L13
.L10:
	movl	-16(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jg	.L15
	movq	$8, -8(%rbp)
	jmp	.L13
.L15:
	movq	$11, -8(%rbp)
	jmp	.L13
.L18:
	nop
.L13:
	jmp	.L17
.L19:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	printMultiplicationTable, .-printMultiplicationTable
	.section	.rodata
.LC1:
	.string	"%d, "
.LC2:
	.string	"Prime numbers up to %d: "
	.text
	.globl	printPrimes
	.type	printPrimes, @function
printPrimes:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$5, -8(%rbp)
.L38:
	cmpq	$10, -8(%rbp)
	ja	.L39
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L23(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L23(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L23:
	.long	.L39-.L23
	.long	.L40-.L23
	.long	.L39-.L23
	.long	.L30-.L23
	.long	.L29-.L23
	.long	.L28-.L23
	.long	.L27-.L23
	.long	.L26-.L23
	.long	.L25-.L23
	.long	.L24-.L23
	.long	.L22-.L23
	.text
.L29:
	cmpl	$0, -12(%rbp)
	je	.L32
	movq	$8, -8(%rbp)
	jmp	.L34
.L32:
	movq	$6, -8(%rbp)
	jmp	.L34
.L25:
	movl	-16(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L34
.L30:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$2, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L34
.L24:
	movl	$10, %edi
	call	putchar@PLT
	movq	$1, -8(%rbp)
	jmp	.L34
.L27:
	addl	$1, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L34
.L28:
	movq	$3, -8(%rbp)
	jmp	.L34
.L22:
	movl	-16(%rbp), %eax
	movl	%eax, %edi
	call	isPrime
	movl	%eax, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L34
.L26:
	movl	-16(%rbp), %eax
	cmpl	-20(%rbp), %eax
	jg	.L36
	movq	$10, -8(%rbp)
	jmp	.L34
.L36:
	movq	$9, -8(%rbp)
	jmp	.L34
.L39:
	nop
.L34:
	jmp	.L38
.L40:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	printPrimes, .-printPrimes
	.section	.rodata
.LC3:
	.string	"Fibonacci Series: %d, %d, "
	.text
	.globl	printFibonacci
	.type	printFibonacci, @function
printFibonacci:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$0, -8(%rbp)
.L54:
	cmpq	$7, -8(%rbp)
	ja	.L55
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L44(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L44(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L44:
	.long	.L49-.L44
	.long	.L55-.L44
	.long	.L48-.L44
	.long	.L47-.L44
	.long	.L56-.L44
	.long	.L45-.L44
	.long	.L55-.L44
	.long	.L43-.L44
	.text
.L47:
	movl	$0, -24(%rbp)
	movl	$1, -20(%rbp)
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$3, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L51
.L45:
	movl	$10, %edi
	call	putchar@PLT
	movq	$4, -8(%rbp)
	jmp	.L51
.L49:
	movq	$3, -8(%rbp)
	jmp	.L51
.L43:
	movl	-16(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jg	.L52
	movq	$2, -8(%rbp)
	jmp	.L51
.L52:
	movq	$5, -8(%rbp)
	jmp	.L51
.L48:
	movl	-24(%rbp), %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-20(%rbp), %eax
	movl	%eax, -24(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -20(%rbp)
	addl	$1, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L51
.L55:
	nop
.L51:
	jmp	.L54
.L56:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	printFibonacci, .-printFibonacci
	.globl	isPrime
	.type	isPrime, @function
isPrime:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movq	$7, -8(%rbp)
.L76:
	cmpq	$9, -8(%rbp)
	ja	.L77
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L60(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L60(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L60:
	.long	.L67-.L60
	.long	.L77-.L60
	.long	.L66-.L60
	.long	.L65-.L60
	.long	.L77-.L60
	.long	.L64-.L60
	.long	.L63-.L60
	.long	.L62-.L60
	.long	.L61-.L60
	.long	.L59-.L60
	.text
.L61:
	movl	$0, %eax
	jmp	.L68
.L65:
	addl	$1, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L69
.L59:
	movl	$1, %eax
	jmp	.L68
.L63:
	movl	-20(%rbp), %eax
	cltd
	idivl	-12(%rbp)
	movl	%edx, %eax
	testl	%eax, %eax
	jne	.L70
	movq	$0, -8(%rbp)
	jmp	.L69
.L70:
	movq	$3, -8(%rbp)
	jmp	.L69
.L64:
	movl	-20(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	cmpl	%eax, -12(%rbp)
	jg	.L72
	movq	$6, -8(%rbp)
	jmp	.L69
.L72:
	movq	$9, -8(%rbp)
	jmp	.L69
.L67:
	movl	$0, %eax
	jmp	.L68
.L62:
	cmpl	$1, -20(%rbp)
	jg	.L74
	movq	$8, -8(%rbp)
	jmp	.L69
.L74:
	movq	$2, -8(%rbp)
	jmp	.L69
.L66:
	movl	$2, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L69
.L77:
	nop
.L69:
	jmp	.L76
.L68:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	isPrime, .-isPrime
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_4zD3_envp(%rip)
	nop
.L79:
	movq	$0, _TIG_IZ_4zD3_argv(%rip)
	nop
.L80:
	movl	$0, _TIG_IZ_4zD3_argc(%rip)
	nop
	nop
.L81:
.L82:
#APP
# 97 "Diegoav87_resaltador_code3.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-4zD3--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_4zD3_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_4zD3_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_4zD3_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L88:
	cmpq	$2, -8(%rbp)
	je	.L83
	cmpq	$2, -8(%rbp)
	ja	.L90
	cmpq	$0, -8(%rbp)
	je	.L85
	cmpq	$1, -8(%rbp)
	jne	.L90
	movq	$2, -8(%rbp)
	jmp	.L86
.L85:
	movl	$0, %eax
	jmp	.L89
.L83:
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movl	$10, %edi
	call	printMultiplicationTable
	movl	$15, %edi
	call	printFibonacci
	movl	$50, %edi
	call	printPrimes
	movq	$0, -8(%rbp)
	jmp	.L86
.L90:
	nop
.L86:
	jmp	.L88
.L89:
	leave
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
