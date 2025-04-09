	.file	"githubber-me_dsathirdsem_9_flatten.c"
	.text
	.globl	_TIG_IZ_39m4_argc
	.bss
	.align 4
	.type	_TIG_IZ_39m4_argc, @object
	.size	_TIG_IZ_39m4_argc, 4
_TIG_IZ_39m4_argc:
	.zero	4
	.globl	_TIG_IZ_39m4_argv
	.align 8
	.type	_TIG_IZ_39m4_argv, @object
	.size	_TIG_IZ_39m4_argv, 8
_TIG_IZ_39m4_argv:
	.zero	8
	.globl	_TIG_IZ_39m4_envp
	.align 8
	.type	_TIG_IZ_39m4_envp, @object
	.size	_TIG_IZ_39m4_envp, 8
_TIG_IZ_39m4_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"\n~~~~Polynomial evaluation P(x,y,z)~~~"
	.align 8
.LC1:
	.string	"\nRepresentation of Polynomial for evaluation: "
	.align 8
.LC2:
	.string	"\nResult of polynomial evaluation is : %d \n"
.LC3:
	.string	"\nEnter the POLY1(x,y,z):  "
.LC4:
	.string	"\nPolynomial 1 is:  "
.LC5:
	.string	"\nEnter the POLY2(x,y,z):  "
.LC6:
	.string	"\nPolynomial 2 is: "
.LC7:
	.string	"\nPolynomial addition result: "
.LC8:
	.string	"\n~~~Menu~~~"
	.align 8
.LC9:
	.string	"\n1.Represent and Evaluate a Polynomial P(x,y,z)"
	.align 8
.LC10:
	.string	"\n2.Find the sum of two polynomials POLY1(x,y,z)"
.LC11:
	.string	"\nEnter your choice:"
.LC12:
	.string	"%d"
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_39m4_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_39m4_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_39m4_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 127 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-39m4--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_39m4_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_39m4_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_39m4_envp(%rip)
	nop
	movq	$3, -16(%rbp)
.L22:
	cmpq	$13, -16(%rbp)
	ja	.L24
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L24-.L8
	.long	.L24-.L8
	.long	.L24-.L8
	.long	.L10-.L8
	.long	.L24-.L8
	.long	.L24-.L8
	.long	.L24-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	read_poly
	movq	%rax, -48(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	display
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	poly_evaluate
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L16
.L9:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	read_poly
	movq	%rax, -40(%rbp)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	display
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	read_poly
	movq	%rax, -32(%rbp)
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	display
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-24(%rbp), %rdx
	movq	-32(%rbp), %rcx
	movq	-40(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	poly_sum
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	display
	movq	$2, -16(%rbp)
	jmp	.L16
.L10:
	movl	-56(%rbp), %eax
	cmpl	$3, %eax
	je	.L17
	cmpl	$3, %eax
	jg	.L18
	cmpl	$1, %eax
	je	.L19
	cmpl	$2, %eax
	je	.L20
	jmp	.L18
.L17:
	movq	$13, -16(%rbp)
	jmp	.L21
.L20:
	movq	$12, -16(%rbp)
	jmp	.L21
.L19:
	movq	$4, -16(%rbp)
	jmp	.L21
.L18:
	movq	$0, -16(%rbp)
	nop
.L21:
	jmp	.L16
.L14:
	call	getnode
	movq	%rax, -48(%rbp)
	call	getnode
	movq	%rax, -40(%rbp)
	call	getnode
	movq	%rax, -32(%rbp)
	call	getnode
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-40(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-32(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-24(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	$2, -16(%rbp)
	jmp	.L16
.L12:
	movq	$1, -16(%rbp)
	jmp	.L16
.L7:
	movl	$0, %edi
	call	exit@PLT
.L15:
	movq	$2, -16(%rbp)
	jmp	.L16
.L13:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-56(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$8, -16(%rbp)
	jmp	.L16
.L24:
	nop
.L16:
	jmp	.L22
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
.LC13:
	.string	"Running out of memory "
	.text
	.globl	getnode
	.type	getnode, @function
getnode:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$3, -16(%rbp)
.L38:
	cmpq	$5, -16(%rbp)
	ja	.L39
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L28(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L28(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L28:
	.long	.L33-.L28
	.long	.L32-.L28
	.long	.L31-.L28
	.long	.L30-.L28
	.long	.L29-.L28
	.long	.L27-.L28
	.text
.L29:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L34
.L32:
	movl	$0, %eax
	jmp	.L35
.L30:
	movq	$5, -16(%rbp)
	jmp	.L34
.L27:
	movl	$24, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L34
.L33:
	cmpq	$0, -24(%rbp)
	jne	.L36
	movq	$4, -16(%rbp)
	jmp	.L34
.L36:
	movq	$2, -16(%rbp)
	jmp	.L34
.L31:
	movq	-24(%rbp), %rax
	jmp	.L35
.L39:
	nop
.L34:
	jmp	.L38
.L35:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	getnode, .-getnode
	.section	.rodata
	.align 8
.LC14:
	.string	"\nEnter the no of terms in the polynomial: "
.LC15:
	.string	"\n\tEnter the %d term: "
.LC16:
	.string	"\n\t\tCoef =  "
	.align 8
.LC17:
	.string	"\n\t\tEnter Pow(x) Pow(y) and Pow(z): "
	.text
	.globl	read_poly
	.type	read_poly, @function
read_poly:
.LFB4:
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
	movq	$4, -16(%rbp)
.L52:
	cmpq	$6, -16(%rbp)
	ja	.L55
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L43(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L43(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L43:
	.long	.L47-.L43
	.long	.L46-.L43
	.long	.L45-.L43
	.long	.L55-.L43
	.long	.L44-.L43
	.long	.L55-.L43
	.long	.L42-.L43
	.text
.L44:
	movq	$1, -16(%rbp)
	jmp	.L48
.L46:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$1, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L48
.L42:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-28(%rbp), %ecx
	movl	-32(%rbp), %edx
	movl	-36(%rbp), %esi
	movl	-40(%rbp), %eax
	movq	-56(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -56(%rbp)
	addl	$1, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L48
.L47:
	movq	-56(%rbp), %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L53
	jmp	.L54
.L45:
	movl	-24(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	jg	.L50
	movq	$6, -16(%rbp)
	jmp	.L48
.L50:
	movq	$0, -16(%rbp)
	jmp	.L48
.L55:
	nop
.L48:
	jmp	.L52
.L54:
	call	__stack_chk_fail@PLT
.L53:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	read_poly, .-read_poly
	.globl	attach
	.type	attach, @function
attach:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movl	%esi, -40(%rbp)
	movl	%edx, -44(%rbp)
	movl	%ecx, -48(%rbp)
	movq	%r8, -56(%rbp)
	movq	$5, -8(%rbp)
.L69:
	cmpq	$7, -8(%rbp)
	ja	.L71
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L59(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L59(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L59:
	.long	.L64-.L59
	.long	.L71-.L59
	.long	.L71-.L59
	.long	.L63-.L59
	.long	.L62-.L59
	.long	.L61-.L59
	.long	.L60-.L59
	.long	.L58-.L59
	.text
.L62:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L65
	movq	$3, -8(%rbp)
	jmp	.L67
.L65:
	movq	$6, -8(%rbp)
	jmp	.L67
.L63:
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L67
.L60:
	movq	-16(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	-24(%rbp), %rax
	movq	-56(%rbp), %rdx
	movq	%rdx, 16(%rax)
	movq	$0, -8(%rbp)
	jmp	.L67
.L61:
	movq	$7, -8(%rbp)
	jmp	.L67
.L64:
	movq	-56(%rbp), %rax
	jmp	.L70
.L58:
	call	getnode
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	movl	-40(%rbp), %edx
	movl	%edx, 4(%rax)
	movq	-24(%rbp), %rax
	movl	-44(%rbp), %edx
	movl	%edx, 8(%rax)
	movq	-24(%rbp), %rax
	movl	-48(%rbp), %edx
	movl	%edx, 12(%rax)
	movq	-56(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L67
.L71:
	nop
.L67:
	jmp	.L69
.L70:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	attach, .-attach
	.section	.rodata
.LC18:
	.string	"%dx^%dy^%dz^%d"
.LC19:
	.string	" + "
.LC20:
	.string	"\nPolynomial does not exist."
	.text
	.globl	display
	.type	display, @function
display:
.LFB8:
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
.L92:
	cmpq	$11, -8(%rbp)
	ja	.L93
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L75(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L75(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L75:
	.long	.L83-.L75
	.long	.L82-.L75
	.long	.L81-.L75
	.long	.L94-.L75
	.long	.L79-.L75
	.long	.L93-.L75
	.long	.L78-.L75
	.long	.L93-.L75
	.long	.L93-.L75
	.long	.L77-.L75
	.long	.L94-.L75
	.long	.L74-.L75
	.text
.L79:
	movq	-16(%rbp), %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L84
	movq	$9, -8(%rbp)
	jmp	.L86
.L84:
	movq	$3, -8(%rbp)
	jmp	.L86
.L82:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L86
.L74:
	movq	-16(%rbp), %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L88
	movq	$0, -8(%rbp)
	jmp	.L86
.L88:
	movq	$4, -8(%rbp)
	jmp	.L86
.L77:
	movq	-16(%rbp), %rax
	movl	12(%rax), %esi
	movq	-16(%rbp), %rax
	movl	8(%rax), %ecx
	movq	-16(%rbp), %rax
	movl	4(%rax), %edx
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%esi, %r8d
	movl	%eax, %esi
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L86
.L78:
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	cmpq	%rax, %rdx
	jne	.L90
	movq	$2, -8(%rbp)
	jmp	.L86
.L90:
	movq	$1, -8(%rbp)
	jmp	.L86
.L83:
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -8(%rbp)
	jmp	.L86
.L81:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L86
.L93:
	nop
.L86:
	jmp	.L92
.L94:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	display, .-display
	.globl	poly_sum
	.type	poly_sum, @function
poly_sum:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -72(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	$58, -8(%rbp)
.L224:
	cmpq	$102, -8(%rbp)
	ja	.L226
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L98(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L98(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L98:
	.long	.L158-.L98
	.long	.L157-.L98
	.long	.L156-.L98
	.long	.L155-.L98
	.long	.L226-.L98
	.long	.L154-.L98
	.long	.L153-.L98
	.long	.L152-.L98
	.long	.L151-.L98
	.long	.L150-.L98
	.long	.L149-.L98
	.long	.L226-.L98
	.long	.L148-.L98
	.long	.L226-.L98
	.long	.L147-.L98
	.long	.L226-.L98
	.long	.L146-.L98
	.long	.L226-.L98
	.long	.L145-.L98
	.long	.L144-.L98
	.long	.L143-.L98
	.long	.L142-.L98
	.long	.L226-.L98
	.long	.L141-.L98
	.long	.L140-.L98
	.long	.L139-.L98
	.long	.L138-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L137-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L136-.L98
	.long	.L226-.L98
	.long	.L135-.L98
	.long	.L226-.L98
	.long	.L134-.L98
	.long	.L133-.L98
	.long	.L226-.L98
	.long	.L132-.L98
	.long	.L131-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L130-.L98
	.long	.L226-.L98
	.long	.L129-.L98
	.long	.L128-.L98
	.long	.L127-.L98
	.long	.L226-.L98
	.long	.L126-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L125-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L124-.L98
	.long	.L226-.L98
	.long	.L123-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L122-.L98
	.long	.L226-.L98
	.long	.L121-.L98
	.long	.L226-.L98
	.long	.L120-.L98
	.long	.L119-.L98
	.long	.L118-.L98
	.long	.L226-.L98
	.long	.L117-.L98
	.long	.L226-.L98
	.long	.L226-.L98
	.long	.L116-.L98
	.long	.L115-.L98
	.long	.L226-.L98
	.long	.L114-.L98
	.long	.L226-.L98
	.long	.L113-.L98
	.long	.L112-.L98
	.long	.L111-.L98
	.long	.L110-.L98
	.long	.L109-.L98
	.long	.L108-.L98
	.long	.L107-.L98
	.long	.L106-.L98
	.long	.L105-.L98
	.long	.L226-.L98
	.long	.L104-.L98
	.long	.L103-.L98
	.long	.L226-.L98
	.long	.L102-.L98
	.long	.L226-.L98
	.long	.L101-.L98
	.long	.L100-.L98
	.long	.L99-.L98
	.long	.L97-.L98
	.text
.L145:
	movq	-16(%rbp), %rdx
	movq	-80(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L159
	movq	$91, -8(%rbp)
	jmp	.L161
.L159:
	movq	$20, -8(%rbp)
	jmp	.L161
.L127:
	cmpl	$1, -48(%rbp)
	je	.L162
	cmpl	$1, -48(%rbp)
	jg	.L163
	cmpl	$-1, -48(%rbp)
	je	.L164
	cmpl	$0, -48(%rbp)
	je	.L165
	jmp	.L163
.L162:
	movq	$89, -8(%rbp)
	jmp	.L166
.L165:
	movq	$68, -8(%rbp)
	jmp	.L166
.L164:
	movq	$12, -8(%rbp)
	jmp	.L166
.L163:
	movq	$88, -8(%rbp)
	nop
.L166:
	jmp	.L161
.L115:
	movl	-36(%rbp), %eax
	movl	%eax, -32(%rbp)
	movq	$35, -8(%rbp)
	jmp	.L161
.L139:
	movq	-16(%rbp), %rdx
	movq	-80(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L167
	movq	$101, -8(%rbp)
	jmp	.L161
.L167:
	movq	$92, -8(%rbp)
	jmp	.L161
.L128:
	movq	-16(%rbp), %rax
	movl	4(%rax), %eax
	testl	%eax, %eax
	je	.L169
	movq	$86, -8(%rbp)
	jmp	.L161
.L169:
	movq	$19, -8(%rbp)
	jmp	.L161
.L126:
	movq	-24(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-24(%rbp), %rax
	movl	4(%rax), %esi
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L137:
	movl	$-1, -52(%rbp)
	movq	$102, -8(%rbp)
	jmp	.L161
.L97:
	movl	-52(%rbp), %eax
	movl	%eax, -48(%rbp)
	movq	$50, -8(%rbp)
	jmp	.L161
.L147:
	movq	-16(%rbp), %rax
	movl	12(%rax), %eax
	testl	%eax, %eax
	je	.L171
	movq	$40, -8(%rbp)
	jmp	.L161
.L171:
	movq	$101, -8(%rbp)
	jmp	.L161
.L114:
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	8(%rax), %eax
	cmpl	%eax, %edx
	jle	.L173
	movq	$76, -8(%rbp)
	jmp	.L161
.L173:
	movq	$48, -8(%rbp)
	jmp	.L161
.L108:
	movq	-24(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-24(%rbp), %rax
	movl	4(%rax), %esi
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L116:
	movl	$0, -40(%rbp)
	movq	$95, -8(%rbp)
	jmp	.L161
.L148:
	movq	-16(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-16(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	4(%rax), %esi
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L99:
	movq	-24(%rbp), %rax
	movl	4(%rax), %edx
	movq	-16(%rbp), %rax
	movl	4(%rax), %eax
	cmpl	%eax, %edx
	jne	.L175
	movq	$61, -8(%rbp)
	jmp	.L161
.L175:
	movq	$38, -8(%rbp)
	jmp	.L161
.L151:
	movq	$84, -8(%rbp)
	jmp	.L161
.L157:
	movq	-16(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-16(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	4(%rax), %esi
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L141:
	movq	-24(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-24(%rbp), %rax
	movl	4(%rax), %esi
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L121:
	movl	$-1, -36(%rbp)
	movq	$80, -8(%rbp)
	jmp	.L161
.L155:
	movq	-24(%rbp), %rax
	movl	12(%rax), %edx
	movq	-16(%rbp), %rax
	movl	12(%rax), %eax
	cmpl	%eax, %edx
	jle	.L177
	movq	$37, -8(%rbp)
	jmp	.L161
.L177:
	movq	$70, -8(%rbp)
	jmp	.L161
.L146:
	movl	$1, -52(%rbp)
	movq	$102, -8(%rbp)
	jmp	.L161
.L140:
	movq	-24(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-24(%rbp), %rax
	movl	4(%rax), %esi
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L142:
	movq	-24(%rbp), %rax
	movl	(%rax), %edx
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	addl	%edx, %eax
	movl	%eax, -28(%rbp)
	movq	-24(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-24(%rbp), %rax
	movl	4(%rax), %esi
	movq	-88(%rbp), %rdi
	movl	-28(%rbp), %eax
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L104:
	movq	-24(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-24(%rbp), %rax
	movl	4(%rax), %esi
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$92, -8(%rbp)
	jmp	.L161
.L117:
	movl	$1, -44(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L161
.L122:
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	8(%rax), %eax
	cmpl	%eax, %edx
	jle	.L179
	movq	$24, -8(%rbp)
	jmp	.L161
.L179:
	movq	$87, -8(%rbp)
	jmp	.L161
.L112:
	movq	-24(%rbp), %rax
	movl	12(%rax), %edx
	movq	-16(%rbp), %rax
	movl	12(%rax), %eax
	cmpl	%eax, %edx
	jne	.L181
	movq	$21, -8(%rbp)
	jmp	.L161
.L181:
	movq	$38, -8(%rbp)
	jmp	.L161
.L100:
	movq	-24(%rbp), %rax
	movl	4(%rax), %edx
	movq	-16(%rbp), %rax
	movl	4(%rax), %eax
	cmpl	%eax, %edx
	jle	.L183
	movq	$16, -8(%rbp)
	jmp	.L161
.L183:
	movq	$30, -8(%rbp)
	jmp	.L161
.L138:
	movq	-24(%rbp), %rax
	movl	12(%rax), %edx
	movq	-16(%rbp), %rax
	movl	12(%rax), %eax
	cmpl	%eax, %edx
	jle	.L185
	movq	$6, -8(%rbp)
	jmp	.L161
.L185:
	movq	$46, -8(%rbp)
	jmp	.L161
.L150:
	movq	-16(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-16(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	4(%rax), %esi
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L123:
	movq	-16(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-16(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	4(%rax), %esi
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L144:
	movq	-24(%rbp), %rax
	movl	8(%rax), %eax
	testl	%eax, %eax
	je	.L187
	movq	$99, -8(%rbp)
	jmp	.L161
.L187:
	movq	$33, -8(%rbp)
	jmp	.L161
.L107:
	movq	$84, -8(%rbp)
	jmp	.L161
.L132:
	movq	-24(%rbp), %rax
	movl	12(%rax), %edx
	movq	-16(%rbp), %rax
	movl	12(%rax), %eax
	cmpl	%eax, %edx
	jne	.L189
	movq	$74, -8(%rbp)
	jmp	.L161
.L189:
	movq	$3, -8(%rbp)
	jmp	.L161
.L153:
	movq	-24(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-24(%rbp), %rax
	movl	4(%rax), %esi
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L133:
	movq	-24(%rbp), %rax
	movl	4(%rax), %eax
	testl	%eax, %eax
	je	.L191
	movq	$86, -8(%rbp)
	jmp	.L161
.L191:
	movq	$49, -8(%rbp)
	jmp	.L161
.L124:
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	8(%rax), %eax
	cmpl	%eax, %edx
	jne	.L193
	movq	$85, -8(%rbp)
	jmp	.L161
.L193:
	movq	$38, -8(%rbp)
	jmp	.L161
.L110:
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	8(%rax), %eax
	cmpl	%eax, %edx
	jge	.L195
	movq	$1, -8(%rbp)
	jmp	.L161
.L195:
	movq	$73, -8(%rbp)
	jmp	.L161
.L125:
	movq	$97, -8(%rbp)
	jmp	.L161
.L113:
	movq	-24(%rbp), %rdx
	movq	-72(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L197
	movq	$25, -8(%rbp)
	jmp	.L161
.L197:
	movq	$92, -8(%rbp)
	jmp	.L161
.L118:
	movl	$0, -32(%rbp)
	movq	$35, -8(%rbp)
	jmp	.L161
.L129:
	movl	$-1, -44(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L161
.L119:
	movq	-24(%rbp), %rax
	movl	12(%rax), %edx
	movq	-16(%rbp), %rax
	movl	12(%rax), %eax
	cmpl	%eax, %edx
	jle	.L199
	movq	$5, -8(%rbp)
	jmp	.L161
.L199:
	movq	$72, -8(%rbp)
	jmp	.L161
.L154:
	movq	-24(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-24(%rbp), %rax
	movl	4(%rax), %esi
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-24(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L106:
	movq	-16(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-16(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	4(%rax), %esi
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$18, -8(%rbp)
	jmp	.L161
.L102:
	movq	-72(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-80(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L120:
	movq	-24(%rbp), %rax
	movl	12(%rax), %edx
	movq	-16(%rbp), %rax
	movl	12(%rax), %eax
	cmpl	%eax, %edx
	jge	.L201
	movq	$41, -8(%rbp)
	jmp	.L161
.L201:
	movq	$89, -8(%rbp)
	jmp	.L161
.L101:
	movq	-24(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	8(%rax), %eax
	cmpl	%eax, %edx
	jne	.L203
	movq	$79, -8(%rbp)
	jmp	.L161
.L203:
	movq	$82, -8(%rbp)
	jmp	.L161
.L136:
	movq	-16(%rbp), %rax
	movl	8(%rax), %eax
	testl	%eax, %eax
	je	.L205
	movq	$99, -8(%rbp)
	jmp	.L161
.L205:
	movq	$10, -8(%rbp)
	jmp	.L161
.L134:
	movl	$1, -36(%rbp)
	movq	$80, -8(%rbp)
	jmp	.L161
.L131:
	movq	-16(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-16(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	4(%rax), %esi
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L103:
	cmpl	$1, -40(%rbp)
	je	.L207
	cmpl	$1, -40(%rbp)
	jg	.L208
	cmpl	$-1, -40(%rbp)
	je	.L209
	cmpl	$0, -40(%rbp)
	je	.L210
	jmp	.L208
.L207:
	movq	$23, -8(%rbp)
	jmp	.L211
.L210:
	movq	$26, -8(%rbp)
	jmp	.L211
.L209:
	movq	$9, -8(%rbp)
	jmp	.L211
.L208:
	movq	$90, -8(%rbp)
	nop
.L211:
	jmp	.L161
.L105:
	movq	-24(%rbp), %rdx
	movq	-72(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L212
	movq	$94, -8(%rbp)
	jmp	.L161
.L212:
	movq	$18, -8(%rbp)
	jmp	.L161
.L149:
	movq	-24(%rbp), %rax
	movl	12(%rax), %eax
	testl	%eax, %eax
	je	.L214
	movq	$40, -8(%rbp)
	jmp	.L161
.L214:
	movq	$14, -8(%rbp)
	jmp	.L161
.L158:
	movq	-16(%rbp), %rax
	movl	12(%rax), %ecx
	movq	-16(%rbp), %rax
	movl	8(%rax), %edx
	movq	-16(%rbp), %rax
	movl	4(%rax), %esi
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movq	-88(%rbp), %rdi
	movq	%rdi, %r8
	movl	%eax, %edi
	call	attach
	movq	%rax, -88(%rbp)
	movq	-16(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$84, -8(%rbp)
	jmp	.L161
.L130:
	movq	-24(%rbp), %rax
	movl	12(%rax), %edx
	movq	-16(%rbp), %rax
	movl	12(%rax), %eax
	cmpl	%eax, %edx
	jge	.L216
	movq	$0, -8(%rbp)
	jmp	.L161
.L216:
	movq	$23, -8(%rbp)
	jmp	.L161
.L152:
	movl	$0, -48(%rbp)
	movq	$50, -8(%rbp)
	jmp	.L161
.L109:
	movq	$84, -8(%rbp)
	jmp	.L161
.L135:
	cmpl	$-1, -32(%rbp)
	je	.L218
	cmpl	$1, -32(%rbp)
	jne	.L219
	movq	$52, -8(%rbp)
	jmp	.L220
.L218:
	movq	$63, -8(%rbp)
	jmp	.L220
.L219:
	movq	$8, -8(%rbp)
	nop
.L220:
	jmp	.L161
.L111:
	movq	-24(%rbp), %rax
	movl	4(%rax), %edx
	movq	-16(%rbp), %rax
	movl	4(%rax), %eax
	cmpl	%eax, %edx
	jne	.L221
	movq	$7, -8(%rbp)
	jmp	.L161
.L221:
	movq	$100, -8(%rbp)
	jmp	.L161
.L156:
	movl	-44(%rbp), %eax
	movl	%eax, -40(%rbp)
	movq	$95, -8(%rbp)
	jmp	.L161
.L143:
	movq	-88(%rbp), %rax
	jmp	.L225
.L226:
	nop
.L161:
	jmp	.L224
.L225:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	poly_sum, .-poly_sum
	.section	.rodata
	.align 8
.LC21:
	.string	"\nEnter the value of x,y and z: "
.LC22:
	.string	"%d %d %d"
	.text
	.globl	poly_evaluate
	.type	poly_evaluate, @function
poly_evaluate:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -40(%rbp)
.L239:
	cmpq	$5, -40(%rbp)
	ja	.L242
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L230(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L230(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L230:
	.long	.L234-.L230
	.long	.L233-.L230
	.long	.L232-.L230
	.long	.L242-.L230
	.long	.L231-.L230
	.long	.L229-.L230
	.text
.L231:
	movq	-48(%rbp), %rdx
	movq	-72(%rbp), %rax
	cmpq	%rax, %rdx
	je	.L235
	movq	$2, -40(%rbp)
	jmp	.L237
.L235:
	movq	$5, -40(%rbp)
	jmp	.L237
.L233:
	movl	$0, -52(%rbp)
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-56(%rbp), %rcx
	leaq	-60(%rbp), %rdx
	leaq	-64(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-72(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -48(%rbp)
	movq	$4, -40(%rbp)
	jmp	.L237
.L229:
	movl	-52(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L240
	jmp	.L241
.L234:
	movq	$1, -40(%rbp)
	jmp	.L237
.L232:
	movq	-48(%rbp), %rax
	movl	4(%rax), %eax
	pxor	%xmm0, %xmm0
	cvtsi2sdl	%eax, %xmm0
	movl	-64(%rbp), %eax
	pxor	%xmm2, %xmm2
	cvtsi2sdl	%eax, %xmm2
	movq	%xmm2, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	pow@PLT
	movq	%xmm0, %rax
	movq	%rax, -32(%rbp)
	movq	-48(%rbp), %rax
	movl	8(%rax), %eax
	pxor	%xmm0, %xmm0
	cvtsi2sdl	%eax, %xmm0
	movl	-60(%rbp), %eax
	pxor	%xmm3, %xmm3
	cvtsi2sdl	%eax, %xmm3
	movq	%xmm3, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	pow@PLT
	movq	%xmm0, %rax
	movq	%rax, -24(%rbp)
	movq	-48(%rbp), %rax
	movl	12(%rax), %eax
	pxor	%xmm0, %xmm0
	cvtsi2sdl	%eax, %xmm0
	movl	-56(%rbp), %eax
	pxor	%xmm4, %xmm4
	cvtsi2sdl	%eax, %xmm4
	movq	%xmm4, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	pow@PLT
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	pxor	%xmm1, %xmm1
	cvtsi2sdl	-52(%rbp), %xmm1
	movq	-48(%rbp), %rax
	movl	(%rax), %eax
	pxor	%xmm0, %xmm0
	cvtsi2sdl	%eax, %xmm0
	mulsd	-32(%rbp), %xmm0
	mulsd	-24(%rbp), %xmm0
	mulsd	-16(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -52(%rbp)
	movq	-48(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -48(%rbp)
	movq	$4, -40(%rbp)
	jmp	.L237
.L242:
	nop
.L237:
	jmp	.L239
.L241:
	call	__stack_chk_fail@PLT
.L240:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	poly_evaluate, .-poly_evaluate
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
