	.file	"Melearner777_Guvi-Ds-with-C_4_flatten.c"
	.text
	.globl	_TIG_IZ_g9cp_argc
	.bss
	.align 4
	.type	_TIG_IZ_g9cp_argc, @object
	.size	_TIG_IZ_g9cp_argc, 4
_TIG_IZ_g9cp_argc:
	.zero	4
	.globl	_TIG_IZ_g9cp_argv
	.align 8
	.type	_TIG_IZ_g9cp_argv, @object
	.size	_TIG_IZ_g9cp_argv, 8
_TIG_IZ_g9cp_argv:
	.zero	8
	.globl	_TIG_IZ_g9cp_envp
	.align 8
	.type	_TIG_IZ_g9cp_envp, @object
	.size	_TIG_IZ_g9cp_envp, 8
_TIG_IZ_g9cp_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Invalid choice. Please try again."
.LC1:
	.string	"Exiting..."
.LC2:
	.string	"%d"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_g9cp_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_g9cp_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_g9cp_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 150 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-g9cp--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_g9cp_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_g9cp_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_g9cp_envp(%rip)
	nop
	movq	$4, -40(%rbp)
.L29:
	cmpq	$16, -40(%rbp)
	ja	.L32
	movq	-40(%rbp), %rax
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
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L32-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L32-.L8
	.long	.L10-.L8
	.long	.L32-.L8
	.long	.L32-.L8
	.long	.L32-.L8
	.long	.L32-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L15:
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	initLibrary
	movq	$0, -40(%rbp)
	jmp	.L19
.L9:
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	searchBook
	movq	$16, -40(%rbp)
	jmp	.L19
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$16, -40(%rbp)
	jmp	.L19
.L17:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L30
	jmp	.L31
.L16:
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	addBook
	movq	$16, -40(%rbp)
	jmp	.L19
.L7:
	movl	-44(%rbp), %eax
	cmpl	$4, %eax
	je	.L21
	movq	$0, -40(%rbp)
	jmp	.L19
.L21:
	movq	$7, -40(%rbp)
	jmp	.L19
.L13:
	leaq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	displayBooks
	movq	$16, -40(%rbp)
	jmp	.L19
.L14:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$16, -40(%rbp)
	jmp	.L19
.L10:
	movl	-44(%rbp), %eax
	cmpl	$4, %eax
	je	.L23
	cmpl	$4, %eax
	jg	.L24
	cmpl	$3, %eax
	je	.L25
	cmpl	$3, %eax
	jg	.L24
	cmpl	$1, %eax
	je	.L26
	cmpl	$2, %eax
	je	.L27
	jmp	.L24
.L23:
	movq	$5, -40(%rbp)
	jmp	.L28
.L25:
	movq	$15, -40(%rbp)
	jmp	.L28
.L27:
	movq	$6, -40(%rbp)
	jmp	.L28
.L26:
	movq	$3, -40(%rbp)
	jmp	.L28
.L24:
	movq	$8, -40(%rbp)
	nop
.L28:
	jmp	.L19
.L18:
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$10, -40(%rbp)
	jmp	.L19
.L12:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$1, -40(%rbp)
	jmp	.L19
.L32:
	nop
.L19:
	jmp	.L29
.L31:
	call	__stack_chk_fail@PLT
.L30:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.globl	initLibrary
	.type	initLibrary, @function
initLibrary:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	$1, -8(%rbp)
.L39:
	cmpq	$2, -8(%rbp)
	je	.L40
	cmpq	$2, -8(%rbp)
	ja	.L41
	cmpq	$0, -8(%rbp)
	je	.L36
	cmpq	$1, -8(%rbp)
	jne	.L41
	movq	$0, -8(%rbp)
	jmp	.L37
.L36:
	movq	-24(%rbp), %rax
	movq	$0, (%rax)
	movq	-24(%rbp), %rax
	movl	$0, 8(%rax)
	movq	-24(%rbp), %rax
	movl	$0, 12(%rax)
	movq	$2, -8(%rbp)
	jmp	.L37
.L41:
	nop
.L37:
	jmp	.L39
.L40:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	initLibrary, .-initLibrary
	.section	.rodata
.LC3:
	.string	"Books found:"
	.align 8
.LC4:
	.string	"No book found with the given title."
	.align 8
.LC5:
	.string	"Title: %s, Author: %s, Year: %d\n"
.LC6:
	.string	" %[^\n]"
	.text
	.globl	searchBook
	.type	searchBook, @function
searchBook:
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$3, -120(%rbp)
.L67:
	cmpq	$14, -120(%rbp)
	ja	.L70
	movq	-120(%rbp), %rax
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
	.long	.L56-.L45
	.long	.L71-.L45
	.long	.L54-.L45
	.long	.L53-.L45
	.long	.L70-.L45
	.long	.L70-.L45
	.long	.L52-.L45
	.long	.L51-.L45
	.long	.L50-.L45
	.long	.L49-.L45
	.long	.L48-.L45
	.long	.L70-.L45
	.long	.L47-.L45
	.long	.L46-.L45
	.long	.L44-.L45
	.text
.L44:
	cmpq	$0, -128(%rbp)
	je	.L57
	movq	$10, -120(%rbp)
	jmp	.L59
.L57:
	movq	$0, -120(%rbp)
	jmp	.L59
.L47:
	cmpl	$0, -136(%rbp)
	jne	.L60
	movq	$13, -120(%rbp)
	jmp	.L59
.L60:
	movq	$1, -120(%rbp)
	jmp	.L59
.L50:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, -136(%rbp)
	movq	$6, -120(%rbp)
	jmp	.L59
.L53:
	movq	$7, -120(%rbp)
	jmp	.L59
.L49:
	movq	-152(%rbp), %rax
	movq	(%rax), %rdx
	movl	-132(%rbp), %eax
	cltq
	imulq	$204, %rax, %rax
	addq	%rdx, %rax
	movq	%rax, %rdx
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strstr@PLT
	movq	%rax, -128(%rbp)
	movq	$14, -120(%rbp)
	jmp	.L59
.L46:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -120(%rbp)
	jmp	.L59
.L52:
	movq	-152(%rbp), %rax
	movq	(%rax), %rdx
	movl	-132(%rbp), %eax
	cltq
	imulq	$204, %rax, %rax
	addq	%rdx, %rax
	movl	200(%rax), %eax
	movq	-152(%rbp), %rdx
	movq	(%rdx), %rcx
	movl	-132(%rbp), %edx
	movslq	%edx, %rdx
	imulq	$204, %rdx, %rdx
	addq	%rcx, %rdx
	addq	$100, %rdx
	movq	-152(%rbp), %rcx
	movq	(%rcx), %rsi
	movl	-132(%rbp), %ecx
	movslq	%ecx, %rcx
	imulq	$204, %rcx, %rcx
	addq	%rsi, %rcx
	movq	%rcx, %rsi
	movl	%eax, %ecx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -120(%rbp)
	jmp	.L59
.L48:
	cmpl	$0, -136(%rbp)
	jne	.L63
	movq	$8, -120(%rbp)
	jmp	.L59
.L63:
	movq	$6, -120(%rbp)
	jmp	.L59
.L56:
	addl	$1, -132(%rbp)
	movq	$2, -120(%rbp)
	jmp	.L59
.L51:
	leaq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -136(%rbp)
	movl	$0, -132(%rbp)
	movq	$2, -120(%rbp)
	jmp	.L59
.L54:
	movq	-152(%rbp), %rax
	movl	8(%rax), %eax
	cmpl	%eax, -132(%rbp)
	jge	.L65
	movq	$9, -120(%rbp)
	jmp	.L59
.L65:
	movq	$12, -120(%rbp)
	jmp	.L59
.L70:
	nop
.L59:
	jmp	.L67
.L71:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L69
	call	__stack_chk_fail@PLT
.L69:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	searchBook, .-searchBook
	.section	.rodata
.LC7:
	.string	"Book added successfully."
.LC8:
	.string	"Memory allocation failed."
	.text
	.globl	addBook
	.type	addBook, @function
addBook:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$296, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -296(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$8, -256(%rbp)
.L94:
	cmpq	$13, -256(%rbp)
	ja	.L97
	movq	-256(%rbp), %rax
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
	.long	.L98-.L75
	.long	.L84-.L75
	.long	.L97-.L75
	.long	.L98-.L75
	.long	.L82-.L75
	.long	.L81-.L75
	.long	.L97-.L75
	.long	.L80-.L75
	.long	.L79-.L75
	.long	.L78-.L75
	.long	.L77-.L75
	.long	.L97-.L75
	.long	.L76-.L75
	.long	.L74-.L75
	.text
.L82:
	movl	-272(%rbp), %eax
	movl	%eax, -276(%rbp)
	movl	-276(%rbp), %eax
	cltq
	imulq	$204, %rax, %rdx
	movq	-296(%rbp), %rax
	movq	(%rax), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movq	%rax, -248(%rbp)
	movq	-248(%rbp), %rax
	movq	%rax, -264(%rbp)
	movq	$9, -256(%rbp)
	jmp	.L86
.L76:
	leaq	-240(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-240(%rbp), %rax
	addq	$100, %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-240(%rbp), %rax
	addq	$200, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	-296(%rbp), %rax
	movl	8(%rax), %eax
	movl	%eax, -268(%rbp)
	movq	-296(%rbp), %rax
	movl	8(%rax), %eax
	leal	1(%rax), %edx
	movq	-296(%rbp), %rax
	movl	%edx, 8(%rax)
	movq	-296(%rbp), %rax
	movq	(%rax), %rdx
	movl	-268(%rbp), %eax
	cltq
	imulq	$204, %rax, %rax
	addq	%rdx, %rax
	movq	-240(%rbp), %rcx
	movq	-232(%rbp), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	-224(%rbp), %rcx
	movq	-216(%rbp), %rbx
	movq	%rcx, 16(%rax)
	movq	%rbx, 24(%rax)
	movq	-208(%rbp), %rcx
	movq	-200(%rbp), %rbx
	movq	%rcx, 32(%rax)
	movq	%rbx, 40(%rax)
	movq	-192(%rbp), %rcx
	movq	-184(%rbp), %rbx
	movq	%rcx, 48(%rax)
	movq	%rbx, 56(%rax)
	movq	-176(%rbp), %rcx
	movq	-168(%rbp), %rbx
	movq	%rcx, 64(%rax)
	movq	%rbx, 72(%rax)
	movq	-160(%rbp), %rcx
	movq	-152(%rbp), %rbx
	movq	%rcx, 80(%rax)
	movq	%rbx, 88(%rax)
	movq	-144(%rbp), %rcx
	movq	-136(%rbp), %rbx
	movq	%rcx, 96(%rax)
	movq	%rbx, 104(%rax)
	movq	-128(%rbp), %rcx
	movq	-120(%rbp), %rbx
	movq	%rcx, 112(%rax)
	movq	%rbx, 120(%rax)
	movq	-112(%rbp), %rcx
	movq	-104(%rbp), %rbx
	movq	%rcx, 128(%rax)
	movq	%rbx, 136(%rax)
	movq	-96(%rbp), %rcx
	movq	-88(%rbp), %rbx
	movq	%rcx, 144(%rax)
	movq	%rbx, 152(%rax)
	movq	-80(%rbp), %rcx
	movq	-72(%rbp), %rbx
	movq	%rcx, 160(%rax)
	movq	%rbx, 168(%rax)
	movq	-64(%rbp), %rcx
	movq	-56(%rbp), %rbx
	movq	%rcx, 176(%rax)
	movq	%rbx, 184(%rax)
	movq	-48(%rbp), %rdx
	movq	%rdx, 192(%rax)
	movl	-40(%rbp), %edx
	movl	%edx, 200(%rax)
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -256(%rbp)
	jmp	.L86
.L79:
	movq	-296(%rbp), %rax
	movl	8(%rax), %edx
	movq	-296(%rbp), %rax
	movl	12(%rax), %eax
	cmpl	%eax, %edx
	jne	.L87
	movq	$1, -256(%rbp)
	jmp	.L86
.L87:
	movq	$12, -256(%rbp)
	jmp	.L86
.L84:
	movq	-296(%rbp), %rax
	movl	12(%rax), %eax
	testl	%eax, %eax
	jne	.L89
	movq	$13, -256(%rbp)
	jmp	.L86
.L89:
	movq	$7, -256(%rbp)
	jmp	.L86
.L78:
	cmpq	$0, -264(%rbp)
	jne	.L92
	movq	$10, -256(%rbp)
	jmp	.L86
.L92:
	movq	$5, -256(%rbp)
	jmp	.L86
.L74:
	movl	$1, -272(%rbp)
	movq	$4, -256(%rbp)
	jmp	.L86
.L81:
	movq	-296(%rbp), %rax
	movq	-264(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-296(%rbp), %rax
	movl	-276(%rbp), %edx
	movl	%edx, 12(%rax)
	movq	$12, -256(%rbp)
	jmp	.L86
.L77:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -256(%rbp)
	jmp	.L86
.L80:
	movq	-296(%rbp), %rax
	movl	12(%rax), %eax
	addl	%eax, %eax
	movl	%eax, -272(%rbp)
	movq	$4, -256(%rbp)
	jmp	.L86
.L97:
	nop
.L86:
	jmp	.L94
.L98:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L96
	call	__stack_chk_fail@PLT
.L96:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	addBook, .-addBook
	.section	.rodata
.LC9:
	.string	"No books in the library."
.LC10:
	.string	"Books in library:"
	.text
	.globl	displayBooks
	.type	displayBooks, @function
displayBooks:
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
	movq	$7, -8(%rbp)
.L115:
	cmpq	$9, -8(%rbp)
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
	.long	.L108-.L102
	.long	.L116-.L102
	.long	.L107-.L102
	.long	.L106-.L102
	.long	.L116-.L102
	.long	.L105-.L102
	.long	.L117-.L102
	.long	.L103-.L102
	.long	.L116-.L102
	.long	.L117-.L102
	.text
.L106:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -8(%rbp)
	jmp	.L109
.L105:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L109
.L108:
	movq	-24(%rbp), %rax
	movq	(%rax), %rdx
	movl	-12(%rbp), %eax
	cltq
	imulq	$204, %rax, %rax
	addq	%rdx, %rax
	movl	200(%rax), %eax
	movq	-24(%rbp), %rdx
	movq	(%rdx), %rcx
	movl	-12(%rbp), %edx
	movslq	%edx, %rdx
	imulq	$204, %rdx, %rdx
	addq	%rcx, %rdx
	addq	$100, %rdx
	movq	-24(%rbp), %rcx
	movq	(%rcx), %rsi
	movl	-12(%rbp), %ecx
	movslq	%ecx, %rcx
	imulq	$204, %rcx, %rcx
	addq	%rsi, %rcx
	movq	%rcx, %rsi
	movl	%eax, %ecx
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L109
.L103:
	movq	-24(%rbp), %rax
	movl	8(%rax), %eax
	testl	%eax, %eax
	jne	.L111
	movq	$3, -8(%rbp)
	jmp	.L109
.L111:
	movq	$5, -8(%rbp)
	jmp	.L109
.L107:
	movq	-24(%rbp), %rax
	movl	8(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jge	.L113
	movq	$0, -8(%rbp)
	jmp	.L109
.L113:
	movq	$6, -8(%rbp)
	jmp	.L109
.L116:
	nop
.L109:
	jmp	.L115
.L117:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	displayBooks, .-displayBooks
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
