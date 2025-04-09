	.file	"Kunwar-Yuvraj_ada-lab_2_flatten.c"
	.text
	.globl	_TIG_IZ_nosc_argc
	.bss
	.align 4
	.type	_TIG_IZ_nosc_argc, @object
	.size	_TIG_IZ_nosc_argc, 4
_TIG_IZ_nosc_argc:
	.zero	4
	.globl	count
	.align 4
	.type	count, @object
	.size	count, 4
count:
	.zero	4
	.globl	_TIG_IZ_nosc_envp
	.align 8
	.type	_TIG_IZ_nosc_envp, @object
	.size	_TIG_IZ_nosc_envp, 8
_TIG_IZ_nosc_envp:
	.zero	8
	.globl	_TIG_IZ_nosc_argv
	.align 8
	.type	_TIG_IZ_nosc_argv, @object
	.size	_TIG_IZ_nosc_argv, 8
_TIG_IZ_nosc_argv:
	.zero	8
	.text
	.globl	binarySearch
	.type	binarySearch, @function
binarySearch:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movl	%edx, -48(%rbp)
	movq	$11, -8(%rbp)
.L26:
	cmpq	$14, -8(%rbp)
	ja	.L27
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
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L27-.L4
	.long	.L27-.L4
	.long	.L9-.L4
	.long	.L27-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L11:
	movl	-16(%rbp), %eax
	subl	-20(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L16
.L3:
	movl	-12(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L16
.L6:
	movl	-12(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L16
.L9:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -48(%rbp)
	jne	.L17
	movq	$5, -8(%rbp)
	jmp	.L16
.L17:
	movq	$10, -8(%rbp)
	jmp	.L16
.L14:
	movl	$0, count(%rip)
	movl	$0, -20(%rbp)
	movl	-44(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -16(%rbp)
	movl	-16(%rbp), %eax
	subl	-20(%rbp), %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L16
.L12:
	movl	count(%rip), %eax
	testl	%eax, %eax
	je	.L19
	movq	$2, -8(%rbp)
	jmp	.L16
.L19:
	movq	$13, -8(%rbp)
	jmp	.L16
.L7:
	movq	$1, -8(%rbp)
	jmp	.L16
.L5:
	movl	$-1, %eax
	jmp	.L21
.L10:
	movl	-12(%rbp), %eax
	jmp	.L21
.L8:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -48(%rbp)
	jle	.L22
	movq	$14, -8(%rbp)
	jmp	.L16
.L22:
	movq	$12, -8(%rbp)
	jmp	.L16
.L15:
	movl	count(%rip), %eax
	addl	$1, %eax
	movl	%eax, count(%rip)
	movq	$3, -8(%rbp)
	jmp	.L16
.L13:
	movl	-20(%rbp), %eax
	cmpl	-16(%rbp), %eax
	jg	.L24
	movq	$8, -8(%rbp)
	jmp	.L16
.L24:
	movq	$13, -8(%rbp)
	jmp	.L16
.L27:
	nop
.L16:
	jmp	.L26
.L21:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	binarySearch, .-binarySearch
	.section	.rodata
.LC0:
	.string	"%d\t%d\n"
.LC1:
	.string	"w"
	.text
	.globl	writeFiles
	.type	writeFiles, @function
writeFiles:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movq	%rdi, -88(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%rcx, -112(%rbp)
	movq	$20, -16(%rbp)
.L56:
	movq	-16(%rbp), %rax
	subq	$4, %rax
	cmpq	$24, %rax
	ja	.L57
	leaq	0(,%rax,4), %rdx
	leaq	.L31(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L31(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L31:
	.long	.L45-.L31
	.long	.L44-.L31
	.long	.L57-.L31
	.long	.L57-.L31
	.long	.L57-.L31
	.long	.L43-.L31
	.long	.L42-.L31
	.long	.L57-.L31
	.long	.L57-.L31
	.long	.L57-.L31
	.long	.L41-.L31
	.long	.L40-.L31
	.long	.L39-.L31
	.long	.L38-.L31
	.long	.L57-.L31
	.long	.L57-.L31
	.long	.L37-.L31
	.long	.L57-.L31
	.long	.L58-.L31
	.long	.L35-.L31
	.long	.L34-.L31
	.long	.L33-.L31
	.long	.L57-.L31
	.long	.L32-.L31
	.long	.L30-.L31
	.text
.L33:
	movl	-72(%rbp), %ecx
	movq	-24(%rbp), %rax
	movq	-112(%rbp), %r8
	movl	$1, %edx
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	*%r8
	movl	count(%rip), %ecx
	movl	-72(%rbp), %edx
	movq	-40(%rbp), %rax
	leaq	.LC0(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$0, -60(%rbp)
	movq	$27, -16(%rbp)
	jmp	.L46
.L45:
	call	rand@PLT
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	cltd
	idivl	-72(%rbp)
	movl	-72(%rbp), %ecx
	movq	-24(%rbp), %rax
	movq	-112(%rbp), %r8
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	*%r8
	movl	count(%rip), %ecx
	movl	-72(%rbp), %edx
	movq	-32(%rbp), %rax
	leaq	.LC0(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	sall	-72(%rbp)
	movq	$15, -16(%rbp)
	jmp	.L46
.L41:
	call	rand@PLT
	movl	%eax, -56(%rbp)
	movl	-60(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	leaq	(%rdx,%rax), %rcx
	movl	-56(%rbp), %eax
	cltd
	idivl	-72(%rbp)
	movl	%edx, %eax
	movl	%eax, (%rcx)
	addl	$1, -60(%rbp)
	movq	$27, -16(%rbp)
	jmp	.L46
.L40:
	cmpl	$1024, -72(%rbp)
	jg	.L47
	movq	$17, -16(%rbp)
	jmp	.L46
.L47:
	movq	$9, -16(%rbp)
	jmp	.L46
.L35:
	movl	-68(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	$1, (%rax)
	addl	$1, -68(%rbp)
	movq	$24, -16(%rbp)
	jmp	.L46
.L39:
	movl	-72(%rbp), %ecx
	movq	-24(%rbp), %rax
	movq	-112(%rbp), %r8
	movl	$1, %edx
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	*%r8
	movl	count(%rip), %ecx
	movl	-72(%rbp), %edx
	movq	-48(%rbp), %rax
	leaq	.LC0(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$0, -64(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L46
.L34:
	movl	-68(%rbp), %eax
	cmpl	-72(%rbp), %eax
	jge	.L49
	movq	$23, -16(%rbp)
	jmp	.L46
.L49:
	movq	$16, -16(%rbp)
	jmp	.L46
.L43:
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$22, -16(%rbp)
	jmp	.L46
.L38:
	movl	-72(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movl	$0, -68(%rbp)
	movq	$24, -16(%rbp)
	jmp	.L46
.L32:
	movl	-60(%rbp), %eax
	cmpl	-72(%rbp), %eax
	jge	.L51
	movq	$14, -16(%rbp)
	jmp	.L46
.L51:
	movq	$4, -16(%rbp)
	jmp	.L46
.L30:
	movq	-88(%rbp), %rax
	leaq	.LC1(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -48(%rbp)
	movq	-96(%rbp), %rax
	leaq	.LC1(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -40(%rbp)
	movq	-104(%rbp), %rax
	leaq	.LC1(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -32(%rbp)
	movl	$2, -72(%rbp)
	movq	$15, -16(%rbp)
	jmp	.L46
.L44:
	movl	-64(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	$0, (%rax)
	addl	$1, -64(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L46
.L42:
	movl	-64(%rbp), %eax
	cmpl	-72(%rbp), %eax
	jge	.L54
	movq	$5, -16(%rbp)
	jmp	.L46
.L54:
	movq	$25, -16(%rbp)
	jmp	.L46
.L37:
	movq	$28, -16(%rbp)
	jmp	.L46
.L57:
	nop
.L46:
	jmp	.L56
.L58:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	writeFiles, .-writeFiles
	.section	.rodata
.LC2:
	.string	"Invalid choice!"
.LC3:
	.string	"1. Tester\n2. Plotter"
.LC4:
	.string	"Enter your choice: "
.LC5:
	.string	"%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, count(%rip)
	nop
.L60:
	movq	$0, _TIG_IZ_nosc_envp(%rip)
	nop
.L61:
	movq	$0, _TIG_IZ_nosc_argv(%rip)
	nop
.L62:
	movl	$0, _TIG_IZ_nosc_argc(%rip)
	nop
	nop
.L63:
.L64:
#APP
# 137 "Kunwar-Yuvraj_ada-lab_2.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-nosc--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_nosc_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_nosc_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_nosc_envp(%rip)
	nop
	movq	$4, -16(%rbp)
.L81:
	cmpq	$7, -16(%rbp)
	ja	.L84
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L67(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L67(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L67:
	.long	.L74-.L67
	.long	.L73-.L67
	.long	.L72-.L67
	.long	.L71-.L67
	.long	.L70-.L67
	.long	.L69-.L67
	.long	.L68-.L67
	.long	.L66-.L67
	.text
.L70:
	movq	$6, -16(%rbp)
	jmp	.L75
.L73:
	movl	-20(%rbp), %eax
	cmpl	$2, %eax
	jne	.L76
	movq	$2, -16(%rbp)
	jmp	.L75
.L76:
	movq	$3, -16(%rbp)
	jmp	.L75
.L71:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -16(%rbp)
	jmp	.L75
.L68:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -16(%rbp)
	jmp	.L75
.L69:
	call	tester
	movq	$7, -16(%rbp)
	jmp	.L75
.L74:
	movl	-20(%rbp), %eax
	cmpl	$1, %eax
	jne	.L78
	movq	$5, -16(%rbp)
	jmp	.L75
.L78:
	movq	$1, -16(%rbp)
	jmp	.L75
.L66:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L82
	jmp	.L83
.L72:
	call	plotter
	movq	$7, -16(%rbp)
	jmp	.L75
.L84:
	nop
.L75:
	jmp	.L81
.L83:
	call	__stack_chk_fail@PLT
.L82:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.globl	linearSearch
	.type	linearSearch, @function
linearSearch:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	%edx, -32(%rbp)
	movq	$0, -8(%rbp)
.L102:
	cmpq	$8, -8(%rbp)
	ja	.L103
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L88(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L88(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L88:
	.long	.L95-.L88
	.long	.L94-.L88
	.long	.L93-.L88
	.long	.L92-.L88
	.long	.L91-.L88
	.long	.L90-.L88
	.long	.L89-.L88
	.long	.L103-.L88
	.long	.L87-.L88
	.text
.L91:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -32(%rbp)
	jne	.L96
	movq	$2, -8(%rbp)
	jmp	.L98
.L96:
	movq	$1, -8(%rbp)
	jmp	.L98
.L87:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L99
	movq	$5, -8(%rbp)
	jmp	.L98
.L99:
	movq	$3, -8(%rbp)
	jmp	.L98
.L94:
	addl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L98
.L92:
	movl	$-1, %eax
	jmp	.L101
.L89:
	movl	$0, count(%rip)
	movl	$0, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L98
.L90:
	movl	count(%rip), %eax
	addl	$1, %eax
	movl	%eax, count(%rip)
	movq	$4, -8(%rbp)
	jmp	.L98
.L95:
	movq	$6, -8(%rbp)
	jmp	.L98
.L93:
	movl	-12(%rbp), %eax
	jmp	.L101
.L103:
	nop
.L98:
	jmp	.L102
.L101:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	linearSearch, .-linearSearch
	.section	.rodata
.LC6:
	.string	"Enter elements: "
.LC7:
	.string	"Enter key to search: "
.LC8:
	.string	"Index by Binary search: %d\n"
.LC9:
	.string	"Index by Linear search: %d\n"
.LC10:
	.string	"Enter number of elements: "
	.text
	.globl	tester
	.type	tester, @function
tester:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	subq	$80, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	movq	%fs:40, %rax
	movq	%rax, -40(%rbp)
	xorl	%eax, %eax
	movq	$0, -72(%rbp)
.L122:
	cmpq	$12, -72(%rbp)
	ja	.L125
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L107(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L107(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L107:
	.long	.L114-.L107
	.long	.L113-.L107
	.long	.L125-.L107
	.long	.L125-.L107
	.long	.L112-.L107
	.long	.L111-.L107
	.long	.L125-.L107
	.long	.L125-.L107
	.long	.L110-.L107
	.long	.L109-.L107
	.long	.L125-.L107
	.long	.L108-.L107
	.long	.L126-.L107
	.text
.L112:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -92(%rbp)
	movq	$5, -72(%rbp)
	jmp	.L115
.L110:
	movl	-100(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
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
.L117:
	cmpq	%rdx, %rsp
	je	.L118
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L117
.L118:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L119
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L119:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -80(%rbp)
	movq	$4, -72(%rbp)
	jmp	.L115
.L113:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-96(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-96(%rbp), %edx
	movl	-100(%rbp), %ecx
	movl	-100(%rbp), %eax
	movslq	%eax, %rsi
	subq	$1, %rsi
	movq	%rsi, -56(%rbp)
	cltq
	movq	%rax, %r12
	movl	$0, %r13d
	movq	-80(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	binarySearch
	movl	%eax, -88(%rbp)
	movl	-88(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-96(%rbp), %edx
	movl	-100(%rbp), %ecx
	movl	-100(%rbp), %eax
	movslq	%eax, %rsi
	subq	$1, %rsi
	movq	%rsi, -48(%rbp)
	cltq
	movq	%rax, %r14
	movl	$0, %r15d
	movq	-80(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	linearSearch
	movl	%eax, -84(%rbp)
	movl	-84(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -72(%rbp)
	jmp	.L115
.L108:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-100(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$8, -72(%rbp)
	jmp	.L115
.L109:
	movl	-92(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -92(%rbp)
	movq	$5, -72(%rbp)
	jmp	.L115
.L111:
	movl	-100(%rbp), %eax
	cmpl	%eax, -92(%rbp)
	jge	.L120
	movq	$9, -72(%rbp)
	jmp	.L115
.L120:
	movq	$1, -72(%rbp)
	jmp	.L115
.L114:
	movq	$11, -72(%rbp)
	jmp	.L115
.L125:
	nop
.L115:
	jmp	.L122
.L126:
	nop
	movq	-40(%rbp), %rax
	subq	%fs:40, %rax
	je	.L124
	call	__stack_chk_fail@PLT
.L124:
	leaq	-32(%rbp), %rsp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	tester, .-tester
	.section	.rodata
.LC11:
	.string	"Generating files..."
.LC12:
	.string	"avg-linear.txt"
.LC13:
	.string	"worst-linear.txt"
.LC14:
	.string	"best-linear.txt"
.LC15:
	.string	"avg-binary.txt"
.LC16:
	.string	"worst-binary.txt"
.LC17:
	.string	"best-binary.txt"
.LC18:
	.string	"Files generated successfully!"
	.text
	.globl	plotter
	.type	plotter, @function
plotter:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L133:
	cmpq	$2, -8(%rbp)
	je	.L128
	cmpq	$2, -8(%rbp)
	ja	.L135
	cmpq	$0, -8(%rbp)
	je	.L130
	cmpq	$1, -8(%rbp)
	jne	.L135
	jmp	.L134
.L130:
	movq	$2, -8(%rbp)
	jmp	.L132
.L128:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	linearSearch(%rip), %rax
	movq	%rax, %rcx
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdx
	leaq	.LC13(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	writeFiles
	leaq	binarySearch(%rip), %rax
	movq	%rax, %rcx
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdx
	leaq	.LC16(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	call	writeFiles
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L132
.L135:
	nop
.L132:
	jmp	.L133
.L134:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	plotter, .-plotter
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
