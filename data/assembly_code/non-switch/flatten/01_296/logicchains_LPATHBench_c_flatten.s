	.file	"logicchains_LPATHBench_c_flatten.c"
	.text
	.globl	_TIG_IZ_Tsjs_argc
	.bss
	.align 4
	.type	_TIG_IZ_Tsjs_argc, @object
	.size	_TIG_IZ_Tsjs_argc, 4
_TIG_IZ_Tsjs_argc:
	.zero	4
	.globl	_TIG_IZ_Tsjs_argv
	.align 8
	.type	_TIG_IZ_Tsjs_argv, @object
	.size	_TIG_IZ_Tsjs_argv, 8
_TIG_IZ_Tsjs_argv:
	.zero	8
	.globl	_TIG_IZ_Tsjs_envp
	.align 8
	.type	_TIG_IZ_Tsjs_envp, @object
	.size	_TIG_IZ_Tsjs_envp, 8
_TIG_IZ_Tsjs_envp:
	.zero	8
	.text
	.type	bitmap_is_set, @function
bitmap_is_set:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$0, -8(%rbp)
.L4:
	cmpq	$0, -8(%rbp)
	jne	.L7
	movq	-32(%rbp), %rax
	shrq	$6, %rax
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rdx
	movq	-32(%rbp), %rax
	andl	$63, %eax
	movl	%eax, %ecx
	shrq	%cl, %rdx
	movq	%rdx, %rax
	andl	$1, %eax
	testq	%rax, %rax
	setne	%al
	jmp	.L6
.L7:
	nop
	jmp	.L4
.L6:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	bitmap_is_set, .-bitmap_is_set
	.globl	max_distance_simple
	.type	max_distance_simple, @function
max_distance_simple:
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
	movq	%rsi, -64(%rbp)
	movq	$13, -8(%rbp)
.L30:
	cmpq	$15, -8(%rbp)
	ja	.L32
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L11(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L11(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L11:
	.long	.L21-.L11
	.long	.L20-.L11
	.long	.L32-.L11
	.long	.L19-.L11
	.long	.L18-.L11
	.long	.L17-.L11
	.long	.L32-.L11
	.long	.L32-.L11
	.long	.L32-.L11
	.long	.L16-.L11
	.long	.L15-.L11
	.long	.L14-.L11
	.long	.L13-.L11
	.long	.L12-.L11
	.long	.L32-.L11
	.long	.L10-.L11
	.text
.L18:
	addl	$1, -40(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L22
.L10:
	movq	-64(%rbp), %rax
	leaq	4(%rax), %rdx
	movl	-40(%rbp), %eax
	salq	$3, %rax
	addq	%rdx, %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %eax
	leaq	0(,%rax,8), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L22
.L13:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	andl	$2147483647, %eax
	movl	%eax, %edx
	movq	-64(%rbp), %rax
	movl	%edx, (%rax)
	movq	$11, -8(%rbp)
	jmp	.L22
.L20:
	movl	-36(%rbp), %eax
	cmpl	-48(%rbp), %eax
	jbe	.L23
	movq	$0, -8(%rbp)
	jmp	.L22
.L23:
	movq	$4, -8(%rbp)
	jmp	.L22
.L19:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	jns	.L25
	movq	$4, -8(%rbp)
	jmp	.L22
.L25:
	movq	$10, -8(%rbp)
	jmp	.L22
.L14:
	movl	-48(%rbp), %eax
	jmp	.L31
.L16:
	movl	-40(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jnb	.L28
	movq	$15, -8(%rbp)
	jmp	.L22
.L28:
	movq	$12, -8(%rbp)
	jmp	.L22
.L12:
	movq	$5, -8(%rbp)
	jmp	.L22
.L17:
	movl	$0, -48(%rbp)
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -44(%rbp)
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	orl	$-2147483648, %eax
	movl	%eax, %edx
	movq	-64(%rbp), %rax
	movl	%edx, (%rax)
	movl	$0, -40(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L22
.L15:
	movq	-16(%rbp), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	max_distance_simple
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, -28(%rbp)
	movq	-24(%rbp), %rax
	movl	4(%rax), %edx
	movl	-28(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -36(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L22
.L21:
	movl	-36(%rbp), %eax
	movl	%eax, -48(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L22
.L32:
	nop
.L22:
	jmp	.L30
.L31:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	max_distance_simple, .-max_distance_simple
	.type	bitmap_set_bit, @function
bitmap_set_bit:
.LFB5:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$1, -8(%rbp)
.L38:
	cmpq	$0, -8(%rbp)
	je	.L39
	cmpq	$1, -8(%rbp)
	jne	.L40
	movq	-32(%rbp), %rax
	shrq	$6, %rax
	leaq	0(,%rax,8), %rcx
	movq	-24(%rbp), %rdx
	addq	%rcx, %rdx
	movq	(%rdx), %rsi
	movq	-32(%rbp), %rdx
	andl	$63, %edx
	movl	$1, %edi
	movl	%edx, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rdx
	leaq	0(,%rax,8), %rcx
	movq	-24(%rbp), %rax
	addq	%rcx, %rax
	orq	%rsi, %rdx
	movq	%rdx, (%rax)
	movq	$0, -8(%rbp)
	jmp	.L36
.L40:
	nop
.L36:
	jmp	.L38
.L39:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	bitmap_set_bit, .-bitmap_set_bit
	.section	.rodata
.LC0:
	.string	"%d LANGUAGE C%s %ld\n"
.LC1:
	.string	"agraph"
.LC2:
	.string	""
	.text
	.globl	main
	.type	main, @function
main:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movl	%edi, -116(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_Tsjs_envp(%rip)
	nop
.L42:
	movq	$0, _TIG_IZ_Tsjs_argv(%rip)
	nop
.L43:
	movl	$0, _TIG_IZ_Tsjs_argc(%rip)
	nop
	nop
.L44:
.L45:
#APP
# 173 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Tsjs--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_Tsjs_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_Tsjs_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_Tsjs_envp(%rip)
	nop
	movq	$10, -96(%rbp)
.L59:
	cmpq	$11, -96(%rbp)
	ja	.L62
	movq	-96(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L48(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L48(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L48:
	.long	.L54-.L48
	.long	.L53-.L48
	.long	.L52-.L48
	.long	.L62-.L48
	.long	.L51-.L48
	.long	.L62-.L48
	.long	.L50-.L48
	.long	.L62-.L48
	.long	.L62-.L48
	.long	.L62-.L48
	.long	.L49-.L48
	.long	.L47-.L48
	.text
.L51:
	movq	-48(%rbp), %rax
	movq	-64(%rbp), %rdx
	subq	%rdx, %rax
	movq	%rax, -32(%rbp)
	movq	-40(%rbp), %rax
	movq	-56(%rbp), %rdx
	subq	%rdx, %rax
	movq	%rax, -24(%rbp)
	movq	$11, -96(%rbp)
	jmp	.L55
.L53:
	movq	-32(%rbp), %rax
	subq	$1, %rax
	movq	%rax, -32(%rbp)
	movq	-24(%rbp), %rax
	addq	$1000000, %rax
	movq	%rax, -24(%rbp)
	movq	$6, -96(%rbp)
	jmp	.L55
.L47:
	movq	-24(%rbp), %rax
	testq	%rax, %rax
	jns	.L56
	movq	$1, -96(%rbp)
	jmp	.L55
.L56:
	movq	$6, -96(%rbp)
	jmp	.L55
.L50:
	movq	-32(%rbp), %rax
	imulq	$1000, %rax, %rsi
	movq	-24(%rbp), %rcx
	movabsq	$2361183241434822607, %rdx
	movq	%rcx, %rax
	imulq	%rdx
	movq	%rdx, %rax
	sarq	$7, %rax
	sarq	$63, %rcx
	movq	%rcx, %rdx
	subq	%rdx, %rax
	addq	%rsi, %rax
	movq	%rax, -88(%rbp)
	movq	-88(%rbp), %rcx
	movq	-104(%rbp), %rdx
	movl	-112(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -96(%rbp)
	jmp	.L55
.L49:
	movq	$2, -96(%rbp)
	jmp	.L55
.L54:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L60
	jmp	.L61
.L52:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	read_graph_file
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, -72(%rbp)
	leaq	-64(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	gettimeofday@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, -104(%rbp)
	movq	-72(%rbp), %rax
	movq	(%rax), %rdx
	movq	-72(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	max_distance_branchless
	movl	%eax, -108(%rbp)
	movl	-108(%rbp), %eax
	movl	%eax, -112(%rbp)
	leaq	-48(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	gettimeofday@PLT
	movq	$4, -96(%rbp)
	jmp	.L55
.L62:
	nop
.L55:
	jmp	.L59
.L61:
	call	__stack_chk_fail@PLT
.L60:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	main, .-main
	.type	bitmap_clear_bit, @function
bitmap_clear_bit:
.LFB10:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	$0, -8(%rbp)
.L68:
	cmpq	$0, -8(%rbp)
	je	.L64
	cmpq	$1, -8(%rbp)
	jne	.L70
	jmp	.L69
.L64:
	movq	-32(%rbp), %rax
	shrq	$6, %rax
	leaq	0(,%rax,8), %rcx
	movq	-24(%rbp), %rdx
	addq	%rcx, %rdx
	movq	(%rdx), %rsi
	movq	-32(%rbp), %rdx
	andl	$63, %edx
	movl	$1, %edi
	movl	%edx, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rdx
	notq	%rdx
	leaq	0(,%rax,8), %rcx
	movq	-24(%rbp), %rax
	addq	%rcx, %rax
	andq	%rsi, %rdx
	movq	%rdx, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L67
.L70:
	nop
.L67:
	jmp	.L68
.L69:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	bitmap_clear_bit, .-bitmap_clear_bit
	.section	.rodata
.LC3:
	.string	"Invalid node at line %d\n"
	.align 8
.LC4:
	.string	"Could not open graph file '%s'\n"
.LC5:
	.string	"%u %u %u\n"
	.align 8
.LC6:
	.string	"Error at line %d before end of file\n"
	.align 8
.LC7:
	.string	"First line should be the non-zero number of nodes"
.LC8:
	.string	"r"
.LC9:
	.string	"%u\n"
	.text
	.globl	read_graph_file
	.type	read_graph_file, @function
read_graph_file:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movq	%rdi, -120(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$38, -40(%rbp)
.L125:
	cmpq	$42, -40(%rbp)
	ja	.L128
	movq	-40(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L74(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L74(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L74:
	.long	.L102-.L74
	.long	.L101-.L74
	.long	.L100-.L74
	.long	.L128-.L74
	.long	.L99-.L74
	.long	.L128-.L74
	.long	.L98-.L74
	.long	.L97-.L74
	.long	.L128-.L74
	.long	.L96-.L74
	.long	.L95-.L74
	.long	.L128-.L74
	.long	.L94-.L74
	.long	.L93-.L74
	.long	.L128-.L74
	.long	.L92-.L74
	.long	.L128-.L74
	.long	.L91-.L74
	.long	.L90-.L74
	.long	.L89-.L74
	.long	.L88-.L74
	.long	.L87-.L74
	.long	.L86-.L74
	.long	.L128-.L74
	.long	.L85-.L74
	.long	.L84-.L74
	.long	.L83-.L74
	.long	.L128-.L74
	.long	.L82-.L74
	.long	.L81-.L74
	.long	.L128-.L74
	.long	.L128-.L74
	.long	.L128-.L74
	.long	.L80-.L74
	.long	.L79-.L74
	.long	.L128-.L74
	.long	.L128-.L74
	.long	.L78-.L74
	.long	.L77-.L74
	.long	.L76-.L74
	.long	.L128-.L74
	.long	.L75-.L74
	.long	.L73-.L74
	.text
.L90:
	movl	-84(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	exit@PLT
.L84:
	movl	-100(%rbp), %edx
	movl	-104(%rbp), %eax
	cmpl	%eax, %edx
	jb	.L103
	movq	$18, -40(%rbp)
	jmp	.L105
.L103:
	movq	$17, -40(%rbp)
	jmp	.L105
.L99:
	movl	-80(%rbp), %eax
	leaq	0(,%rax,8), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	-76(%rbp), %edx
	movl	%edx, (%rax)
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$1, -40(%rbp)
	jmp	.L105
.L92:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	feof@PLT
	movl	%eax, -68(%rbp)
	movq	$42, -40(%rbp)
	jmp	.L105
.L94:
	movl	-84(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	exit@PLT
.L101:
	movq	-48(%rbp), %rax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L126
	jmp	.L127
.L85:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	exit@PLT
.L87:
	leaq	-92(%rbp), %rsi
	leaq	-96(%rbp), %rcx
	leaq	-100(%rbp), %rdx
	movq	-64(%rbp), %rax
	movq	%rsi, %r8
	leaq	.LC5(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movl	%eax, -72(%rbp)
	movq	$22, -40(%rbp)
	jmp	.L105
.L83:
	movl	-104(%rbp), %eax
	testl	%eax, %eax
	jne	.L107
	movq	$34, -40(%rbp)
	jmp	.L105
.L107:
	movq	$33, -40(%rbp)
	jmp	.L105
.L96:
	movl	-84(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	exit@PLT
.L93:
	movl	-100(%rbp), %eax
	cmpl	%eax, -80(%rbp)
	je	.L109
	movq	$39, -40(%rbp)
	jmp	.L105
.L109:
	movq	$7, -40(%rbp)
	jmp	.L105
.L89:
	addl	$1, -84(%rbp)
	movq	$13, -40(%rbp)
	jmp	.L105
.L91:
	movl	-96(%rbp), %edx
	movl	-104(%rbp), %eax
	cmpl	%eax, %edx
	jb	.L111
	movq	$29, -40(%rbp)
	jmp	.L105
.L111:
	movq	$10, -40(%rbp)
	jmp	.L105
.L98:
	cmpq	$0, -64(%rbp)
	jne	.L113
	movq	$24, -40(%rbp)
	jmp	.L105
.L113:
	movq	$2, -40(%rbp)
	jmp	.L105
.L77:
	movq	$28, -40(%rbp)
	jmp	.L105
.L79:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, %edi
	call	exit@PLT
.L86:
	cmpl	$3, -72(%rbp)
	jne	.L115
	movq	$19, -40(%rbp)
	jmp	.L105
.L115:
	movq	$15, -40(%rbp)
	jmp	.L105
.L82:
	movq	-120(%rbp), %rax
	leaq	.LC8(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -64(%rbp)
	movq	$6, -40(%rbp)
	jmp	.L105
.L80:
	movl	-104(%rbp), %eax
	imull	$100, %eax, %eax
	movl	%eax, %eax
	leaq	(%rax,%rax), %rdx
	movl	-104(%rbp), %eax
	movl	%eax, %eax
	addq	%rdx, %rax
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	-104(%rbp), %eax
	movl	%eax, %eax
	movl	$8, %esi
	movq	%rax, %rdi
	call	calloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$1, -84(%rbp)
	movl	$-1, -80(%rbp)
	movl	$0, -76(%rbp)
	movq	$21, -40(%rbp)
	jmp	.L105
.L78:
	movl	-80(%rbp), %eax
	leaq	0(,%rax,8), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	-76(%rbp), %edx
	movl	%edx, (%rax)
	movl	-76(%rbp), %eax
	salq	$3, %rax
	addq	$4, %rax
	addq	%rax, -56(%rbp)
	movq	$20, -40(%rbp)
	jmp	.L105
.L75:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, %edi
	call	exit@PLT
.L95:
	movl	-100(%rbp), %eax
	movl	%eax, %eax
	leaq	0(,%rax,8), %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	movq	%rax, (%rdx)
	movl	-100(%rbp), %eax
	movl	%eax, -80(%rbp)
	movl	$0, -76(%rbp)
	movq	$7, -40(%rbp)
	jmp	.L105
.L73:
	cmpl	$0, -68(%rbp)
	je	.L117
	movq	$4, -40(%rbp)
	jmp	.L105
.L117:
	movq	$9, -40(%rbp)
	jmp	.L105
.L102:
	cmpl	$1, -88(%rbp)
	je	.L119
	movq	$41, -40(%rbp)
	jmp	.L105
.L119:
	movq	$26, -40(%rbp)
	jmp	.L105
.L76:
	cmpl	$2, -84(%rbp)
	jbe	.L121
	movq	$37, -40(%rbp)
	jmp	.L105
.L121:
	movq	$10, -40(%rbp)
	jmp	.L105
.L97:
	movl	-100(%rbp), %eax
	movl	%eax, %eax
	leaq	0(,%rax,8), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	-96(%rbp), %ecx
	movl	-76(%rbp), %edx
	movl	%ecx, 4(%rax,%rdx,8)
	movl	-100(%rbp), %eax
	movl	%eax, %eax
	leaq	0(,%rax,8), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	-92(%rbp), %ecx
	movl	-76(%rbp), %edx
	movl	%ecx, 8(%rax,%rdx,8)
	addl	$1, -76(%rbp)
	movq	$21, -40(%rbp)
	jmp	.L105
.L81:
	movl	-84(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	exit@PLT
.L100:
	leaq	-104(%rbp), %rdx
	movq	-64(%rbp), %rax
	leaq	.LC9(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movl	%eax, -88(%rbp)
	movq	$0, -40(%rbp)
	jmp	.L105
.L88:
	movl	-100(%rbp), %eax
	cmpl	%eax, -80(%rbp)
	jbe	.L123
	movq	$12, -40(%rbp)
	jmp	.L105
.L123:
	movq	$25, -40(%rbp)
	jmp	.L105
.L128:
	nop
.L105:
	jmp	.L125
.L127:
	call	__stack_chk_fail@PLT
.L126:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	read_graph_file, .-read_graph_file
	.globl	max_distance_bitmap
	.type	max_distance_bitmap, @function
max_distance_bitmap:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movq	%rdi, -72(%rbp)
	movl	%esi, -76(%rbp)
	movq	%rdx, -88(%rbp)
	movq	$1, -16(%rbp)
.L151:
	cmpq	$15, -16(%rbp)
	ja	.L153
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L132(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L132(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L132:
	.long	.L142-.L132
	.long	.L141-.L132
	.long	.L140-.L132
	.long	.L153-.L132
	.long	.L139-.L132
	.long	.L138-.L132
	.long	.L137-.L132
	.long	.L136-.L132
	.long	.L135-.L132
	.long	.L153-.L132
	.long	.L153-.L132
	.long	.L153-.L132
	.long	.L134-.L132
	.long	.L153-.L132
	.long	.L133-.L132
	.long	.L131-.L132
	.text
.L139:
	movl	-76(%rbp), %eax
	leaq	0(,%rax,8), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -52(%rbp)
	movl	$0, -48(%rbp)
	movl	-76(%rbp), %edx
	movq	-88(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	bitmap_set_bit
	movl	$0, -44(%rbp)
	movq	$15, -16(%rbp)
	jmp	.L143
.L133:
	movq	-24(%rbp), %rax
	leaq	4(%rax), %rdx
	movl	-44(%rbp), %eax
	salq	$3, %rax
	addq	%rdx, %rax
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	4(%rax), %eax
	movl	%eax, -40(%rbp)
	movq	-8(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %edx
	movq	-88(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	bitmap_is_set
	movb	%al, -53(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L143
.L131:
	movl	-44(%rbp), %eax
	cmpl	-52(%rbp), %eax
	jnb	.L144
	movq	$14, -16(%rbp)
	jmp	.L143
.L144:
	movq	$12, -16(%rbp)
	jmp	.L143
.L134:
	movl	-76(%rbp), %edx
	movq	-88(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	bitmap_clear_bit
	movq	$5, -16(%rbp)
	jmp	.L143
.L135:
	movl	-40(%rbp), %eax
	cmpl	-48(%rbp), %eax
	jbe	.L146
	movq	$6, -16(%rbp)
	jmp	.L143
.L146:
	movq	$7, -16(%rbp)
	jmp	.L143
.L141:
	movq	$4, -16(%rbp)
	jmp	.L143
.L137:
	movl	-40(%rbp), %eax
	movl	%eax, -48(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L143
.L138:
	movl	-48(%rbp), %eax
	jmp	.L152
.L142:
	movq	-88(%rbp), %rdx
	movl	-36(%rbp), %ecx
	movq	-72(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	max_distance_bitmap
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	addl	%eax, -40(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L143
.L136:
	addl	$1, -44(%rbp)
	movq	$15, -16(%rbp)
	jmp	.L143
.L140:
	cmpb	$0, -53(%rbp)
	je	.L149
	movq	$7, -16(%rbp)
	jmp	.L143
.L149:
	movq	$0, -16(%rbp)
	jmp	.L143
.L153:
	nop
.L143:
	jmp	.L151
.L152:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	max_distance_bitmap, .-max_distance_bitmap
	.globl	max_distance_branchless
	.type	max_distance_branchless, @function
max_distance_branchless:
.LFB13:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1296, %rsp
	movq	%rdi, -1288(%rbp)
	movq	%rsi, -1296(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$16, -1232(%rbp)
.L181:
	cmpq	$20, -1232(%rbp)
	ja	.L184
	movq	-1232(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L157(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L157(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L157:
	.long	.L170-.L157
	.long	.L184-.L157
	.long	.L169-.L157
	.long	.L184-.L157
	.long	.L168-.L157
	.long	.L167-.L157
	.long	.L184-.L157
	.long	.L166-.L157
	.long	.L165-.L157
	.long	.L164-.L157
	.long	.L163-.L157
	.long	.L184-.L157
	.long	.L184-.L157
	.long	.L162-.L157
	.long	.L184-.L157
	.long	.L184-.L157
	.long	.L161-.L157
	.long	.L160-.L157
	.long	.L159-.L157
	.long	.L158-.L157
	.long	.L156-.L157
	.text
.L159:
	movq	-1240(%rbp), %rax
	movl	(%rax), %eax
	testl	%eax, %eax
	js	.L171
	movq	$17, -1232(%rbp)
	jmp	.L173
.L171:
	movq	$5, -1232(%rbp)
	jmp	.L173
.L168:
	movq	-1296(%rbp), %rax
	movl	(%rax), %eax
	andl	$2147483647, %eax
	movl	%eax, %edx
	movq	-1296(%rbp), %rax
	movl	%edx, (%rax)
	movq	$19, -1232(%rbp)
	jmp	.L173
.L165:
	movl	-1268(%rbp), %eax
	movq	-816(%rbp,%rax,8), %rax
	movq	%rax, -1224(%rbp)
	movq	-1224(%rbp), %rdx
	movq	-1288(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	max_distance_branchless
	movl	%eax, -1256(%rbp)
	movl	-1256(%rbp), %eax
	movl	%eax, -1252(%rbp)
	movl	-1268(%rbp), %eax
	movl	-1216(%rbp,%rax,4), %edx
	movl	-1252(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -1260(%rbp)
	movq	$10, -1232(%rbp)
	jmp	.L173
.L161:
	movq	$7, -1232(%rbp)
	jmp	.L173
.L164:
	movl	-1260(%rbp), %eax
	movl	%eax, -1264(%rbp)
	movq	$0, -1232(%rbp)
	jmp	.L173
.L162:
	cmpl	$0, -1272(%rbp)
	je	.L174
	movq	$2, -1232(%rbp)
	jmp	.L173
.L174:
	movq	$0, -1232(%rbp)
	jmp	.L173
.L158:
	movl	-1264(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L182
	jmp	.L183
.L160:
	addl	$1, -1268(%rbp)
	movq	$5, -1232(%rbp)
	jmp	.L173
.L167:
	addq	$8, -1248(%rbp)
	subl	$1, -1272(%rbp)
	movq	$13, -1232(%rbp)
	jmp	.L173
.L163:
	movl	-1260(%rbp), %eax
	cmpl	-1264(%rbp), %eax
	jbe	.L177
	movq	$9, -1232(%rbp)
	jmp	.L173
.L177:
	movq	$0, -1232(%rbp)
	jmp	.L173
.L170:
	subl	$1, -1268(%rbp)
	movq	$20, -1232(%rbp)
	jmp	.L173
.L166:
	movq	-1296(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, -1272(%rbp)
	movq	-1296(%rbp), %rax
	movl	(%rax), %eax
	orl	$-2147483648, %eax
	movl	%eax, %edx
	movq	-1296(%rbp), %rax
	movl	%edx, (%rax)
	movl	$1, -1268(%rbp)
	movq	-1296(%rbp), %rax
	addq	$4, %rax
	movq	%rax, -1248(%rbp)
	movl	$0, -1264(%rbp)
	movq	$2, -1232(%rbp)
	jmp	.L173
.L169:
	movq	-1248(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %eax
	leaq	0(,%rax,8), %rdx
	movq	-1288(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -1240(%rbp)
	movl	-1268(%rbp), %eax
	movq	-1240(%rbp), %rdx
	movq	%rdx, -816(%rbp,%rax,8)
	movq	-1248(%rbp), %rax
	movl	4(%rax), %edx
	movl	-1268(%rbp), %eax
	movl	%edx, -1216(%rbp,%rax,4)
	movq	$18, -1232(%rbp)
	jmp	.L173
.L156:
	cmpl	$0, -1268(%rbp)
	je	.L179
	movq	$8, -1232(%rbp)
	jmp	.L173
.L179:
	movq	$4, -1232(%rbp)
	jmp	.L173
.L184:
	nop
.L173:
	jmp	.L181
.L183:
	call	__stack_chk_fail@PLT
.L182:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE13:
	.size	max_distance_branchless, .-max_distance_branchless
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
