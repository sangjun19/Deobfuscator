	.file	"Zylve_c-projects_Main_flatten.c"
	.text
	.globl	_TIG_IZ_wQik_argv
	.bss
	.align 8
	.type	_TIG_IZ_wQik_argv, @object
	.size	_TIG_IZ_wQik_argv, 8
_TIG_IZ_wQik_argv:
	.zero	8
	.globl	_TIG_IZ_wQik_envp
	.align 8
	.type	_TIG_IZ_wQik_envp, @object
	.size	_TIG_IZ_wQik_envp, 8
_TIG_IZ_wQik_envp:
	.zero	8
	.globl	_TIG_IZ_wQik_argc
	.align 4
	.type	_TIG_IZ_wQik_argc, @object
	.size	_TIG_IZ_wQik_argc, 4
_TIG_IZ_wQik_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
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
	movq	$0, _TIG_IZ_wQik_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_wQik_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_wQik_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 106 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-wQik--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_wQik_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_wQik_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_wQik_envp(%rip)
	nop
	movq	$0, -88(%rbp)
.L17:
	cmpq	$7, -88(%rbp)
	ja	.L20
	movq	-88(%rbp), %rax
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
	.long	.L12-.L8
	.long	.L20-.L8
	.long	.L20-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L20-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	-92(%rbp), %eax
	cltq
	movl	-80(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -92(%rbp)
	movq	$3, -88(%rbp)
	jmp	.L13
.L11:
	movl	-92(%rbp), %eax
	cmpl	$17, %eax
	ja	.L14
	movq	$4, -88(%rbp)
	jmp	.L13
.L14:
	movq	$7, -88(%rbp)
	jmp	.L13
.L9:
	movl	$4, -80(%rbp)
	movl	$6, -76(%rbp)
	movl	$1, -72(%rbp)
	movl	$542, -68(%rbp)
	movl	$1, -64(%rbp)
	movl	$7, -60(%rbp)
	movl	$3, -56(%rbp)
	movl	$1, -52(%rbp)
	movl	$7, -48(%rbp)
	movl	$9, -44(%rbp)
	movl	$6, -40(%rbp)
	movl	$2, -36(%rbp)
	movl	$1, -32(%rbp)
	movl	$6, -28(%rbp)
	movl	$32, -24(%rbp)
	movl	$12, -20(%rbp)
	movl	$7, -16(%rbp)
	movl	$34, -12(%rbp)
	leaq	-80(%rbp), %rax
	movl	$18, %esi
	movq	%rax, %rdi
	call	MergeSort
	movl	$0, -92(%rbp)
	movq	$3, -88(%rbp)
	jmp	.L13
.L12:
	movq	$6, -88(%rbp)
	jmp	.L13
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L18
	jmp	.L19
.L20:
	nop
.L13:
	jmp	.L17
.L19:
	call	__stack_chk_fail@PLT
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	Merge
	.type	Merge, @function
Merge:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%rcx, -64(%rbp)
	movq	%r8, -72(%rbp)
	movq	$20, -8(%rbp)
.L49:
	cmpq	$22, -8(%rbp)
	ja	.L50
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L24(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L24(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L24:
	.long	.L36-.L24
	.long	.L50-.L24
	.long	.L51-.L24
	.long	.L50-.L24
	.long	.L34-.L24
	.long	.L33-.L24
	.long	.L50-.L24
	.long	.L32-.L24
	.long	.L31-.L24
	.long	.L50-.L24
	.long	.L50-.L24
	.long	.L30-.L24
	.long	.L29-.L24
	.long	.L50-.L24
	.long	.L50-.L24
	.long	.L50-.L24
	.long	.L28-.L24
	.long	.L50-.L24
	.long	.L27-.L24
	.long	.L50-.L24
	.long	.L26-.L24
	.long	.L25-.L24
	.long	.L23-.L24
	.text
.L27:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-48(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	cmpl	%eax, %edx
	jle	.L37
	movq	$16, -8(%rbp)
	jmp	.L39
.L37:
	movq	$7, -8(%rbp)
	jmp	.L39
.L34:
	movl	-16(%rbp), %eax
	cltq
	cmpq	%rax, -64(%rbp)
	jbe	.L40
	movq	$8, -8(%rbp)
	jmp	.L39
.L40:
	movq	$0, -8(%rbp)
	jmp	.L39
.L29:
	movl	$0, -20(%rbp)
	movl	$0, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L39
.L31:
	movl	-12(%rbp), %eax
	cltq
	cmpq	%rax, -72(%rbp)
	jbe	.L42
	movq	$18, -8(%rbp)
	jmp	.L39
.L42:
	movq	$0, -8(%rbp)
	jmp	.L39
.L28:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	-20(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L39
.L25:
	movl	-16(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	-20(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -16(%rbp)
	addl	$1, -20(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L39
.L30:
	movl	-12(%rbp), %eax
	cltq
	cmpq	%rax, -72(%rbp)
	jbe	.L44
	movq	$22, -8(%rbp)
	jmp	.L39
.L44:
	movq	$2, -8(%rbp)
	jmp	.L39
.L23:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	-20(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -12(%rbp)
	addl	$1, -20(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L39
.L33:
	addl	$1, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L39
.L36:
	movl	-16(%rbp), %eax
	cltq
	cmpq	%rax, -64(%rbp)
	jbe	.L46
	movq	$21, -8(%rbp)
	jmp	.L39
.L46:
	movq	$11, -8(%rbp)
	jmp	.L39
.L32:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	-20(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-56(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L39
.L26:
	movq	$12, -8(%rbp)
	jmp	.L39
.L50:
	nop
.L39:
	jmp	.L49
.L51:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	Merge, .-Merge
	.globl	MergeSort
	.type	MergeSort, @function
MergeSort:
.LFB2:
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
	subq	$144, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	movq	%rdi, -136(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -40(%rbp)
	xorl	%eax, %eax
	movq	$13, -80(%rbp)
.L82:
	cmpq	$18, -80(%rbp)
	ja	.L84
	movq	-80(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L55(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L55(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L55:
	.long	.L84-.L55
	.long	.L67-.L55
	.long	.L66-.L55
	.long	.L65-.L55
	.long	.L64-.L55
	.long	.L84-.L55
	.long	.L63-.L55
	.long	.L62-.L55
	.long	.L61-.L55
	.long	.L85-.L55
	.long	.L84-.L55
	.long	.L59-.L55
	.long	.L58-.L55
	.long	.L57-.L55
	.long	.L56-.L55
	.long	.L84-.L55
	.long	.L84-.L55
	.long	.L84-.L55
	.long	.L85-.L55
	.text
.L64:
	movl	-124(%rbp), %eax
	movslq	%eax, %rdx
	movq	-144(%rbp), %rax
	subq	%rdx, %rax
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -88(%rbp)
	movq	-88(%rbp), %rax
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
.L69:
	cmpq	%rdx, %rsp
	je	.L70
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L69
.L70:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L71
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L71:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -104(%rbp)
	movq	$6, -80(%rbp)
	jmp	.L72
.L56:
	movq	-96(%rbp), %rax
	shrq	$2, %rax
	movq	%rax, %rdx
	movl	-124(%rbp), %eax
	movslq	%eax, %rcx
	subq	$1, %rcx
	movq	%rcx, -72(%rbp)
	cltq
	movq	%rax, %r12
	movl	$0, %r13d
	movq	-112(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	MergeSort
	movq	-88(%rbp), %rax
	shrq	$2, %rax
	movq	%rax, %rdx
	movl	-124(%rbp), %eax
	movslq	%eax, %rcx
	movq	-144(%rbp), %rax
	subq	%rcx, %rax
	movq	%rax, %rcx
	subq	$1, %rcx
	movq	%rcx, -64(%rbp)
	movq	%rax, %r14
	movl	$0, %r15d
	movq	-104(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	MergeSort
	movq	-88(%rbp), %rax
	shrq	$2, %rax
	movq	%rax, %rdi
	movq	-96(%rbp), %rax
	shrq	$2, %rax
	movq	%rax, %rcx
	movl	-124(%rbp), %eax
	movslq	%eax, %rdx
	movq	-144(%rbp), %rax
	subq	%rdx, %rax
	movq	%rax, %rdx
	subq	$1, %rdx
	movq	%rdx, -56(%rbp)
	movq	%rax, -160(%rbp)
	movq	$0, -152(%rbp)
	movl	-124(%rbp), %eax
	movslq	%eax, %rdx
	subq	$1, %rdx
	movq	%rdx, -48(%rbp)
	cltq
	movq	%rax, -176(%rbp)
	movq	$0, -168(%rbp)
	movq	-136(%rbp), %rdx
	movq	-104(%rbp), %rsi
	movq	-112(%rbp), %rax
	movq	%rdi, %r8
	movq	%rax, %rdi
	call	Merge
	movq	$9, -80(%rbp)
	jmp	.L72
.L58:
	movl	-124(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -96(%rbp)
	movq	-96(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %edi
	movl	$0, %edx
	divq	%rdi
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L73:
	cmpq	%rdx, %rsp
	je	.L74
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L73
.L74:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L75
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L75:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -112(%rbp)
	movq	$4, -80(%rbp)
	jmp	.L72
.L61:
	movl	-116(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	movl	-116(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-112(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	movq	$3, -80(%rbp)
	jmp	.L72
.L67:
	movl	-116(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-136(%rbp), %rax
	addq	%rdx, %rax
	movl	-120(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-104(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -120(%rbp)
	movq	$3, -80(%rbp)
	jmp	.L72
.L65:
	addl	$1, -116(%rbp)
	movq	$11, -80(%rbp)
	jmp	.L72
.L59:
	movl	-116(%rbp), %eax
	cltq
	cmpq	%rax, -144(%rbp)
	jbe	.L76
	movq	$2, -80(%rbp)
	jmp	.L72
.L76:
	movq	$14, -80(%rbp)
	jmp	.L72
.L57:
	cmpq	$1, -144(%rbp)
	ja	.L78
	movq	$18, -80(%rbp)
	jmp	.L72
.L78:
	movq	$7, -80(%rbp)
	jmp	.L72
.L63:
	movl	$0, -120(%rbp)
	movl	$0, -116(%rbp)
	movq	$11, -80(%rbp)
	jmp	.L72
.L62:
	movq	-144(%rbp), %rax
	shrq	%rax
	movl	%eax, -124(%rbp)
	movq	$12, -80(%rbp)
	jmp	.L72
.L66:
	movl	-116(%rbp), %eax
	cmpl	-124(%rbp), %eax
	jge	.L80
	movq	$8, -80(%rbp)
	jmp	.L72
.L80:
	movq	$1, -80(%rbp)
	jmp	.L72
.L84:
	nop
.L72:
	jmp	.L82
.L85:
	nop
	movq	-40(%rbp), %rax
	subq	%fs:40, %rax
	je	.L83
	call	__stack_chk_fail@PLT
.L83:
	leaq	-32(%rbp), %rsp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	MergeSort, .-MergeSort
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
