	.file	"pjcali_codeProjects_3_flatten.c"
	.text
	.globl	_TIG_IZ_FwmR_envp
	.bss
	.align 8
	.type	_TIG_IZ_FwmR_envp, @object
	.size	_TIG_IZ_FwmR_envp, 8
_TIG_IZ_FwmR_envp:
	.zero	8
	.globl	_TIG_IZ_FwmR_argc
	.align 4
	.type	_TIG_IZ_FwmR_argc, @object
	.size	_TIG_IZ_FwmR_argc, 4
_TIG_IZ_FwmR_argc:
	.zero	4
	.globl	_TIG_IZ_FwmR_argv
	.align 8
	.type	_TIG_IZ_FwmR_argv, @object
	.size	_TIG_IZ_FwmR_argv, 8
_TIG_IZ_FwmR_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"\n\nDONE.\n"
.LC1:
	.string	"the original array:"
.LC2:
	.string	"[%d] "
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
	subq	$144, %rsp
	movl	%edi, -116(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_FwmR_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_FwmR_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_FwmR_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 148 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-FwmR--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_FwmR_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_FwmR_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_FwmR_envp(%rip)
	nop
	movq	$1, -88(%rbp)
.L18:
	cmpq	$9, -88(%rbp)
	ja	.L21
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
	.long	.L21-.L8
	.long	.L13-.L8
	.long	.L21-.L8
	.long	.L21-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L21-.L8
	.long	.L7-.L8
	.text
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L19
	jmp	.L20
.L13:
	movq	$5, -88(%rbp)
	jmp	.L15
.L7:
	movl	-104(%rbp), %edx
	leaq	-80(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	reduce
	movl	%eax, -96(%rbp)
	movl	-96(%rbp), %eax
	movl	%eax, -92(%rbp)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -88(%rbp)
	jmp	.L15
.L10:
	movl	-100(%rbp), %eax
	cmpl	-104(%rbp), %eax
	jge	.L16
	movq	$7, -88(%rbp)
	jmp	.L15
.L16:
	movq	$9, -88(%rbp)
	jmp	.L15
.L11:
	movl	$16, -104(%rbp)
	movl	$9, -80(%rbp)
	movl	$1, -76(%rbp)
	movl	$1, -72(%rbp)
	movl	$6, -68(%rbp)
	movl	$7, -64(%rbp)
	movl	$1, -60(%rbp)
	movl	$2, -56(%rbp)
	movl	$3, -52(%rbp)
	movl	$3, -48(%rbp)
	movl	$5, -44(%rbp)
	movl	$6, -40(%rbp)
	movl	$6, -36(%rbp)
	movl	$6, -32(%rbp)
	movl	$6, -28(%rbp)
	movl	$7, -24(%rbp)
	movl	$9, -20(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -100(%rbp)
	movq	$6, -88(%rbp)
	jmp	.L15
.L9:
	movl	-100(%rbp), %eax
	cltq
	movl	-80(%rbp,%rax,4), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -100(%rbp)
	movq	$6, -88(%rbp)
	jmp	.L15
.L21:
	nop
.L15:
	jmp	.L18
.L20:
	call	__stack_chk_fail@PLT
.L19:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC3:
	.string	"\ntop 3 values: %d, %d, %d\n"
.LC4:
	.string	"\nthe resulting array is..."
	.text
	.globl	reduce
	.type	reduce, @function
reduce:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r13
	pushq	%r12
	subq	$96, %rsp
	.cfi_offset 13, -24
	.cfi_offset 12, -32
	movq	%rdi, -104(%rbp)
	movl	%esi, -108(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$4, -56(%rbp)
.L94:
	cmpq	$47, -56(%rbp)
	ja	.L97
	movq	-56(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L25(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L25(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L25:
	.long	.L56-.L25
	.long	.L55-.L25
	.long	.L97-.L25
	.long	.L97-.L25
	.long	.L54-.L25
	.long	.L53-.L25
	.long	.L52-.L25
	.long	.L51-.L25
	.long	.L97-.L25
	.long	.L97-.L25
	.long	.L50-.L25
	.long	.L49-.L25
	.long	.L48-.L25
	.long	.L97-.L25
	.long	.L47-.L25
	.long	.L46-.L25
	.long	.L97-.L25
	.long	.L97-.L25
	.long	.L45-.L25
	.long	.L44-.L25
	.long	.L43-.L25
	.long	.L42-.L25
	.long	.L41-.L25
	.long	.L97-.L25
	.long	.L40-.L25
	.long	.L39-.L25
	.long	.L38-.L25
	.long	.L37-.L25
	.long	.L36-.L25
	.long	.L35-.L25
	.long	.L34-.L25
	.long	.L33-.L25
	.long	.L32-.L25
	.long	.L31-.L25
	.long	.L97-.L25
	.long	.L30-.L25
	.long	.L97-.L25
	.long	.L97-.L25
	.long	.L97-.L25
	.long	.L97-.L25
	.long	.L97-.L25
	.long	.L29-.L25
	.long	.L28-.L25
	.long	.L27-.L25
	.long	.L26-.L25
	.long	.L97-.L25
	.long	.L97-.L25
	.long	.L24-.L25
	.text
.L45:
	movl	-76(%rbp), %eax
	movl	%eax, -72(%rbp)
	movl	-80(%rbp), %eax
	movl	%eax, -76(%rbp)
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -80(%rbp)
	movq	$47, -56(%rbp)
	jmp	.L57
.L39:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -80(%rbp)
	je	.L58
	movq	$22, -56(%rbp)
	jmp	.L57
.L58:
	movq	$14, -56(%rbp)
	jmp	.L57
.L54:
	movq	$32, -56(%rbp)
	jmp	.L57
.L34:
	movl	$0, -68(%rbp)
	movl	$0, -84(%rbp)
	movq	$42, -56(%rbp)
	jmp	.L57
.L47:
	addl	$1, -84(%rbp)
	movq	$29, -56(%rbp)
	jmp	.L57
.L46:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	-68(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-64(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -68(%rbp)
	movq	$33, -56(%rbp)
	jmp	.L57
.L33:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -72(%rbp)
	jge	.L60
	movq	$5, -56(%rbp)
	jmp	.L57
.L60:
	movq	$47, -56(%rbp)
	jmp	.L57
.L48:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -72(%rbp)
	je	.L62
	movq	$15, -56(%rbp)
	jmp	.L57
.L62:
	movq	$33, -56(%rbp)
	jmp	.L57
.L55:
	movl	-88(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
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
.L64:
	cmpq	%rdx, %rsp
	je	.L65
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L64
.L65:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L66
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L66:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -64(%rbp)
	movq	$30, -56(%rbp)
	jmp	.L57
.L40:
	movl	-88(%rbp), %eax
	cltq
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -104(%rbp)
	movl	-88(%rbp), %eax
	movslq	%eax, %rdx
	subq	$1, %rdx
	movq	%rdx, -40(%rbp)
	cltq
	movq	%rax, %r12
	movl	$0, %r13d
	movq	-64(%rbp), %rax
	movq	%rax, -104(%rbp)
	movl	-72(%rbp), %ecx
	movl	-76(%rbp), %edx
	movl	-80(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -84(%rbp)
	movq	$21, -56(%rbp)
	jmp	.L57
.L42:
	movl	-84(%rbp), %eax
	cmpl	-88(%rbp), %eax
	jge	.L67
	movq	$28, -56(%rbp)
	jmp	.L57
.L67:
	movq	$20, -56(%rbp)
	jmp	.L57
.L38:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -76(%rbp)
	je	.L69
	movq	$12, -56(%rbp)
	jmp	.L57
.L69:
	movq	$33, -56(%rbp)
	jmp	.L57
.L49:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -72(%rbp)
	movq	$47, -56(%rbp)
	jmp	.L57
.L44:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -80(%rbp)
	jge	.L71
	movq	$18, -56(%rbp)
	jmp	.L57
.L71:
	movq	$44, -56(%rbp)
	jmp	.L57
.L32:
	movl	$0, -88(%rbp)
	movl	$0, -80(%rbp)
	movl	$0, -76(%rbp)
	movl	$0, -72(%rbp)
	movl	$0, -84(%rbp)
	movq	$43, -56(%rbp)
	jmp	.L57
.L52:
	addl	$1, -88(%rbp)
	movq	$14, -56(%rbp)
	jmp	.L57
.L37:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -80(%rbp)
	je	.L73
	movq	$26, -56(%rbp)
	jmp	.L57
.L73:
	movq	$33, -56(%rbp)
	jmp	.L57
.L41:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -76(%rbp)
	je	.L75
	movq	$10, -56(%rbp)
	jmp	.L57
.L75:
	movq	$14, -56(%rbp)
	jmp	.L57
.L36:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -84(%rbp)
	movq	$21, -56(%rbp)
	jmp	.L57
.L24:
	addl	$1, -84(%rbp)
	movq	$43, -56(%rbp)
	jmp	.L57
.L26:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -76(%rbp)
	jge	.L77
	movq	$35, -56(%rbp)
	jmp	.L57
.L77:
	movq	$31, -56(%rbp)
	jmp	.L57
.L53:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -76(%rbp)
	je	.L79
	movq	$7, -56(%rbp)
	jmp	.L57
.L79:
	movq	$47, -56(%rbp)
	jmp	.L57
.L31:
	addl	$1, -84(%rbp)
	movq	$42, -56(%rbp)
	jmp	.L57
.L29:
	movl	-76(%rbp), %eax
	movl	%eax, -72(%rbp)
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -76(%rbp)
	movq	$47, -56(%rbp)
	jmp	.L57
.L50:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -72(%rbp)
	je	.L81
	movq	$6, -56(%rbp)
	jmp	.L57
.L81:
	movq	$14, -56(%rbp)
	jmp	.L57
.L28:
	movl	-84(%rbp), %eax
	cmpl	-108(%rbp), %eax
	jge	.L83
	movq	$27, -56(%rbp)
	jmp	.L57
.L83:
	movq	$24, -56(%rbp)
	jmp	.L57
.L56:
	movl	$0, -84(%rbp)
	movq	$29, -56(%rbp)
	jmp	.L57
.L51:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -80(%rbp)
	je	.L85
	movq	$11, -56(%rbp)
	jmp	.L57
.L85:
	movq	$47, -56(%rbp)
	jmp	.L57
.L30:
	movl	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-104(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -80(%rbp)
	je	.L87
	movq	$41, -56(%rbp)
	jmp	.L57
.L87:
	movq	$31, -56(%rbp)
	jmp	.L57
.L35:
	movl	-84(%rbp), %eax
	cmpl	-108(%rbp), %eax
	jge	.L89
	movq	$25, -56(%rbp)
	jmp	.L57
.L89:
	movq	$1, -56(%rbp)
	jmp	.L57
.L27:
	movl	-84(%rbp), %eax
	cmpl	-108(%rbp), %eax
	jge	.L91
	movq	$19, -56(%rbp)
	jmp	.L57
.L91:
	movq	$0, -56(%rbp)
	jmp	.L57
.L43:
	movl	-88(%rbp), %eax
	movq	-24(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L95
	jmp	.L96
.L97:
	nop
.L57:
	jmp	.L94
.L96:
	call	__stack_chk_fail@PLT
.L95:
	leaq	-16(%rbp), %rsp
	popq	%r12
	popq	%r13
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	reduce, .-reduce
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
