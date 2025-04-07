	.file	"gaurshivangi_k-core_kcore_flatten.c"
	.text
	.globl	_TIG_IZ_tBbc_argv
	.bss
	.align 8
	.type	_TIG_IZ_tBbc_argv, @object
	.size	_TIG_IZ_tBbc_argv, 8
_TIG_IZ_tBbc_argv:
	.zero	8
	.globl	_TIG_IZ_tBbc_envp
	.align 8
	.type	_TIG_IZ_tBbc_envp, @object
	.size	_TIG_IZ_tBbc_envp, 8
_TIG_IZ_tBbc_envp:
	.zero	8
	.globl	_TIG_IZ_tBbc_argc
	.align 4
	.type	_TIG_IZ_tBbc_argc, @object
	.size	_TIG_IZ_tBbc_argc, 4
_TIG_IZ_tBbc_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"enter core size: "
.LC1:
	.string	"%d"
.LC2:
	.string	"r"
.LC3:
	.string	"kcoregraph"
.LC4:
	.string	"%d "
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movl	%edi, -132(%rbp)
	movq	%rsi, -144(%rbp)
	movq	%rdx, -152(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_tBbc_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_tBbc_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_tBbc_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 121 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-tBbc--0
# 0 "" 2
#NO_APP
	movl	-132(%rbp), %eax
	movl	%eax, _TIG_IZ_tBbc_argc(%rip)
	movq	-144(%rbp), %rax
	movq	%rax, _TIG_IZ_tBbc_argv(%rip)
	movq	-152(%rbp), %rax
	movq	%rax, _TIG_IZ_tBbc_envp(%rip)
	nop
	movq	$28, -40(%rbp)
.L63:
	cmpq	$46, -40(%rbp)
	ja	.L66
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
	.long	.L38-.L8
	.long	.L37-.L8
	.long	.L36-.L8
	.long	.L35-.L8
	.long	.L34-.L8
	.long	.L66-.L8
	.long	.L66-.L8
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L31-.L8
	.long	.L66-.L8
	.long	.L30-.L8
	.long	.L66-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L66-.L8
	.long	.L66-.L8
	.long	.L26-.L8
	.long	.L66-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L66-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L66-.L8
	.long	.L66-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L66-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L66-.L8
	.long	.L16-.L8
	.long	.L66-.L8
	.long	.L15-.L8
	.long	.L66-.L8
	.long	.L14-.L8
	.long	.L66-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L66-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L26:
	movl	-108(%rbp), %eax
	cmpl	%eax, -84(%rbp)
	jge	.L39
	movq	$42, -40(%rbp)
	jmp	.L41
.L39:
	movq	$21, -40(%rbp)
	jmp	.L41
.L34:
	movl	-116(%rbp), %eax
	cmpl	%eax, -100(%rbp)
	jge	.L42
	movq	$46, -40(%rbp)
	jmp	.L41
.L42:
	movq	$24, -40(%rbp)
	jmp	.L41
.L19:
	movl	-112(%rbp), %eax
	cmpl	%eax, -76(%rbp)
	jge	.L44
	movq	$15, -40(%rbp)
	jmp	.L41
.L44:
	movq	$40, -40(%rbp)
	jmp	.L41
.L28:
	movl	-92(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$-1, %eax
	je	.L46
	movq	$0, -40(%rbp)
	jmp	.L41
.L46:
	movq	$2, -40(%rbp)
	jmp	.L41
.L27:
	movl	-76(%rbp), %edx
	movq	-56(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	degree
	movl	%eax, -72(%rbp)
	movq	$3, -40(%rbp)
	jmp	.L41
.L18:
	movl	-92(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -88(%rbp)
	jge	.L48
	movq	$27, -40(%rbp)
	jmp	.L41
.L48:
	movq	$41, -40(%rbp)
	jmp	.L41
.L32:
	movl	-88(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -68(%rbp)
	movl	-88(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	$-1, (%rax)
	movl	-92(%rbp), %r8d
	leaq	-112(%rbp), %rdi
	leaq	-116(%rbp), %rcx
	movl	-68(%rbp), %edx
	movq	-48(%rbp), %rsi
	movq	-56(%rbp), %rax
	movl	%r8d, %r9d
	movq	%rdi, %r8
	movq	%rax, %rdi
	call	remove_node
	movl	-68(%rbp), %edx
	movq	-56(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	degree
	movl	%eax, -84(%rbp)
	movq	$18, -40(%rbp)
	jmp	.L41
.L9:
	movl	$0, -76(%rbp)
	movq	$30, -40(%rbp)
	jmp	.L41
.L37:
	movl	$0, -100(%rbp)
	movq	$4, -40(%rbp)
	jmp	.L41
.L23:
	cmpl	$0, -96(%rbp)
	je	.L50
	movq	$11, -40(%rbp)
	jmp	.L41
.L50:
	movq	$45, -40(%rbp)
	jmp	.L41
.L35:
	movl	-108(%rbp), %eax
	cmpl	%eax, -72(%rbp)
	jl	.L52
	movq	$20, -40(%rbp)
	jmp	.L41
.L52:
	movq	$36, -40(%rbp)
	jmp	.L41
.L22:
	movl	-112(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movl	-116(%rbp), %eax
	movl	%eax, (%rdx)
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-108(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$1, -96(%rbp)
	movq	$23, -40(%rbp)
	jmp	.L41
.L24:
	addl	$1, -88(%rbp)
	movq	$31, -40(%rbp)
	jmp	.L41
.L15:
	addl	$1, -76(%rbp)
	movq	$30, -40(%rbp)
	jmp	.L41
.L30:
	movl	$0, -96(%rbp)
	movl	$0, -92(%rbp)
	movq	$43, -40(%rbp)
	jmp	.L41
.L31:
	movl	-108(%rbp), %eax
	cmpl	%eax, -80(%rbp)
	jge	.L54
	movq	$14, -40(%rbp)
	jmp	.L41
.L54:
	movq	$32, -40(%rbp)
	jmp	.L41
.L29:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -64(%rbp)
	leaq	-112(%rbp), %rdx
	movq	-64(%rbp), %rax
	leaq	.LC1(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	leaq	-116(%rbp), %rdx
	movq	-64(%rbp), %rax
	leaq	.LC1(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movl	-116(%rbp), %eax
	addl	%eax, %eax
	movl	%eax, -116(%rbp)
	movl	-112(%rbp), %eax
	addl	$2, %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	-116(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -48(%rbp)
	movl	$0, -104(%rbp)
	movq	$34, -40(%rbp)
	jmp	.L41
.L17:
	movl	$0, -96(%rbp)
	movq	$41, -40(%rbp)
	jmp	.L41
.L13:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L64
	jmp	.L65
.L21:
	movl	-88(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$-1, %eax
	jne	.L57
	movq	$41, -40(%rbp)
	jmp	.L41
.L57:
	movq	$8, -40(%rbp)
	jmp	.L41
.L14:
	movl	-104(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	leaq	.LC1(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	addl	$1, -104(%rbp)
	movq	$34, -40(%rbp)
	jmp	.L41
.L16:
	movl	-112(%rbp), %eax
	cmpl	%eax, -104(%rbp)
	jge	.L59
	movq	$38, -40(%rbp)
	jmp	.L41
.L59:
	movq	$1, -40(%rbp)
	jmp	.L41
.L20:
	movq	$13, -40(%rbp)
	jmp	.L41
.L12:
	addl	$1, -92(%rbp)
	movq	$43, -40(%rbp)
	jmp	.L41
.L11:
	movl	$1, -96(%rbp)
	movq	$21, -40(%rbp)
	jmp	.L41
.L38:
	movl	$1, -96(%rbp)
	movl	-92(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -88(%rbp)
	movq	$31, -40(%rbp)
	jmp	.L41
.L7:
	movl	-100(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rax, %rdx
	movq	-64(%rbp), %rax
	leaq	.LC1(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	addl	$1, -100(%rbp)
	movq	$4, -40(%rbp)
	jmp	.L41
.L33:
	movl	-92(%rbp), %edx
	movq	-56(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	degree
	movl	%eax, -80(%rbp)
	movq	$9, -40(%rbp)
	jmp	.L41
.L10:
	movl	-112(%rbp), %eax
	cmpl	%eax, -92(%rbp)
	jge	.L61
	movq	$7, -40(%rbp)
	jmp	.L41
.L61:
	movq	$23, -40(%rbp)
	jmp	.L41
.L36:
	movl	$0, -96(%rbp)
	movq	$41, -40(%rbp)
	jmp	.L41
.L25:
	movl	-76(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$36, -40(%rbp)
	jmp	.L41
.L66:
	nop
.L41:
	jmp	.L63
.L65:
	call	__stack_chk_fail@PLT
.L64:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.globl	remove_node
	.type	remove_node, @function
remove_node:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movl	%edx, -52(%rbp)
	movq	%rcx, -64(%rbp)
	movq	%r8, -72(%rbp)
	movl	%r9d, -56(%rbp)
	movq	$6, -8(%rbp)
.L95:
	cmpq	$21, -8(%rbp)
	ja	.L96
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L70(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L70(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L70:
	.long	.L97-.L70
	.long	.L96-.L70
	.long	.L81-.L70
	.long	.L80-.L70
	.long	.L79-.L70
	.long	.L96-.L70
	.long	.L78-.L70
	.long	.L77-.L70
	.long	.L96-.L70
	.long	.L76-.L70
	.long	.L75-.L70
	.long	.L74-.L70
	.long	.L96-.L70
	.long	.L73-.L70
	.long	.L96-.L70
	.long	.L72-.L70
	.long	.L96-.L70
	.long	.L96-.L70
	.long	.L96-.L70
	.long	.L96-.L70
	.long	.L71-.L70
	.long	.L69-.L70
	.text
.L79:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -56(%rbp)
	je	.L83
	movq	$7, -8(%rbp)
	jmp	.L85
.L83:
	movq	$9, -8(%rbp)
	jmp	.L85
.L72:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %edx
	subl	$1, %edx
	movl	%edx, (%rax)
	addl	$1, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L85
.L80:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -16(%rbp)
	jge	.L86
	movq	$20, -8(%rbp)
	jmp	.L85
.L86:
	movq	$11, -8(%rbp)
	jmp	.L85
.L69:
	movl	-20(%rbp), %eax
	movl	%eax, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L85
.L74:
	movl	-52(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L85
.L76:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L88
	movq	$21, -8(%rbp)
	jmp	.L85
.L88:
	movq	$0, -8(%rbp)
	jmp	.L85
.L73:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L90
	movq	$4, -8(%rbp)
	jmp	.L85
.L90:
	movq	$9, -8(%rbp)
	jmp	.L85
.L78:
	movl	-52(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -20(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L85
.L75:
	movq	-64(%rbp), %rax
	movl	(%rax), %eax
	leal	-1(%rax), %edx
	movq	-64(%rbp), %rax
	movl	%edx, (%rax)
	movq	$0, -8(%rbp)
	jmp	.L85
.L77:
	addl	$1, -20(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L85
.L81:
	movq	-72(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L93
	movq	$15, -8(%rbp)
	jmp	.L85
.L93:
	movq	$10, -8(%rbp)
	jmp	.L85
.L71:
	movl	-16(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	-16(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-48(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L85
.L96:
	nop
.L85:
	jmp	.L95
.L97:
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	remove_node, .-remove_node
	.globl	degree
	.type	degree, @function
degree:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$1, -8(%rbp)
.L103:
	cmpq	$0, -8(%rbp)
	je	.L99
	cmpq	$1, -8(%rbp)
	jne	.L105
	movl	-28(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	-28(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-24(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rdx), %edx
	subl	%edx, %eax
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L101
.L99:
	movl	-12(%rbp), %eax
	jmp	.L104
.L105:
	nop
.L101:
	jmp	.L103
.L104:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	degree, .-degree
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
