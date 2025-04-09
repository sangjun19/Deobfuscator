	.file	"Apoorva0915_problem-Sol_ls_bs_flatten.c"
	.text
	.globl	d
	.bss
	.align 4
	.type	d, @object
	.size	d, 4
d:
	.zero	4
	.globl	_TIG_IZ_93av_argc
	.align 4
	.type	_TIG_IZ_93av_argc, @object
	.size	_TIG_IZ_93av_argc, 4
_TIG_IZ_93av_argc:
	.zero	4
	.globl	_TIG_IZ_93av_envp
	.align 8
	.type	_TIG_IZ_93av_envp, @object
	.size	_TIG_IZ_93av_envp, 8
_TIG_IZ_93av_envp:
	.zero	8
	.globl	_TIG_IZ_93av_argv
	.align 8
	.type	_TIG_IZ_93av_argv, @object
	.size	_TIG_IZ_93av_argv, 8
_TIG_IZ_93av_argv:
	.zero	8
	.globl	c
	.align 4
	.type	c, @object
	.size	c, 4
c:
	.zero	4
	.text
	.globl	linearSearch
	.type	linearSearch, @function
linearSearch:
.LFB0:
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
	movq	$3, -8(%rbp)
.L16:
	cmpq	$7, -8(%rbp)
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
	.long	.L9-.L4
	.long	.L18-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L18-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L6:
	addl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L10
.L7:
	movl	$0, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L10
.L5:
	movl	c(%rip), %eax
	addl	$1, %eax
	movl	%eax, c(%rip)
	movq	$2, -8(%rbp)
	jmp	.L10
.L9:
	movl	c(%rip), %eax
	jmp	.L17
.L3:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jg	.L12
	movq	$6, -8(%rbp)
	jmp	.L10
.L12:
	movq	$0, -8(%rbp)
	jmp	.L10
.L8:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -32(%rbp)
	jne	.L14
	movq	$0, -8(%rbp)
	jmp	.L10
.L14:
	movq	$4, -8(%rbp)
	jmp	.L10
.L18:
	nop
.L10:
	jmp	.L16
.L17:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	linearSearch, .-linearSearch
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
	movq	$3, -8(%rbp)
.L39:
	cmpq	$11, -8(%rbp)
	ja	.L41
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L22(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L22(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L22:
	.long	.L41-.L22
	.long	.L30-.L22
	.long	.L29-.L22
	.long	.L28-.L22
	.long	.L27-.L22
	.long	.L26-.L22
	.long	.L25-.L22
	.long	.L24-.L22
	.long	.L41-.L22
	.long	.L23-.L22
	.long	.L41-.L22
	.long	.L21-.L22
	.text
.L27:
	movl	-20(%rbp), %eax
	cmpl	-16(%rbp), %eax
	jg	.L31
	movq	$11, -8(%rbp)
	jmp	.L33
.L31:
	movq	$2, -8(%rbp)
	jmp	.L33
.L30:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -48(%rbp)
	jge	.L34
	movq	$6, -8(%rbp)
	jmp	.L33
.L34:
	movq	$9, -8(%rbp)
	jmp	.L33
.L28:
	movq	$7, -8(%rbp)
	jmp	.L33
.L21:
	movl	-20(%rbp), %edx
	movl	-16(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, -12(%rbp)
	movl	d(%rip), %eax
	addl	$1, %eax
	movl	%eax, d(%rip)
	movq	$5, -8(%rbp)
	jmp	.L33
.L23:
	movl	-12(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L33
.L25:
	movl	-12(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L33
.L26:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -48(%rbp)
	jne	.L36
	movq	$2, -8(%rbp)
	jmp	.L33
.L36:
	movq	$1, -8(%rbp)
	jmp	.L33
.L24:
	movl	$0, -20(%rbp)
	movl	-44(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L33
.L29:
	movl	d(%rip), %eax
	jmp	.L40
.L41:
	nop
.L33:
	jmp	.L39
.L40:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	binarySearch, .-binarySearch
	.section	.rodata
.LC0:
	.string	"%d %d"
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, d(%rip)
	nop
.L43:
	movl	$0, c(%rip)
	nop
.L44:
	movq	$0, _TIG_IZ_93av_envp(%rip)
	nop
.L45:
	movq	$0, _TIG_IZ_93av_argv(%rip)
	nop
.L46:
	movl	$0, _TIG_IZ_93av_argc(%rip)
	nop
	nop
.L47:
.L48:
#APP
# 105 "Apoorva0915_problem-Sol_ls_bs.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-93av--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_93av_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_93av_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_93av_envp(%rip)
	nop
	movq	$1, -56(%rbp)
.L54:
	cmpq	$2, -56(%rbp)
	je	.L49
	cmpq	$2, -56(%rbp)
	ja	.L57
	cmpq	$0, -56(%rbp)
	je	.L51
	cmpq	$1, -56(%rbp)
	jne	.L57
	movq	$2, -56(%rbp)
	jmp	.L52
.L51:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L55
	jmp	.L56
.L49:
	movl	$1, -48(%rbp)
	movl	$3, -44(%rbp)
	movl	$5, -40(%rbp)
	movl	$7, -36(%rbp)
	movl	$9, -32(%rbp)
	movl	$11, -28(%rbp)
	movl	$13, -24(%rbp)
	movl	$15, -20(%rbp)
	movl	$17, -16(%rbp)
	movl	$19, -12(%rbp)
	leaq	-48(%rbp), %rax
	movl	$13, %edx
	movl	$10, %esi
	movq	%rax, %rdi
	call	binarySearch
	movl	%eax, -64(%rbp)
	leaq	-48(%rbp), %rax
	movl	$13, %edx
	movl	$10, %esi
	movq	%rax, %rdi
	call	linearSearch
	movl	%eax, -60(%rbp)
	movl	-64(%rbp), %edx
	movl	-60(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -56(%rbp)
	jmp	.L52
.L57:
	nop
.L52:
	jmp	.L54
.L56:
	call	__stack_chk_fail@PLT
.L55:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
