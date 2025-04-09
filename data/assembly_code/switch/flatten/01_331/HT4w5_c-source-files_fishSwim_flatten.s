	.file	"HT4w5_c-source-files_fishSwim_flatten.c"
	.text
	.globl	_TIG_IZ_wNb1_argc
	.bss
	.align 4
	.type	_TIG_IZ_wNb1_argc, @object
	.size	_TIG_IZ_wNb1_argc, 4
_TIG_IZ_wNb1_argc:
	.zero	4
	.globl	_TIG_IZ_wNb1_envp
	.align 8
	.type	_TIG_IZ_wNb1_envp, @object
	.size	_TIG_IZ_wNb1_envp, 8
_TIG_IZ_wNb1_envp:
	.zero	8
	.globl	_TIG_IZ_wNb1_argv
	.align 8
	.type	_TIG_IZ_wNb1_argv, @object
	.size	_TIG_IZ_wNb1_argv, 8
_TIG_IZ_wNb1_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"%d%d"
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
	movq	$0, _TIG_IZ_wNb1_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_wNb1_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_wNb1_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 124 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-wNb1--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_wNb1_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_wNb1_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_wNb1_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L27:
	cmpq	$13, -16(%rbp)
	ja	.L30
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
	.long	.L30-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L30-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L30-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L30-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L9:
	addl	$1, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L18
.L17:
	movl	-32(%rbp), %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	cmpl	$7, %eax
	je	.L19
	cmpl	$7, %eax
	jg	.L20
	cmpl	$5, %eax
	jg	.L21
	testl	%eax, %eax
	jg	.L22
	jmp	.L20
.L21:
	cmpl	$6, %eax
	jne	.L20
	movq	$12, -16(%rbp)
	jmp	.L23
.L19:
	movq	$2, -16(%rbp)
	jmp	.L23
.L22:
	movq	$5, -16(%rbp)
	jmp	.L23
.L20:
	movq	$6, -16(%rbp)
	nop
.L23:
	jmp	.L18
.L15:
	movl	-20(%rbp), %eax
	cmpl	-24(%rbp), %eax
	jg	.L24
	movq	$1, -16(%rbp)
	jmp	.L18
.L24:
	movq	$9, -16(%rbp)
	jmp	.L18
.L11:
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -16(%rbp)
	jmp	.L18
.L7:
	leaq	-28(%rbp), %rdx
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-28(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-1840700269, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$2, %edx
	movl	%eax, %ecx
	sarl	$31, %ecx
	subl	%ecx, %edx
	movl	%edx, -24(%rbp)
	movl	-24(%rbp), %ecx
	movl	%ecx, %edx
	sall	$3, %edx
	subl	%ecx, %edx
	subl	%edx, %eax
	movl	%eax, -24(%rbp)
	movl	-28(%rbp), %eax
	movslq	%eax, %rdx
	imulq	$-1840700269, %rdx, %rdx
	shrq	$32, %rdx
	addl	%eax, %edx
	sarl	$2, %edx
	sarl	$31, %eax
	subl	%eax, %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	movl	%eax, -28(%rbp)
	movl	$0, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L18
.L13:
	movq	$12, -16(%rbp)
	jmp	.L18
.L14:
	movl	-28(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -28(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L18
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L28
	jmp	.L29
.L12:
	movq	$13, -16(%rbp)
	jmp	.L18
.L16:
	movl	-32(%rbp), %eax
	subl	$7, %eax
	movl	%eax, -32(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L18
.L30:
	nop
.L18:
	jmp	.L27
.L29:
	call	__stack_chk_fail@PLT
.L28:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
