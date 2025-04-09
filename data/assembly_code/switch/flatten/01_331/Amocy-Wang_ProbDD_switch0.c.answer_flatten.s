	.file	"Amocy-Wang_ProbDD_switch0.c.answer_flatten.c"
	.text
	.globl	_TIG_IZ_AaPt_envp
	.bss
	.align 8
	.type	_TIG_IZ_AaPt_envp, @object
	.size	_TIG_IZ_AaPt_envp, 8
_TIG_IZ_AaPt_envp:
	.zero	8
	.globl	_TIG_IZ_AaPt_argc
	.align 4
	.type	_TIG_IZ_AaPt_argc, @object
	.size	_TIG_IZ_AaPt_argc, 4
_TIG_IZ_AaPt_argc:
	.zero	4
	.globl	_TIG_IZ_AaPt_argv
	.align 8
	.type	_TIG_IZ_AaPt_argv, @object
	.size	_TIG_IZ_AaPt_argv, 8
_TIG_IZ_AaPt_argv:
	.zero	8
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
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_AaPt_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_AaPt_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_AaPt_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 106 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-AaPt--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_AaPt_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_AaPt_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_AaPt_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L18:
	cmpq	$6, -8(%rbp)
	ja	.L20
	movq	-8(%rbp), %rax
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
	.long	.L20-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	$0, -16(%rbp)
	movl	$3, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L14
.L13:
	movq	$4, -8(%rbp)
	jmp	.L14
.L11:
	movq	$6, -8(%rbp)
	jmp	.L14
.L7:
	movl	-16(%rbp), %edx
	movl	-12(%rbp), %eax
	addl	%edx, %eax
	jmp	.L19
.L9:
	movl	$0, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L14
.L12:
	cmpl	$3, -12(%rbp)
	jne	.L16
	movq	$5, -8(%rbp)
	jmp	.L17
.L16:
	movq	$3, -8(%rbp)
	nop
.L17:
	jmp	.L14
.L20:
	nop
.L14:
	jmp	.L18
.L19:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
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
