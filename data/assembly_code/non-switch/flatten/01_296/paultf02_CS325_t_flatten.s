	.file	"paultf02_CS325_t_flatten.c"
	.text
	.globl	_TIG_IZ_bE37_argv
	.bss
	.align 8
	.type	_TIG_IZ_bE37_argv, @object
	.size	_TIG_IZ_bE37_argv, 8
_TIG_IZ_bE37_argv:
	.zero	8
	.globl	_TIG_IZ_bE37_argc
	.align 4
	.type	_TIG_IZ_bE37_argc, @object
	.size	_TIG_IZ_bE37_argc, 4
_TIG_IZ_bE37_argc:
	.zero	4
	.globl	a
	.align 4
	.type	a, @object
	.size	a, 4
a:
	.zero	4
	.globl	_TIG_IZ_bE37_envp
	.align 8
	.type	_TIG_IZ_bE37_envp, @object
	.size	_TIG_IZ_bE37_envp, 8
_TIG_IZ_bE37_envp:
	.zero	8
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
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movl	$0, a(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_bE37_envp(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_bE37_argv(%rip)
	nop
.L4:
	movl	$0, _TIG_IZ_bE37_argc(%rip)
	nop
	nop
.L5:
.L6:
#APP
# 133 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-bE37--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_bE37_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_bE37_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_bE37_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L11:
	cmpq	$0, -8(%rbp)
	je	.L7
	cmpq	$1, -8(%rbp)
	jne	.L13
	movl	a(%rip), %eax
	addl	$1, %eax
	movl	%eax, a(%rip)
	movq	$0, -8(%rbp)
	jmp	.L9
.L7:
	movl	a(%rip), %eax
	jmp	.L12
.L13:
	nop
.L9:
	jmp	.L11
.L12:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
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
