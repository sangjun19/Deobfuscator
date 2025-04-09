	.file	"bishoykadees_Eclipse_ATMEGA32A_Projects_main_flatten.c"
	.text
	.globl	_TIG_IZ_yIHq_argc
	.bss
	.align 4
	.type	_TIG_IZ_yIHq_argc, @object
	.size	_TIG_IZ_yIHq_argc, 4
_TIG_IZ_yIHq_argc:
	.zero	4
	.globl	_TIG_IZ_yIHq_argv
	.align 8
	.type	_TIG_IZ_yIHq_argv, @object
	.size	_TIG_IZ_yIHq_argv, 8
_TIG_IZ_yIHq_argv:
	.zero	8
	.globl	_TIG_IZ_yIHq_envp
	.align 8
	.type	_TIG_IZ_yIHq_envp, @object
	.size	_TIG_IZ_yIHq_envp, 8
_TIG_IZ_yIHq_envp:
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
	movq	$0, _TIG_IZ_yIHq_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_yIHq_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_yIHq_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 104 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-yIHq--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_yIHq_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_yIHq_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_yIHq_envp(%rip)
	nop
	movq	$0, -8(%rbp)
.L8:
	cmpq	$0, -8(%rbp)
	je	.L11
	nop
	jmp	.L8
.L11:
	nop
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
