	.file	"Rusewww_Programing_main1_flatten.c"
	.text
	.globl	_TIG_IZ_A8Jh_envp
	.bss
	.align 8
	.type	_TIG_IZ_A8Jh_envp, @object
	.size	_TIG_IZ_A8Jh_envp, 8
_TIG_IZ_A8Jh_envp:
	.zero	8
	.globl	_TIG_IZ_A8Jh_argv
	.align 8
	.type	_TIG_IZ_A8Jh_argv, @object
	.size	_TIG_IZ_A8Jh_argv, 8
_TIG_IZ_A8Jh_argv:
	.zero	8
	.globl	_TIG_IZ_A8Jh_argc
	.align 4
	.type	_TIG_IZ_A8Jh_argc, @object
	.size	_TIG_IZ_A8Jh_argc, 4
_TIG_IZ_A8Jh_argc:
	.zero	4
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
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_A8Jh_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_A8Jh_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_A8Jh_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-A8Jh--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_A8Jh_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_A8Jh_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_A8Jh_envp(%rip)
	nop
	movq	$2, -8(%rbp)
.L11:
	cmpq	$2, -8(%rbp)
	je	.L6
	cmpq	$2, -8(%rbp)
	ja	.L13
	cmpq	$0, -8(%rbp)
	je	.L8
	cmpq	$1, -8(%rbp)
	jne	.L13
	movss	.LC0(%rip), %xmm0
	movss	%xmm0, -32(%rbp)
	movss	.LC0(%rip), %xmm0
	movss	%xmm0, -28(%rbp)
	movss	.LC0(%rip), %xmm0
	movss	%xmm0, -24(%rbp)
	movss	-32(%rbp), %xmm0
	mulss	-28(%rbp), %xmm0
	movss	-24(%rbp), %xmm1
	mulss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	movss	-28(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-24(%rbp), %xmm1
	movss	-32(%rbp), %xmm0
	mulss	-24(%rbp), %xmm0
	addss	%xmm0, %xmm1
	movss	-32(%rbp), %xmm0
	mulss	-28(%rbp), %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -16(%rbp)
	movss	-20(%rbp), %xmm0
	divss	-16(%rbp), %xmm0
	movss	%xmm0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L9
.L8:
	movl	$0, %eax
	jmp	.L12
.L6:
	movq	$1, -8(%rbp)
	jmp	.L9
.L13:
	nop
.L9:
	jmp	.L11
.L12:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC0:
	.long	1056964608
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
