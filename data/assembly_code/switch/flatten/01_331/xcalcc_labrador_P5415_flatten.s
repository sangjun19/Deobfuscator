	.file	"xcalcc_labrador_P5415_flatten.c"
	.text
	.globl	_TIG_IZ_zoqi_argc
	.bss
	.align 4
	.type	_TIG_IZ_zoqi_argc, @object
	.size	_TIG_IZ_zoqi_argc, 4
_TIG_IZ_zoqi_argc:
	.zero	4
	.globl	_TIG_IZ_zoqi_argv
	.align 8
	.type	_TIG_IZ_zoqi_argv, @object
	.size	_TIG_IZ_zoqi_argv, 8
_TIG_IZ_zoqi_argv:
	.zero	8
	.globl	_TIG_IZ_zoqi_envp
	.align 8
	.type	_TIG_IZ_zoqi_envp, @object
	.size	_TIG_IZ_zoqi_envp, 8
_TIG_IZ_zoqi_envp:
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
	movq	$0, _TIG_IZ_zoqi_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_zoqi_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_zoqi_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-zoqi--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_zoqi_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_zoqi_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_zoqi_envp(%rip)
	nop
	movq	$3, -8(%rbp)
.L12:
	cmpq	$3, -8(%rbp)
	je	.L6
	cmpq	$3, -8(%rbp)
	ja	.L14
	cmpq	$0, -8(%rbp)
	je	.L8
	cmpq	$2, -8(%rbp)
	je	.L9
	jmp	.L14
.L6:
	movl	$2, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L10
.L8:
	movq	$2, -8(%rbp)
	nop
	jmp	.L10
.L9:
	movl	-12(%rbp), %eax
	jmp	.L13
.L14:
	nop
.L10:
	jmp	.L12
.L13:
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
