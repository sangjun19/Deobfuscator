	.file	"ForNeVeR_Cesium_switch_with_fallthrough_variable_flatten.c"
	.text
	.globl	_TIG_IZ_5aoN_envp
	.bss
	.align 8
	.type	_TIG_IZ_5aoN_envp, @object
	.size	_TIG_IZ_5aoN_envp, 8
_TIG_IZ_5aoN_envp:
	.zero	8
	.globl	_TIG_IZ_5aoN_argv
	.align 8
	.type	_TIG_IZ_5aoN_argv, @object
	.size	_TIG_IZ_5aoN_argv, 8
_TIG_IZ_5aoN_argv:
	.zero	8
	.globl	_TIG_IZ_5aoN_argc
	.align 4
	.type	_TIG_IZ_5aoN_argc, @object
	.size	_TIG_IZ_5aoN_argc, 4
_TIG_IZ_5aoN_argc:
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
	movq	$0, _TIG_IZ_5aoN_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_5aoN_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_5aoN_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-5aoN--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_5aoN_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_5aoN_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_5aoN_envp(%rip)
	nop
	movq	$3, -8(%rbp)
.L17:
	cmpq	$4, -8(%rbp)
	ja	.L19
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
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	cmpl	$1, -20(%rbp)
	jne	.L13
	movq	$2, -8(%rbp)
	jmp	.L14
.L13:
	movq	$1, -8(%rbp)
	nop
.L14:
	jmp	.L15
.L11:
	movl	-12(%rbp), %eax
	movl	%eax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L15
.L9:
	movl	$1, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L15
.L12:
	movl	-16(%rbp), %eax
	jmp	.L18
.L10:
	movl	$42, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L15
.L19:
	nop
.L15:
	jmp	.L17
.L18:
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
