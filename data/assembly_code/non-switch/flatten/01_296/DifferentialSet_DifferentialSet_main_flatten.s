	.file	"DifferentialSet_DifferentialSet_main_flatten.c"
	.text
	.globl	_TIG_IZ_Or72_envp
	.bss
	.align 8
	.type	_TIG_IZ_Or72_envp, @object
	.size	_TIG_IZ_Or72_envp, 8
_TIG_IZ_Or72_envp:
	.zero	8
	.globl	_TIG_IZ_Or72_argv
	.align 8
	.type	_TIG_IZ_Or72_argv, @object
	.size	_TIG_IZ_Or72_argv, 8
_TIG_IZ_Or72_argv:
	.zero	8
	.globl	_TIG_IZ_Or72_argc
	.align 4
	.type	_TIG_IZ_Or72_argc, @object
	.size	_TIG_IZ_Or72_argc, 4
_TIG_IZ_Or72_argc:
	.zero	4
	.globl	y
	.align 4
	.type	y, @object
	.size	y, 4
y:
	.zero	4
	.globl	x
	.align 4
	.type	x, @object
	.size	x, 4
x:
	.zero	4
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
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
	movl	$0, y(%rip)
	nop
.L2:
	movl	$0, x(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_Or72_envp(%rip)
	nop
.L4:
	movq	$0, _TIG_IZ_Or72_argv(%rip)
	nop
.L5:
	movl	$0, _TIG_IZ_Or72_argc(%rip)
	nop
	nop
.L6:
.L7:
#APP
# 113 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Or72--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_Or72_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_Or72_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_Or72_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L15:
	cmpq	$2, -8(%rbp)
	je	.L8
	cmpq	$2, -8(%rbp)
	ja	.L17
	cmpq	$0, -8(%rbp)
	je	.L10
	cmpq	$1, -8(%rbp)
	jne	.L17
	movl	x(%rip), %eax
	testl	%eax, %eax
	je	.L11
	movq	$2, -8(%rbp)
	jmp	.L13
.L11:
	movq	$0, -8(%rbp)
	jmp	.L13
.L10:
	movl	$0, %eax
	jmp	.L16
.L8:
	movl	$1, y(%rip)
	movq	$0, -8(%rbp)
	jmp	.L13
.L17:
	nop
.L13:
	jmp	.L15
.L16:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
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
