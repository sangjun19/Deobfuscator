	.file	"secy2520_Software-SystemLab_6_flatten.c"
	.text
	.globl	_TIG_IZ_11qt_argc
	.bss
	.align 4
	.type	_TIG_IZ_11qt_argc, @object
	.size	_TIG_IZ_11qt_argc, 4
_TIG_IZ_11qt_argc:
	.zero	4
	.globl	_TIG_IZ_11qt_argv
	.align 8
	.type	_TIG_IZ_11qt_argv, @object
	.size	_TIG_IZ_11qt_argv, 8
_TIG_IZ_11qt_argv:
	.zero	8
	.globl	_TIG_IZ_11qt_envp
	.align 8
	.type	_TIG_IZ_11qt_envp, @object
	.size	_TIG_IZ_11qt_envp, 8
_TIG_IZ_11qt_envp:
	.zero	8
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
	subq	$272, %rsp
	movl	%edi, -244(%rbp)
	movq	%rsi, -256(%rbp)
	movq	%rdx, -264(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_11qt_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_11qt_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_11qt_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 126 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-11qt--0
# 0 "" 2
#NO_APP
	movl	-244(%rbp), %eax
	movl	%eax, _TIG_IZ_11qt_argc(%rip)
	movq	-256(%rbp), %rax
	movq	%rax, _TIG_IZ_11qt_argv(%rip)
	movq	-264(%rbp), %rax
	movq	%rax, _TIG_IZ_11qt_envp(%rip)
	nop
	movq	$0, -224(%rbp)
.L17:
	cmpq	$4, -224(%rbp)
	ja	.L20
	movq	-224(%rbp), %rax
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
	movl	-228(%rbp), %eax
	movslq	%eax, %rdx
	leaq	-208(%rbp), %rax
	movq	%rax, %rsi
	movl	$1, %edi
	call	write@PLT
	movq	$1, -224(%rbp)
	jmp	.L13
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L18
	jmp	.L19
.L9:
	leaq	-208(%rbp), %rax
	movl	$200, %edx
	movq	%rax, %rsi
	movl	$0, %edi
	call	read@PLT
	movq	%rax, -216(%rbp)
	movq	-216(%rbp), %rax
	movl	%eax, -228(%rbp)
	movq	$2, -224(%rbp)
	jmp	.L13
.L12:
	movq	$3, -224(%rbp)
	jmp	.L13
.L10:
	cmpl	$1, -228(%rbp)
	jle	.L15
	movq	$4, -224(%rbp)
	jmp	.L13
.L15:
	movq	$1, -224(%rbp)
	jmp	.L13
.L20:
	nop
.L13:
	jmp	.L17
.L19:
	call	__stack_chk_fail@PLT
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
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
