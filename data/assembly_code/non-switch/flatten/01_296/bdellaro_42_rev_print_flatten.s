	.file	"bdellaro_42_rev_print_flatten.c"
	.text
	.globl	_TIG_IZ_Xqhp_argc
	.bss
	.align 4
	.type	_TIG_IZ_Xqhp_argc, @object
	.size	_TIG_IZ_Xqhp_argc, 4
_TIG_IZ_Xqhp_argc:
	.zero	4
	.globl	_TIG_IZ_Xqhp_envp
	.align 8
	.type	_TIG_IZ_Xqhp_envp, @object
	.size	_TIG_IZ_Xqhp_envp, 8
_TIG_IZ_Xqhp_envp:
	.zero	8
	.globl	_TIG_IZ_Xqhp_argv
	.align 8
	.type	_TIG_IZ_Xqhp_argv, @object
	.size	_TIG_IZ_Xqhp_argv, 8
_TIG_IZ_Xqhp_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"\n"
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_Xqhp_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Xqhp_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Xqhp_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 104 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Xqhp--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_Xqhp_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_Xqhp_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_Xqhp_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L14:
	cmpq	$3, -8(%rbp)
	je	.L6
	cmpq	$3, -8(%rbp)
	ja	.L16
	cmpq	$2, -8(%rbp)
	je	.L8
	cmpq	$2, -8(%rbp)
	ja	.L16
	cmpq	$0, -8(%rbp)
	je	.L9
	cmpq	$1, -8(%rbp)
	jne	.L16
	cmpl	$2, -20(%rbp)
	jne	.L10
	movq	$3, -8(%rbp)
	jmp	.L12
.L10:
	movq	$0, -8(%rbp)
	jmp	.L12
.L6:
	movq	-32(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	rev_print
	movq	$0, -8(%rbp)
	jmp	.L12
.L9:
	movl	$1, %edx
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	movl	$1, %edi
	call	write@PLT
	movq	$2, -8(%rbp)
	jmp	.L12
.L8:
	movl	$0, %eax
	jmp	.L15
.L16:
	nop
.L12:
	jmp	.L14
.L15:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	rev_print
	.type	rev_print, @function
rev_print:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$3, -16(%rbp)
.L33:
	cmpq	$10, -16(%rbp)
	ja	.L34
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L20(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L20(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L20:
	.long	.L34-.L20
	.long	.L26-.L20
	.long	.L25-.L20
	.long	.L24-.L20
	.long	.L34-.L20
	.long	.L23-.L20
	.long	.L34-.L20
	.long	.L22-.L20
	.long	.L34-.L20
	.long	.L21-.L20
	.long	.L35-.L20
	.text
.L26:
	movq	-24(%rbp), %rax
	movq	%rax, -8(%rbp)
	subq	$1, -24(%rbp)
	movq	-8(%rbp), %rax
	movl	$1, %edx
	movq	%rax, %rsi
	movl	$1, %edi
	call	write@PLT
	movq	$9, -16(%rbp)
	jmp	.L27
.L24:
	movq	$5, -16(%rbp)
	jmp	.L27
.L21:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L28
	movq	$1, -16(%rbp)
	jmp	.L27
.L28:
	movq	$10, -16(%rbp)
	jmp	.L27
.L23:
	movq	-24(%rbp), %rax
	movzbl	(%rax), %eax
	testb	%al, %al
	je	.L30
	movq	$2, -16(%rbp)
	jmp	.L27
.L30:
	movq	$7, -16(%rbp)
	jmp	.L27
.L22:
	subq	$1, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L27
.L25:
	addq	$1, -24(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L27
.L34:
	nop
.L27:
	jmp	.L33
.L35:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	rev_print, .-rev_print
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
