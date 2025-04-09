	.file	"mohamed-elatar_Master-Embedded-System_main_flatten.c"
	.text
	.globl	_TIG_IZ_QSka_envp
	.bss
	.align 8
	.type	_TIG_IZ_QSka_envp, @object
	.size	_TIG_IZ_QSka_envp, 8
_TIG_IZ_QSka_envp:
	.zero	8
	.globl	_TIG_IZ_QSka_argv
	.align 8
	.type	_TIG_IZ_QSka_argv, @object
	.size	_TIG_IZ_QSka_argv, 8
_TIG_IZ_QSka_argv:
	.zero	8
	.globl	_TIG_IZ_QSka_argc
	.align 4
	.type	_TIG_IZ_QSka_argc, @object
	.size	_TIG_IZ_QSka_argc, 4
_TIG_IZ_QSka_argc:
	.zero	4
	.text
	.globl	ascll_to_decimal
	.type	ascll_to_decimal, @function
ascll_to_decimal:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -40(%rbp)
	movq	$10, -8(%rbp)
.L23:
	cmpq	$13, -8(%rbp)
	ja	.L25
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L25-.L4
	.long	.L25-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L25-.L4
	.long	.L25-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L25-.L4
	.long	.L3-.L4
	.text
.L10:
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$45, %al
	jne	.L13
	movq	$13, -8(%rbp)
	jmp	.L15
.L13:
	movq	$2, -8(%rbp)
	jmp	.L15
.L11:
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$57, %al
	jg	.L16
	movq	$5, -8(%rbp)
	jmp	.L15
.L16:
	movq	$6, -8(%rbp)
	jmp	.L15
.L5:
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	cmpb	$47, %al
	jle	.L18
	movq	$3, -8(%rbp)
	jmp	.L15
.L18:
	movq	$6, -8(%rbp)
	jmp	.L15
.L7:
	movl	$0, -24(%rbp)
	movl	$1, -20(%rbp)
	movl	$0, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L15
.L3:
	movl	$-1, -20(%rbp)
	addl	$1, -24(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L15
.L8:
	movl	-20(%rbp), %eax
	imull	-16(%rbp), %eax
	jmp	.L24
.L9:
	movl	-24(%rbp), %eax
	movl	%eax, -12(%rbp)
	addl	$1, -24(%rbp)
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	leal	-48(%rax), %ecx
	movl	-16(%rbp), %edx
	movl	%edx, %eax
	sall	$2, %eax
	addl	%edx, %eax
	addl	%eax, %eax
	addl	%ecx, %eax
	movl	%eax, -16(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L15
.L6:
	movq	$9, -8(%rbp)
	jmp	.L15
.L12:
	movl	-24(%rbp), %eax
	movslq	%eax, %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	subl	$49, %eax
	cmpl	$8, %eax
	ja	.L21
	movq	$11, -8(%rbp)
	jmp	.L22
.L21:
	movq	$6, -8(%rbp)
	nop
.L22:
	jmp	.L15
.L25:
	nop
.L15:
	jmp	.L23
.L24:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	ascll_to_decimal, .-ascll_to_decimal
	.section	.rodata
.LC0:
	.string	"decimal number is :%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
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
	movq	$0, _TIG_IZ_QSka_envp(%rip)
	nop
.L27:
	movq	$0, _TIG_IZ_QSka_argv(%rip)
	nop
.L28:
	movl	$0, _TIG_IZ_QSka_argc(%rip)
	nop
	nop
.L29:
.L30:
#APP
# 106 "mohamed-elatar_Master-Embedded-System_main.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-QSka--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_QSka_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_QSka_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_QSka_envp(%rip)
	nop
	movq	$0, -24(%rbp)
.L36:
	cmpq	$2, -24(%rbp)
	je	.L39
	cmpq	$2, -24(%rbp)
	ja	.L40
	cmpq	$0, -24(%rbp)
	je	.L33
	cmpq	$1, -24(%rbp)
	jne	.L40
	movb	$49, -13(%rbp)
	movb	$50, -12(%rbp)
	movb	$51, -11(%rbp)
	movb	$52, -10(%rbp)
	movb	$0, -9(%rbp)
	leaq	-13(%rbp), %rax
	movq	%rax, %rdi
	call	ascll_to_decimal
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -24(%rbp)
	jmp	.L34
.L33:
	movq	$1, -24(%rbp)
	jmp	.L34
.L40:
	nop
.L34:
	jmp	.L36
.L39:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L38
	call	__stack_chk_fail@PLT
.L38:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
