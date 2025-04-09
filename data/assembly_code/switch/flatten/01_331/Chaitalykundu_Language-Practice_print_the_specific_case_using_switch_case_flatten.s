	.file	"Chaitalykundu_Language-Practice_print_the_specific_case_using_switch_case_flatten.c"
	.text
	.globl	_TIG_IZ_QX51_envp
	.bss
	.align 8
	.type	_TIG_IZ_QX51_envp, @object
	.size	_TIG_IZ_QX51_envp, 8
_TIG_IZ_QX51_envp:
	.zero	8
	.globl	_TIG_IZ_QX51_argv
	.align 8
	.type	_TIG_IZ_QX51_argv, @object
	.size	_TIG_IZ_QX51_argv, 8
_TIG_IZ_QX51_argv:
	.zero	8
	.globl	_TIG_IZ_QX51_argc
	.align 4
	.type	_TIG_IZ_QX51_argc, @object
	.size	_TIG_IZ_QX51_argc, 4
_TIG_IZ_QX51_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"\nI'm Case 1"
.LC1:
	.string	"\nI'm Case 3"
.LC2:
	.string	"\nI'm in default"
.LC3:
	.string	"\nI'm Case 2"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
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
	movq	$0, _TIG_IZ_QX51_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_QX51_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_QX51_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-QX51--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_QX51_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_QX51_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_QX51_envp(%rip)
	nop
	movq	$0, -8(%rbp)
.L22:
	cmpq	$8, -8(%rbp)
	ja	.L24
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
	.long	.L14-.L8
	.long	.L24-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L24-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L15
.L7:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L15
.L12:
	movl	$0, %eax
	jmp	.L23
.L10:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L15
.L14:
	movl	$2, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L15
.L9:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L15
.L13:
	cmpl	$3, -12(%rbp)
	je	.L17
	cmpl	$3, -12(%rbp)
	jg	.L18
	cmpl	$1, -12(%rbp)
	je	.L19
	cmpl	$2, -12(%rbp)
	je	.L20
	jmp	.L18
.L17:
	movq	$8, -8(%rbp)
	jmp	.L21
.L20:
	movq	$7, -8(%rbp)
	jmp	.L21
.L19:
	movq	$4, -8(%rbp)
	jmp	.L21
.L18:
	movq	$5, -8(%rbp)
	nop
.L21:
	jmp	.L15
.L24:
	nop
.L15:
	jmp	.L22
.L23:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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
