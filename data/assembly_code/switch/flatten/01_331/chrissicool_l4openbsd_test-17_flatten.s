	.file	"chrissicool_l4openbsd_test-17_flatten.c"
	.text
	.globl	_TIG_IZ_qsDO_argc
	.bss
	.align 4
	.type	_TIG_IZ_qsDO_argc, @object
	.size	_TIG_IZ_qsDO_argc, 4
_TIG_IZ_qsDO_argc:
	.zero	4
	.globl	_TIG_IZ_qsDO_envp
	.align 8
	.type	_TIG_IZ_qsDO_envp, @object
	.size	_TIG_IZ_qsDO_envp, 8
_TIG_IZ_qsDO_envp:
	.zero	8
	.globl	_TIG_IZ_qsDO_argv
	.align 8
	.type	_TIG_IZ_qsDO_argv, @object
	.size	_TIG_IZ_qsDO_argv, 8
_TIG_IZ_qsDO_argv:
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
	movq	$0, _TIG_IZ_qsDO_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_qsDO_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_qsDO_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 105 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-qsDO--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_qsDO_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_qsDO_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_qsDO_envp(%rip)
	nop
	movq	$6, -8(%rbp)
.L37:
	cmpq	$25, -8(%rbp)
	ja	.L38
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
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L38-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L38-.L8
	.long	.L17-.L8
	.long	.L38-.L8
	.long	.L16-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L38-.L8
	.long	.L38-.L8
	.long	.L13-.L8
	.long	.L38-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L38-.L8
	.long	.L9-.L8
	.long	.L38-.L8
	.long	.L7-.L8
	.text
.L7:
	movq	$0, -8(%rbp)
	jmp	.L22
.L18:
	movl	-12(%rbp), %eax
	jmp	.L23
.L14:
	addl	$1, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L22
.L16:
	movl	$1, %eax
	jmp	.L23
.L20:
	movl	$1, -12(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L22
.L9:
	cmpl	$0, -12(%rbp)
	jne	.L24
	movq	$0, -8(%rbp)
	jmp	.L25
.L24:
	movq	$25, -8(%rbp)
	nop
.L25:
	jmp	.L22
.L19:
	movl	$1, %eax
	jmp	.L23
.L10:
	cmpl	$0, -20(%rbp)
	je	.L26
	movq	$3, -8(%rbp)
	jmp	.L22
.L26:
	movq	$14, -8(%rbp)
	jmp	.L22
.L15:
	cmpl	$3, -20(%rbp)
	je	.L28
	cmpl	$3, -20(%rbp)
	jg	.L29
	cmpl	$1, -20(%rbp)
	je	.L30
	cmpl	$2, -20(%rbp)
	je	.L31
	jmp	.L29
.L28:
	movq	$23, -8(%rbp)
	jmp	.L32
.L31:
	movq	$19, -8(%rbp)
	jmp	.L32
.L30:
	movq	$1, -8(%rbp)
	jmp	.L32
.L29:
	movq	$0, -8(%rbp)
	nop
.L32:
	jmp	.L22
.L12:
	movl	$2, -12(%rbp)
	movq	$17, -8(%rbp)
	jmp	.L22
.L13:
	cmpl	$4, -12(%rbp)
	jg	.L33
	movq	$20, -8(%rbp)
	jmp	.L22
.L33:
	movq	$0, -8(%rbp)
	jmp	.L22
.L17:
	movl	$0, -12(%rbp)
	movq	$13, -8(%rbp)
	jmp	.L22
.L21:
	cmpl	$4, -20(%rbp)
	jg	.L35
	movq	$21, -8(%rbp)
	jmp	.L22
.L35:
	movq	$4, -8(%rbp)
	jmp	.L22
.L11:
	addl	$1, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L22
.L38:
	nop
.L22:
	jmp	.L37
.L23:
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
