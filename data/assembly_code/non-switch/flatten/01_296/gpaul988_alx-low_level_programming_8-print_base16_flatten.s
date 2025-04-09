	.file	"gpaul988_alx-low_level_programming_8-print_base16_flatten.c"
	.text
	.globl	_TIG_IZ_iK0y_envp
	.bss
	.align 8
	.type	_TIG_IZ_iK0y_envp, @object
	.size	_TIG_IZ_iK0y_envp, 8
_TIG_IZ_iK0y_envp:
	.zero	8
	.globl	_TIG_IZ_iK0y_argc
	.align 4
	.type	_TIG_IZ_iK0y_argc, @object
	.size	_TIG_IZ_iK0y_argc, 4
_TIG_IZ_iK0y_argc:
	.zero	4
	.globl	_TIG_IZ_iK0y_argv
	.align 8
	.type	_TIG_IZ_iK0y_argv, @object
	.size	_TIG_IZ_iK0y_argv, 8
_TIG_IZ_iK0y_argv:
	.zero	8
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_iK0y_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_iK0y_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_iK0y_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 88 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-iK0y--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_iK0y_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_iK0y_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_iK0y_envp(%rip)
	nop
	movq	$8, -8(%rbp)
.L22:
	cmpq	$11, -8(%rbp)
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
	.long	.L24-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L24-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L24-.L8
	.long	.L24-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L13:
	movl	$0, %eax
	jmp	.L23
.L11:
	movl	$48, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L17
.L15:
	movl	$10, %edi
	call	putchar@PLT
	movq	$4, -8(%rbp)
	jmp	.L17
.L7:
	cmpl	$57, -12(%rbp)
	jg	.L18
	movq	$5, -8(%rbp)
	jmp	.L17
.L18:
	movq	$2, -8(%rbp)
	jmp	.L17
.L10:
	movl	-12(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$1, -12(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L17
.L12:
	movl	-12(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	addl	$1, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L17
.L9:
	cmpl	$102, -12(%rbp)
	jg	.L20
	movq	$9, -8(%rbp)
	jmp	.L17
.L20:
	movq	$1, -8(%rbp)
	jmp	.L17
.L14:
	movl	$97, -12(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L17
.L24:
	nop
.L17:
	jmp	.L22
.L23:
	leave
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
