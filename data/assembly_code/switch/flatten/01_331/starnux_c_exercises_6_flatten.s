	.file	"starnux_c_exercises_6_flatten.c"
	.text
	.globl	_TIG_IZ_dKVy_envp
	.bss
	.align 8
	.type	_TIG_IZ_dKVy_envp, @object
	.size	_TIG_IZ_dKVy_envp, 8
_TIG_IZ_dKVy_envp:
	.zero	8
	.globl	_TIG_IZ_dKVy_argv
	.align 8
	.type	_TIG_IZ_dKVy_argv, @object
	.size	_TIG_IZ_dKVy_argv, 8
_TIG_IZ_dKVy_argv:
	.zero	8
	.globl	_TIG_IZ_dKVy_argc
	.align 4
	.type	_TIG_IZ_dKVy_argc, @object
	.size	_TIG_IZ_dKVy_argc, 4
_TIG_IZ_dKVy_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"ONE"
.LC1:
	.string	"TWO"
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
	movq	$0, _TIG_IZ_dKVy_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_dKVy_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_dKVy_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-dKVy--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_dKVy_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_dKVy_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_dKVy_envp(%rip)
	nop
	movq	$4, -8(%rbp)
.L19:
	cmpq	$8, -8(%rbp)
	ja	.L21
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
	.long	.L13-.L8
	.long	.L21-.L8
	.long	.L12-.L8
	.long	.L21-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L21-.L8
	.long	.L7-.L8
	.text
.L11:
	movl	$10, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L14
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L14
.L9:
	cmpl	$49, -12(%rbp)
	je	.L15
	cmpl	$50, -12(%rbp)
	jne	.L16
	movq	$2, -8(%rbp)
	jmp	.L17
.L15:
	movq	$8, -8(%rbp)
	jmp	.L17
.L16:
	movq	$5, -8(%rbp)
	nop
.L17:
	jmp	.L14
.L10:
	movq	$0, -8(%rbp)
	jmp	.L14
.L13:
	movl	$0, %eax
	jmp	.L20
.L12:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L14
.L21:
	nop
.L14:
	jmp	.L19
.L20:
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
