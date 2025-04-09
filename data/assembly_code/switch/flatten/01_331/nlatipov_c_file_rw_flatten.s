	.file	"nlatipov_c_file_rw_flatten.c"
	.text
	.globl	_TIG_IZ_Sxkg_envp
	.bss
	.align 8
	.type	_TIG_IZ_Sxkg_envp, @object
	.size	_TIG_IZ_Sxkg_envp, 8
_TIG_IZ_Sxkg_envp:
	.zero	8
	.globl	_TIG_IZ_Sxkg_argv
	.align 8
	.type	_TIG_IZ_Sxkg_argv, @object
	.size	_TIG_IZ_Sxkg_argv, 8
_TIG_IZ_Sxkg_argv:
	.zero	8
	.globl	_TIG_IZ_Sxkg_argc
	.align 4
	.type	_TIG_IZ_Sxkg_argc, @object
	.size	_TIG_IZ_Sxkg_argc, 4
_TIG_IZ_Sxkg_argc:
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_Sxkg_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Sxkg_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Sxkg_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 86 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Sxkg--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_Sxkg_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_Sxkg_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_Sxkg_envp(%rip)
	nop
	movq	$10, -8(%rbp)
.L26:
	cmpq	$13, -8(%rbp)
	ja	.L28
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
	.long	.L16-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L28-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L28-.L8
	.long	.L7-.L8
	.text
.L15:
	cmpl	$0, -12(%rbp)
	je	.L17
	movq	$13, -8(%rbp)
	jmp	.L19
.L17:
	movq	$6, -8(%rbp)
	jmp	.L19
.L11:
	movl	$0, %eax
	jmp	.L27
.L9:
	cmpl	$10, -16(%rbp)
	je	.L21
	cmpl	$32, -16(%rbp)
	jne	.L22
	movq	$7, -8(%rbp)
	jmp	.L23
.L21:
	movq	$0, -8(%rbp)
	jmp	.L23
.L22:
	movq	$4, -8(%rbp)
	nop
.L23:
	jmp	.L19
.L7:
	movl	-16(%rbp), %eax
	movl	%eax, %edi
	call	putchar@PLT
	movq	$6, -8(%rbp)
	jmp	.L19
.L13:
	call	getchar@PLT
	movl	%eax, -16(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L19
.L14:
	cmpl	$-1, -16(%rbp)
	je	.L24
	movq	$11, -8(%rbp)
	jmp	.L19
.L24:
	movq	$8, -8(%rbp)
	jmp	.L19
.L10:
	movl	$1, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L19
.L16:
	movl	$10, %edi
	call	putchar@PLT
	movl	$1, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L19
.L12:
	movl	$0, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L19
.L28:
	nop
.L19:
	jmp	.L26
.L27:
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
