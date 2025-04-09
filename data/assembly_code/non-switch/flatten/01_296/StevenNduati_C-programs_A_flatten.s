	.file	"StevenNduati_C-programs_A_flatten.c"
	.text
	.globl	_TIG_IZ_wUrD_argc
	.bss
	.align 4
	.type	_TIG_IZ_wUrD_argc, @object
	.size	_TIG_IZ_wUrD_argc, 4
_TIG_IZ_wUrD_argc:
	.zero	4
	.globl	_TIG_IZ_wUrD_argv
	.align 8
	.type	_TIG_IZ_wUrD_argv, @object
	.size	_TIG_IZ_wUrD_argv, 8
_TIG_IZ_wUrD_argv:
	.zero	8
	.globl	_TIG_IZ_wUrD_envp
	.align 8
	.type	_TIG_IZ_wUrD_envp, @object
	.size	_TIG_IZ_wUrD_envp, 8
_TIG_IZ_wUrD_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"The sum is %d"
.LC1:
	.string	"%d \n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movq	$0, _TIG_IZ_wUrD_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_wUrD_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_wUrD_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 123 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-wUrD--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_wUrD_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_wUrD_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_wUrD_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L19:
	cmpq	$10, -8(%rbp)
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
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L21-.L8
	.long	.L21-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L21-.L8
	.long	.L21-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L12:
	cmpl	$0, -12(%rbp)
	je	.L15
	movq	$9, -8(%rbp)
	jmp	.L17
.L15:
	movq	$8, -8(%rbp)
	jmp	.L17
.L10:
	movl	-16(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -8(%rbp)
	jmp	.L17
.L13:
	movq	$5, -8(%rbp)
	jmp	.L17
.L9:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-20(%rbp), %eax
	addl	%eax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L17
.L11:
	movl	$10, -20(%rbp)
	movl	$0, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L17
.L7:
	movl	$0, %eax
	jmp	.L20
.L14:
	movl	-20(%rbp), %eax
	movl	%eax, -12(%rbp)
	addl	$1, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L17
.L21:
	nop
.L17:
	jmp	.L19
.L20:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
