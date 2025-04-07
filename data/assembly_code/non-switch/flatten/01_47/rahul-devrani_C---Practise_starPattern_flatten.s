	.file	"rahul-devrani_C---Practise_starPattern_flatten.c"
	.text
	.globl	_TIG_IZ_1ECm_argv
	.bss
	.align 8
	.type	_TIG_IZ_1ECm_argv, @object
	.size	_TIG_IZ_1ECm_argv, 8
_TIG_IZ_1ECm_argv:
	.zero	8
	.globl	_TIG_IZ_1ECm_envp
	.align 8
	.type	_TIG_IZ_1ECm_envp, @object
	.size	_TIG_IZ_1ECm_envp, 8
_TIG_IZ_1ECm_envp:
	.zero	8
	.globl	_TIG_IZ_1ECm_argc
	.align 4
	.type	_TIG_IZ_1ECm_argc, @object
	.size	_TIG_IZ_1ECm_argc, 4
_TIG_IZ_1ECm_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Enter the number of rows"
.LC1:
	.string	"%d"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_1ECm_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_1ECm_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_1ECm_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 87 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-1ECm--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_1ECm_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_1ECm_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_1ECm_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L22:
	cmpq	$13, -16(%rbp)
	ja	.L25
	movq	-16(%rbp), %rax
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
	.long	.L15-.L8
	.long	.L25-.L8
	.long	.L25-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L25-.L8
	.long	.L11-.L8
	.long	.L25-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L25-.L8
	.long	.L25-.L8
	.long	.L7-.L8
	.text
.L13:
	movl	-28(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L16
	movq	$10, -16(%rbp)
	jmp	.L18
.L16:
	movq	$3, -16(%rbp)
	jmp	.L18
.L14:
	movl	$10, %edi
	call	putchar@PLT
	addl	$1, -24(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L18
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L23
	jmp	.L24
.L7:
	movl	$0, -20(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L18
.L12:
	movq	$0, -16(%rbp)
	jmp	.L18
.L9:
	movl	$42, %edi
	call	putchar@PLT
	addl	$1, -20(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L18
.L15:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -24(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L18
.L11:
	movl	-28(%rbp), %eax
	cmpl	%eax, -24(%rbp)
	jge	.L20
	movq	$13, -16(%rbp)
	jmp	.L18
.L20:
	movq	$9, -16(%rbp)
	jmp	.L18
.L25:
	nop
.L18:
	jmp	.L22
.L24:
	call	__stack_chk_fail@PLT
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
