	.file	"escarlate-lunar_Faculdade_tipo-de-combustivel_flatten.c"
	.text
	.globl	_TIG_IZ_nOpX_envp
	.bss
	.align 8
	.type	_TIG_IZ_nOpX_envp, @object
	.size	_TIG_IZ_nOpX_envp, 8
_TIG_IZ_nOpX_envp:
	.zero	8
	.globl	_TIG_IZ_nOpX_argc
	.align 4
	.type	_TIG_IZ_nOpX_argc, @object
	.size	_TIG_IZ_nOpX_argc, 4
_TIG_IZ_nOpX_argc:
	.zero	4
	.globl	_TIG_IZ_nOpX_argv
	.align 8
	.type	_TIG_IZ_nOpX_argv, @object
	.size	_TIG_IZ_nOpX_argv, 8
_TIG_IZ_nOpX_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"MUITO OBRIGADO"
.LC2:
	.string	"Alcool: %d\n"
.LC3:
	.string	"Gasolina: %d\n"
.LC4:
	.string	"Diesel: %d\n"
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
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_nOpX_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_nOpX_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_nOpX_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 127 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-nOpX--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_nOpX_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_nOpX_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_nOpX_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L27:
	cmpq	$15, -16(%rbp)
	ja	.L30
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
	.long	.L30-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L30-.L8
	.long	.L14-.L8
	.long	.L30-.L8
	.long	.L13-.L8
	.long	.L30-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L30-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L30-.L8
	.long	.L7-.L8
	.text
.L7:
	movl	$0, -28(%rbp)
	movl	$0, -24(%rbp)
	movl	$0, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L18
.L10:
	addl	$1, -28(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L18
.L17:
	movl	-32(%rbp), %eax
	cmpl	$4, %eax
	je	.L19
	movq	$3, -16(%rbp)
	jmp	.L18
.L19:
	movq	$2, -16(%rbp)
	jmp	.L18
.L15:
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$9, -16(%rbp)
	jmp	.L18
.L12:
	movl	-32(%rbp), %eax
	cmpl	$3, %eax
	je	.L21
	cmpl	$3, %eax
	jg	.L22
	cmpl	$1, %eax
	je	.L23
	cmpl	$2, %eax
	je	.L24
	jmp	.L22
.L21:
	movq	$13, -16(%rbp)
	jmp	.L25
.L24:
	movq	$10, -16(%rbp)
	jmp	.L25
.L23:
	movq	$12, -16(%rbp)
	jmp	.L25
.L22:
	movq	$1, -16(%rbp)
	nop
.L25:
	jmp	.L18
.L9:
	addl	$1, -20(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L18
.L14:
	movq	$15, -16(%rbp)
	jmp	.L18
.L11:
	addl	$1, -24(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L18
.L13:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L28
	jmp	.L29
.L16:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L18
.L30:
	nop
.L18:
	jmp	.L27
.L29:
	call	__stack_chk_fail@PLT
.L28:
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
