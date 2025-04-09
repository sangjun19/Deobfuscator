	.file	"uSiiLaS_visual30__flatten.c"
	.text
	.globl	_TIG_IZ_WhDq_argv
	.bss
	.align 8
	.type	_TIG_IZ_WhDq_argv, @object
	.size	_TIG_IZ_WhDq_argv, 8
_TIG_IZ_WhDq_argv:
	.zero	8
	.globl	_TIG_IZ_WhDq_argc
	.align 4
	.type	_TIG_IZ_WhDq_argc, @object
	.size	_TIG_IZ_WhDq_argc, 4
_TIG_IZ_WhDq_argc:
	.zero	4
	.globl	_TIG_IZ_WhDq_envp
	.align 8
	.type	_TIG_IZ_WhDq_envp, @object
	.size	_TIG_IZ_WhDq_envp, 8
_TIG_IZ_WhDq_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"VOTA\303\207AO NAO OBRIGATORIA!"
.LC1:
	.string	"VOTA\303\207AO OBRIGATORIA!"
.LC2:
	.string	"Digite sua idade: "
.LC3:
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
	movq	$0, _TIG_IZ_WhDq_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_WhDq_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_WhDq_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 85 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-WhDq--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_WhDq_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_WhDq_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_WhDq_envp(%rip)
	nop
	movq	$3, -16(%rbp)
.L22:
	cmpq	$7, -16(%rbp)
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
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L11:
	movl	-20(%rbp), %eax
	cmpl	$65, %eax
	jle	.L16
	movq	$7, -16(%rbp)
	jmp	.L18
.L16:
	movq	$0, -16(%rbp)
	jmp	.L18
.L14:
	movl	-20(%rbp), %eax
	cmpl	$17, %eax
	jg	.L19
	movq	$6, -16(%rbp)
	jmp	.L18
.L19:
	movq	$4, -16(%rbp)
	jmp	.L18
.L12:
	movq	$2, -16(%rbp)
	jmp	.L18
.L9:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L18
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L23
	jmp	.L24
.L15:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L18
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L18
.L13:
	movl	$0, -20(%rbp)
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -16(%rbp)
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
