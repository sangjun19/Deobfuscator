	.file	"Yclpaiva_C_programming_ficha4_04_flatten.c"
	.text
	.globl	_TIG_IZ_8VIe_argc
	.bss
	.align 4
	.type	_TIG_IZ_8VIe_argc, @object
	.size	_TIG_IZ_8VIe_argc, 4
_TIG_IZ_8VIe_argc:
	.zero	4
	.globl	_TIG_IZ_8VIe_envp
	.align 8
	.type	_TIG_IZ_8VIe_envp, @object
	.size	_TIG_IZ_8VIe_envp, 8
_TIG_IZ_8VIe_envp:
	.zero	8
	.globl	_TIG_IZ_8VIe_argv
	.align 8
	.type	_TIG_IZ_8VIe_argv, @object
	.size	_TIG_IZ_8VIe_argv, 8
_TIG_IZ_8VIe_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Informe o valor do produto: "
.LC1:
	.string	"%f"
.LC2:
	.string	"A vista (1) | A prazo (2): "
.LC3:
	.string	"%d"
.LC5:
	.string	"Valor a ser pago: %2.f\n"
.LC6:
	.string	"Op\303\247\303\243o incorreta"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_8VIe_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_8VIe_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_8VIe_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-8VIe--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_8VIe_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_8VIe_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_8VIe_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L20:
	cmpq	$9, -16(%rbp)
	ja	.L23
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
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L23-.L8
	.long	.L9-.L8
	.long	.L23-.L8
	.long	.L23-.L8
	.long	.L7-.L8
	.text
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -16(%rbp)
	jmp	.L15
.L13:
	movq	$4, -16(%rbp)
	jmp	.L15
.L11:
	movss	-20(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movsd	.LC4(%rip), %xmm0
	mulsd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$9, -16(%rbp)
	jmp	.L15
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L21
	jmp	.L22
.L9:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -16(%rbp)
	jmp	.L15
.L14:
	movl	-24(%rbp), %eax
	cmpl	$1, %eax
	je	.L17
	cmpl	$2, %eax
	jne	.L18
	movq	$2, -16(%rbp)
	jmp	.L19
.L17:
	movq	$3, -16(%rbp)
	jmp	.L19
.L18:
	movq	$6, -16(%rbp)
	nop
.L19:
	jmp	.L15
.L12:
	movss	-20(%rbp), %xmm0
	pxor	%xmm2, %xmm2
	cvtss2sd	%xmm0, %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$9, -16(%rbp)
	jmp	.L15
.L23:
	nop
.L15:
	jmp	.L20
.L22:
	call	__stack_chk_fail@PLT
.L21:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC4:
	.long	-858993459
	.long	1072483532
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
