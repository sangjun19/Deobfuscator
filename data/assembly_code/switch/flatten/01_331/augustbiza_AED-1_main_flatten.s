	.file	"augustbiza_AED-1_main_flatten.c"
	.text
	.globl	_TIG_IZ_JLkG_envp
	.bss
	.align 8
	.type	_TIG_IZ_JLkG_envp, @object
	.size	_TIG_IZ_JLkG_envp, 8
_TIG_IZ_JLkG_envp:
	.zero	8
	.globl	_TIG_IZ_JLkG_argv
	.align 8
	.type	_TIG_IZ_JLkG_argv, @object
	.size	_TIG_IZ_JLkG_argv, 8
_TIG_IZ_JLkG_argv:
	.zero	8
	.globl	_TIG_IZ_JLkG_argc
	.align 4
	.type	_TIG_IZ_JLkG_argc, @object
	.size	_TIG_IZ_JLkG_argc, 4
_TIG_IZ_JLkG_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%f - %f = %f\n"
.LC1:
	.string	"%f + %f = %f\n"
.LC2:
	.string	"Erro.\n"
.LC3:
	.string	"%f * %f = %f\n"
.LC4:
	.string	"Calculadora Simples."
	.align 8
.LC5:
	.string	"Operacoes:\n1-Adica0\n2-Subtracao\n3-Multiplicacao\n4-Divisao\nInsira a opera\303\247\303\243o desejada: "
.LC6:
	.string	"%d"
.LC7:
	.string	"Insira o primeiro numero: "
.LC8:
	.string	"%f"
.LC9:
	.string	"Insira o segundo numero: "
.LC10:
	.string	"%f / %f = %f\n"
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
	movq	$0, _TIG_IZ_JLkG_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_JLkG_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_JLkG_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 112 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-JLkG--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_JLkG_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_JLkG_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_JLkG_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L25:
	cmpq	$12, -16(%rbp)
	ja	.L28
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
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L28-.L8
	.long	.L28-.L8
	.long	.L14-.L8
	.long	.L28-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L28-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L14:
	movss	-24(%rbp), %xmm0
	movss	-20(%rbp), %xmm1
	subss	%xmm1, %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movss	-20(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movss	-24(%rbp), %xmm2
	pxor	%xmm3, %xmm3
	cvtss2sd	%xmm2, %xmm3
	movq	%xmm3, %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L17
.L7:
	movss	-24(%rbp), %xmm1
	movss	-20(%rbp), %xmm0
	addss	%xmm1, %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movss	-20(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movss	-24(%rbp), %xmm2
	pxor	%xmm4, %xmm4
	cvtss2sd	%xmm2, %xmm4
	movq	%xmm4, %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L17
.L11:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -16(%rbp)
	jmp	.L17
.L15:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L26
	jmp	.L27
.L9:
	movss	-24(%rbp), %xmm1
	movss	-20(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movss	-20(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movss	-24(%rbp), %xmm2
	pxor	%xmm5, %xmm5
	cvtss2sd	%xmm2, %xmm5
	movq	%xmm5, %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L17
.L13:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$10, -16(%rbp)
	jmp	.L17
.L10:
	movl	-28(%rbp), %eax
	cmpl	$4, %eax
	je	.L19
	cmpl	$4, %eax
	jg	.L20
	cmpl	$3, %eax
	je	.L21
	cmpl	$3, %eax
	jg	.L20
	cmpl	$1, %eax
	je	.L22
	cmpl	$2, %eax
	je	.L23
	jmp	.L20
.L19:
	movq	$0, -16(%rbp)
	jmp	.L24
.L21:
	movq	$11, -16(%rbp)
	jmp	.L24
.L23:
	movq	$4, -16(%rbp)
	jmp	.L24
.L22:
	movq	$12, -16(%rbp)
	jmp	.L24
.L20:
	movq	$8, -16(%rbp)
	nop
.L24:
	jmp	.L17
.L16:
	movss	-24(%rbp), %xmm0
	movss	-20(%rbp), %xmm1
	divss	%xmm1, %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movss	-20(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movss	-24(%rbp), %xmm2
	pxor	%xmm6, %xmm6
	cvtss2sd	%xmm2, %xmm6
	movq	%xmm6, %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L17
.L12:
	movq	$6, -16(%rbp)
	jmp	.L17
.L28:
	nop
.L17:
	jmp	.L25
.L27:
	call	__stack_chk_fail@PLT
.L26:
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
