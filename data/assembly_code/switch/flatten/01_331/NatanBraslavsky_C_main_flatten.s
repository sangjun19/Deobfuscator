	.file	"NatanBraslavsky_C_main_flatten.c"
	.text
	.globl	_TIG_IZ_8eYq_envp
	.bss
	.align 8
	.type	_TIG_IZ_8eYq_envp, @object
	.size	_TIG_IZ_8eYq_envp, 8
_TIG_IZ_8eYq_envp:
	.zero	8
	.globl	_TIG_IZ_8eYq_argc
	.align 4
	.type	_TIG_IZ_8eYq_argc, @object
	.size	_TIG_IZ_8eYq_argc, 4
_TIG_IZ_8eYq_argc:
	.zero	4
	.globl	_TIG_IZ_8eYq_argv
	.align 8
	.type	_TIG_IZ_8eYq_argv, @object
	.size	_TIG_IZ_8eYq_argv, 8
_TIG_IZ_8eYq_argv:
	.zero	8
	.section	.rodata
.LC1:
	.string	"Digite um n\303\272mero: "
.LC2:
	.string	"%f"
.LC3:
	.string	"Digite outro n\303\272mero: "
	.align 8
.LC4:
	.string	"\nAdi\303\247\303\243o       ( + )\nSubtra\303\247\303\243o    ( - )\nMultiplica\303\247\303\243o( * )\nDivis\303\243o      ( / ) \nDigite a opera\303\247\303\243o: "
.LC5:
	.string	" %c"
.LC6:
	.string	"%.2f + %f = %.2f"
.LC7:
	.string	"%.2f - %f = %.2f"
.LC8:
	.string	"%.2f * %.2f = %.2f"
	.align 8
.LC9:
	.string	"N\303\243o se pode fazer divis\303\243o por 0."
.LC10:
	.string	"%.2f / %.2f = %.2f"
.LC11:
	.string	"Inv\303\241lido"
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
	movq	$0, _TIG_IZ_8eYq_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_8eYq_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_8eYq_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 126 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-8eYq--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_8eYq_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_8eYq_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_8eYq_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L31:
	cmpq	$14, -16(%rbp)
	ja	.L35
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
	.long	.L19-.L8
	.long	.L35-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L35-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L35-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	pxor	%xmm0, %xmm0
	movss	%xmm0, -24(%rbp)
	pxor	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-25(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$13, -16(%rbp)
	jmp	.L20
.L10:
	movss	-20(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jp	.L21
	pxor	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jne	.L21
	movq	$6, -16(%rbp)
	jmp	.L20
.L21:
	movq	$0, -16(%rbp)
	jmp	.L20
.L17:
	movss	-24(%rbp), %xmm1
	movss	-20(%rbp), %xmm0
	addss	%xmm1, %xmm0
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
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L20
.L11:
	movss	-24(%rbp), %xmm0
	movss	-20(%rbp), %xmm1
	subss	%xmm1, %xmm0
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
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L20
.L13:
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
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L20
.L9:
	movzbl	-25(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	je	.L24
	cmpl	$47, %eax
	jg	.L25
	cmpl	$45, %eax
	je	.L26
	cmpl	$45, %eax
	jg	.L25
	cmpl	$42, %eax
	je	.L27
	cmpl	$43, %eax
	je	.L28
	jmp	.L25
.L24:
	movq	$12, -16(%rbp)
	jmp	.L29
.L27:
	movq	$9, -16(%rbp)
	jmp	.L29
.L26:
	movq	$11, -16(%rbp)
	jmp	.L29
.L28:
	movq	$3, -16(%rbp)
	jmp	.L29
.L25:
	movq	$7, -16(%rbp)
	nop
.L29:
	jmp	.L20
.L15:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L20
.L16:
	movq	$14, -16(%rbp)
	jmp	.L20
.L12:
	movl	$0, %eax
	jmp	.L32
.L19:
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
	movq	$2, -16(%rbp)
	jmp	.L20
.L14:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -16(%rbp)
	jmp	.L20
.L18:
	movl	$0, %eax
	jmp	.L32
.L35:
	nop
.L20:
	jmp	.L31
.L32:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L33
	call	__stack_chk_fail@PLT
.L33:
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
