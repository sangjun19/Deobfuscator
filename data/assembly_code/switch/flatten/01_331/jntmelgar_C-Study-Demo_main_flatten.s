	.file	"jntmelgar_C-Study-Demo_main_flatten.c"
	.text
	.globl	_TIG_IZ_f8dv_argc
	.bss
	.align 4
	.type	_TIG_IZ_f8dv_argc, @object
	.size	_TIG_IZ_f8dv_argc, 4
_TIG_IZ_f8dv_argc:
	.zero	4
	.globl	_TIG_IZ_f8dv_envp
	.align 8
	.type	_TIG_IZ_f8dv_envp, @object
	.size	_TIG_IZ_f8dv_envp, 8
_TIG_IZ_f8dv_envp:
	.zero	8
	.globl	_TIG_IZ_f8dv_argv
	.align 8
	.type	_TIG_IZ_f8dv_argv, @object
	.size	_TIG_IZ_f8dv_argv, 8
_TIG_IZ_f8dv_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"informe o percurso a ser percorrido em KM do carro: "
.LC1:
	.string	"%f"
	.align 8
.LC2:
	.string	"\nInforme o tipo de carro: \nA = 8km/litro \nB = 9km/litro \nC = 12km/litro"
.LC3:
	.string	"%s"
	.align 8
.LC5:
	.string	"O consumo estimado \303\251 %.0fkm/l"
.LC8:
	.string	"Carro invalido"
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
	movq	$0, _TIG_IZ_f8dv_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_f8dv_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_f8dv_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 112 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-f8dv--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_f8dv_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_f8dv_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_f8dv_envp(%rip)
	nop
	movq	$0, -16(%rbp)
.L23:
	cmpq	$13, -16(%rbp)
	ja	.L26
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
	.long	.L26-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L26-.L8
	.long	.L11-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L7-.L8
	.text
.L12:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-25(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$6, -16(%rbp)
	jmp	.L16
.L13:
	movss	-24(%rbp), %xmm1
	movss	.LC4(%rip), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm2, %xmm2
	cvtss2sd	-20(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L16
.L10:
	movss	-24(%rbp), %xmm1
	movss	.LC6(%rip), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm3, %xmm3
	cvtss2sd	-20(%rbp), %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L16
.L7:
	movss	-24(%rbp), %xmm1
	movss	.LC7(%rip), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm4, %xmm4
	cvtss2sd	-20(%rbp), %xmm4
	movq	%xmm4, %rax
	movq	%rax, %xmm0
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L16
.L11:
	movzbl	-25(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$99, %eax
	je	.L17
	cmpl	$99, %eax
	jg	.L18
	cmpl	$97, %eax
	je	.L19
	cmpl	$98, %eax
	je	.L20
	jmp	.L18
.L17:
	movq	$3, -16(%rbp)
	jmp	.L21
.L20:
	movq	$13, -16(%rbp)
	jmp	.L21
.L19:
	movq	$9, -16(%rbp)
	jmp	.L21
.L18:
	movq	$10, -16(%rbp)
	nop
.L21:
	jmp	.L16
.L9:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L16
.L15:
	movq	$4, -16(%rbp)
	jmp	.L16
.L14:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L24
	jmp	.L25
.L26:
	nop
.L16:
	jmp	.L23
.L25:
	call	__stack_chk_fail@PLT
.L24:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC4:
	.long	1094713344
	.align 4
.LC6:
	.long	1090519040
	.align 4
.LC7:
	.long	1091567616
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
