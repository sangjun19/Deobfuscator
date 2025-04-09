	.file	"juliosxlc_lista_4_flatten.c"
	.text
	.globl	_TIG_IZ_agJG_argc
	.bss
	.align 4
	.type	_TIG_IZ_agJG_argc, @object
	.size	_TIG_IZ_agJG_argc, 4
_TIG_IZ_agJG_argc:
	.zero	4
	.globl	_TIG_IZ_agJG_envp
	.align 8
	.type	_TIG_IZ_agJG_envp, @object
	.size	_TIG_IZ_agJG_envp, 8
_TIG_IZ_agJG_envp:
	.zero	8
	.globl	_TIG_IZ_agJG_argv
	.align 8
	.type	_TIG_IZ_agJG_argv, @object
	.size	_TIG_IZ_agJG_argv, 8
_TIG_IZ_agJG_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Escolha uma das opcoes:\n "
.LC1:
	.string	"1. Digitar um novo valor:\n "
.LC2:
	.string	"2. Sair:\n "
.LC3:
	.string	"%d"
.LC4:
	.string	"Digite um novo valor:"
.LC5:
	.string	"Menu finalizado"
	.align 8
.LC7:
	.string	"opcao invalida, digite outro item do menu."
	.align 8
.LC8:
	.string	"A quatindade de valores inseridos foram de: %d\n"
.LC9:
	.string	"A media e: %d\n"
	.align 8
.LC10:
	.string	"A quantidade de valores positivos encontrados foram de: %d\n"
	.align 8
.LC11:
	.string	"A quantidade de valores negativos encontrados foram de: %d\n"
	.align 8
.LC12:
	.string	"A porcetagem de valores positivos em relacao ao total de numeros encontrados foram de: %.2f%%\n"
	.align 8
.LC13:
	.string	"A porcetagem de valores negativos em relacao ao total de numeros encontrados foram de: %.2f%%\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_agJG_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_agJG_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_agJG_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 111 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-agJG--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_agJG_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_agJG_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_agJG_envp(%rip)
	nop
	movq	$23, -16(%rbp)
.L34:
	cmpq	$23, -16(%rbp)
	ja	.L37
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
	.long	.L37-.L8
	.long	.L22-.L8
	.long	.L37-.L8
	.long	.L21-.L8
	.long	.L37-.L8
	.long	.L20-.L8
	.long	.L37-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L37-.L8
	.long	.L37-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L37-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L37-.L8
	.long	.L37-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L14:
	cmpl	$0, -36(%rbp)
	jle	.L23
	movq	$10, -16(%rbp)
	jmp	.L25
.L23:
	movq	$20, -16(%rbp)
	jmp	.L25
.L18:
	movl	-48(%rbp), %eax
	cmpl	$1, %eax
	je	.L26
	cmpl	$2, %eax
	jne	.L27
	movq	$16, -16(%rbp)
	jmp	.L28
.L26:
	movq	$3, -16(%rbp)
	jmp	.L28
.L27:
	movq	$5, -16(%rbp)
	nop
.L28:
	jmp	.L25
.L22:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$8, -16(%rbp)
	jmp	.L25
.L7:
	movq	$17, -16(%rbp)
	jmp	.L25
.L21:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-52(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -36(%rbp)
	movl	-52(%rbp), %eax
	addl	%eax, -40(%rbp)
	movq	$7, -16(%rbp)
	jmp	.L25
.L13:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$13, -16(%rbp)
	jmp	.L25
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L35
	jmp	.L36
.L17:
	addl	$1, -32(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L25
.L15:
	movl	-48(%rbp), %eax
	cmpl	$2, %eax
	je	.L30
	movq	$1, -16(%rbp)
	jmp	.L25
.L30:
	movq	$14, -16(%rbp)
	jmp	.L25
.L12:
	movl	$0, -40(%rbp)
	movl	$0, -36(%rbp)
	movl	$0, -32(%rbp)
	movl	$0, -28(%rbp)
	pxor	%xmm0, %xmm0
	movss	%xmm0, -24(%rbp)
	pxor	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L25
.L9:
	addl	$1, -28(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L25
.L20:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$13, -16(%rbp)
	jmp	.L25
.L16:
	movl	-40(%rbp), %eax
	cltd
	idivl	-36(%rbp)
	movl	%eax, -44(%rbp)
	movl	-32(%rbp), %eax
	cltd
	idivl	-36(%rbp)
	imull	$100, %eax, %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movss	%xmm0, -20(%rbp)
	movl	-28(%rbp), %eax
	cltd
	idivl	-36(%rbp)
	imull	$100, %eax, %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movss	%xmm0, -24(%rbp)
	movq	$20, -16(%rbp)
	jmp	.L25
.L19:
	movl	-52(%rbp), %eax
	testl	%eax, %eax
	jle	.L32
	movq	$9, -16(%rbp)
	jmp	.L25
.L32:
	movq	$22, -16(%rbp)
	jmp	.L25
.L11:
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-32(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-28(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	pxor	%xmm1, %xmm1
	cvtss2sd	-24(%rbp), %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	pxor	%xmm2, %xmm2
	cvtss2sd	-20(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$21, -16(%rbp)
	jmp	.L25
.L37:
	nop
.L25:
	jmp	.L34
.L36:
	call	__stack_chk_fail@PLT
.L35:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
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
