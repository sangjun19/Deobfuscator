	.file	"GabrielSbarbosa_Graduacao---FATEC---Gabriel_programa05_flatten.c"
	.text
	.globl	_TIG_IZ_I2os_envp
	.bss
	.align 8
	.type	_TIG_IZ_I2os_envp, @object
	.size	_TIG_IZ_I2os_envp, 8
_TIG_IZ_I2os_envp:
	.zero	8
	.globl	_TIG_IZ_I2os_argv
	.align 8
	.type	_TIG_IZ_I2os_argv, @object
	.size	_TIG_IZ_I2os_argv, 8
_TIG_IZ_I2os_argv:
	.zero	8
	.globl	_TIG_IZ_I2os_argc
	.align 4
	.type	_TIG_IZ_I2os_argc, @object
	.size	_TIG_IZ_I2os_argc, 4
_TIG_IZ_I2os_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"SUBTRACAO!"
.LC1:
	.string	"Informe 2 operandos: "
.LC2:
	.string	"%f%f"
.LC3:
	.string	" Resultado: %.3f \n"
	.text
	.globl	realizarSubtracao
	.type	realizarSubtracao, @function
realizarSubtracao:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -16(%rbp)
.L7:
	cmpq	$2, -16(%rbp)
	je	.L2
	cmpq	$2, -16(%rbp)
	ja	.L11
	cmpq	$0, -16(%rbp)
	je	.L4
	cmpq	$1, -16(%rbp)
	jne	.L11
	jmp	.L10
.L4:
	movq	$2, -16(%rbp)
	jmp	.L6
.L2:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movss	-28(%rbp), %xmm0
	movss	-24(%rbp), %xmm1
	subss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm2, %xmm2
	cvtss2sd	-20(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L6
.L11:
	nop
.L6:
	jmp	.L7
.L10:
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L9
	call	__stack_chk_fail@PLT
.L9:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	realizarSubtracao, .-realizarSubtracao
	.section	.rodata
.LC4:
	.string	"SOMA!"
	.text
	.globl	realizarSoma
	.type	realizarSoma, @function
realizarSoma:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -16(%rbp)
.L18:
	cmpq	$2, -16(%rbp)
	je	.L21
	cmpq	$2, -16(%rbp)
	ja	.L22
	cmpq	$0, -16(%rbp)
	je	.L15
	cmpq	$1, -16(%rbp)
	jne	.L22
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movss	-28(%rbp), %xmm1
	movss	-24(%rbp), %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm2, %xmm2
	cvtss2sd	-20(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$2, -16(%rbp)
	jmp	.L16
.L15:
	movq	$1, -16(%rbp)
	jmp	.L16
.L22:
	nop
.L16:
	jmp	.L18
.L21:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L20
	call	__stack_chk_fail@PLT
.L20:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	realizarSoma, .-realizarSoma
	.section	.rodata
.LC5:
	.string	"Multiplica\303\247\303\243o "
.LC6:
	.string	" operacao invalida "
.LC7:
	.string	"Informe a operacao desejada: "
.LC8:
	.string	"%i"
.LC9:
	.string	" divisao \n "
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
	movq	$0, _TIG_IZ_I2os_envp(%rip)
	nop
.L24:
	movq	$0, _TIG_IZ_I2os_argv(%rip)
	nop
.L25:
	movl	$0, _TIG_IZ_I2os_argc(%rip)
	nop
	nop
.L26:
.L27:
#APP
# 66 "GabrielSbarbosa_Graduacao---FATEC---Gabriel_programa05.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-I2os--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_I2os_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_I2os_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_I2os_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L47:
	cmpq	$11, -16(%rbp)
	ja	.L50
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L30(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L30(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L30:
	.long	.L38-.L30
	.long	.L50-.L30
	.long	.L37-.L30
	.long	.L36-.L30
	.long	.L50-.L30
	.long	.L35-.L30
	.long	.L34-.L30
	.long	.L33-.L30
	.long	.L32-.L30
	.long	.L31-.L30
	.long	.L50-.L30
	.long	.L29-.L30
	.text
.L32:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -16(%rbp)
	jmp	.L39
.L36:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L48
	jmp	.L49
.L29:
	call	realizarSoma
	movq	$3, -16(%rbp)
	jmp	.L39
.L31:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -16(%rbp)
	jmp	.L39
.L34:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -16(%rbp)
	jmp	.L39
.L35:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L39
.L38:
	call	realizarSubtracao
	movq	$3, -16(%rbp)
	jmp	.L39
.L33:
	movq	$6, -16(%rbp)
	jmp	.L39
.L37:
	movl	-20(%rbp), %eax
	cmpl	$4, %eax
	je	.L41
	cmpl	$4, %eax
	jg	.L42
	cmpl	$3, %eax
	je	.L43
	cmpl	$3, %eax
	jg	.L42
	cmpl	$1, %eax
	je	.L44
	cmpl	$2, %eax
	je	.L45
	jmp	.L42
.L41:
	movq	$5, -16(%rbp)
	jmp	.L46
.L43:
	movq	$8, -16(%rbp)
	jmp	.L46
.L45:
	movq	$0, -16(%rbp)
	jmp	.L46
.L44:
	movq	$11, -16(%rbp)
	jmp	.L46
.L42:
	movq	$9, -16(%rbp)
	nop
.L46:
	jmp	.L39
.L50:
	nop
.L39:
	jmp	.L47
.L49:
	call	__stack_chk_fail@PLT
.L48:
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
