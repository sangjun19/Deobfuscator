	.file	"Guilhermeiiu_exercicios-em-C_3_flatten.c"
	.text
	.globl	_TIG_IZ_6You_argv
	.bss
	.align 8
	.type	_TIG_IZ_6You_argv, @object
	.size	_TIG_IZ_6You_argv, 8
_TIG_IZ_6You_argv:
	.zero	8
	.globl	_TIG_IZ_6You_envp
	.align 8
	.type	_TIG_IZ_6You_envp, @object
	.size	_TIG_IZ_6You_envp, 8
_TIG_IZ_6You_envp:
	.zero	8
	.globl	_TIG_IZ_6You_argc
	.align 4
	.type	_TIG_IZ_6You_argc, @object
	.size	_TIG_IZ_6You_argc, 4
_TIG_IZ_6You_argc:
	.zero	4
	.text
	.globl	subtrair
	.type	subtrair, @function
subtrair:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -20(%rbp)
	movss	%xmm1, -24(%rbp)
	movq	%rdi, -32(%rbp)
	movq	$0, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$1, -8(%rbp)
	jne	.L8
	jmp	.L7
.L2:
	movss	-20(%rbp), %xmm0
	subss	-24(%rbp), %xmm0
	movq	-32(%rbp), %rax
	movss	%xmm0, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L5
.L8:
	nop
.L5:
	jmp	.L6
.L7:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	subtrair, .-subtrair
	.globl	somar
	.type	somar, @function
somar:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -20(%rbp)
	movss	%xmm1, -24(%rbp)
	movq	%rdi, -32(%rbp)
	movq	$0, -8(%rbp)
.L14:
	cmpq	$0, -8(%rbp)
	je	.L10
	cmpq	$1, -8(%rbp)
	jne	.L16
	jmp	.L15
.L10:
	movss	-20(%rbp), %xmm0
	addss	-24(%rbp), %xmm0
	movq	-32(%rbp), %rax
	movss	%xmm0, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L13
.L16:
	nop
.L13:
	jmp	.L14
.L15:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	somar, .-somar
	.section	.rodata
.LC0:
	.string	"Portuguese"
.LC1:
	.string	"Digite um n\303\272mero inicial: "
.LC2:
	.string	"%f"
.LC3:
	.string	"Opera\303\247\303\243o inv\303\241lida."
.LC4:
	.string	"Digite o pr\303\263ximo n\303\272mero: "
	.align 8
.LC5:
	.string	"Digite a opera\303\247\303\243o (+, -, *, /) ou 'F' para finalizar: "
.LC6:
	.string	" %c"
.LC7:
	.string	"Resultado: %.2f\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
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
	movq	$0, _TIG_IZ_6You_envp(%rip)
	nop
.L18:
	movq	$0, _TIG_IZ_6You_argv(%rip)
	nop
.L19:
	movl	$0, _TIG_IZ_6You_argc(%rip)
	nop
	nop
.L20:
.L21:
#APP
# 124 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-6You--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_6You_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_6You_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_6You_envp(%rip)
	nop
	movq	$13, -16(%rbp)
.L55:
	cmpq	$23, -16(%rbp)
	ja	.L58
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L24(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L24(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L24:
	.long	.L40-.L24
	.long	.L58-.L24
	.long	.L39-.L24
	.long	.L58-.L24
	.long	.L38-.L24
	.long	.L37-.L24
	.long	.L36-.L24
	.long	.L58-.L24
	.long	.L35-.L24
	.long	.L58-.L24
	.long	.L58-.L24
	.long	.L34-.L24
	.long	.L33-.L24
	.long	.L32-.L24
	.long	.L31-.L24
	.long	.L30-.L24
	.long	.L29-.L24
	.long	.L58-.L24
	.long	.L28-.L24
	.long	.L27-.L24
	.long	.L26-.L24
	.long	.L58-.L24
	.long	.L25-.L24
	.long	.L23-.L24
	.text
.L28:
	movzbl	-29(%rbp), %eax
	cmpb	$70, %al
	je	.L41
	movq	$22, -16(%rbp)
	jmp	.L43
.L41:
	movq	$11, -16(%rbp)
	jmp	.L43
.L38:
	movss	-24(%rbp), %xmm0
	movl	-28(%rbp), %eax
	leaq	-20(%rbp), %rdx
	movq	%rdx, %rdi
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	somar
	movq	$5, -16(%rbp)
	jmp	.L43
.L31:
	movzbl	-29(%rbp), %eax
	cmpb	$42, %al
	jne	.L44
	movq	$19, -16(%rbp)
	jmp	.L43
.L44:
	movq	$2, -16(%rbp)
	jmp	.L43
.L30:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	movl	$6, %edi
	call	setlocale@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$22, -16(%rbp)
	jmp	.L43
.L33:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$18, -16(%rbp)
	jmp	.L43
.L35:
	movss	-24(%rbp), %xmm0
	movl	-28(%rbp), %eax
	leaq	-20(%rbp), %rdx
	movq	%rdx, %rdi
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	dividir
	movq	$5, -16(%rbp)
	jmp	.L43
.L23:
	movzbl	-29(%rbp), %eax
	cmpb	$43, %al
	jne	.L46
	movq	$4, -16(%rbp)
	jmp	.L43
.L46:
	movq	$6, -16(%rbp)
	jmp	.L43
.L29:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$23, -16(%rbp)
	jmp	.L43
.L34:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L56
	jmp	.L57
.L32:
	movq	$15, -16(%rbp)
	jmp	.L43
.L27:
	movss	-24(%rbp), %xmm0
	movl	-28(%rbp), %eax
	leaq	-20(%rbp), %rdx
	movq	%rdx, %rdi
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	multiplicar
	movq	$5, -16(%rbp)
	jmp	.L43
.L36:
	movzbl	-29(%rbp), %eax
	cmpb	$45, %al
	jne	.L49
	movq	$0, -16(%rbp)
	jmp	.L43
.L49:
	movq	$14, -16(%rbp)
	jmp	.L43
.L25:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-29(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$20, -16(%rbp)
	jmp	.L43
.L37:
	movss	-20(%rbp), %xmm0
	pxor	%xmm2, %xmm2
	cvtss2sd	%xmm0, %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movss	-20(%rbp), %xmm0
	movss	%xmm0, -28(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L43
.L40:
	movss	-24(%rbp), %xmm0
	movl	-28(%rbp), %eax
	leaq	-20(%rbp), %rdx
	movq	%rdx, %rdi
	movaps	%xmm0, %xmm1
	movd	%eax, %xmm0
	call	subtrair
	movq	$5, -16(%rbp)
	jmp	.L43
.L39:
	movzbl	-29(%rbp), %eax
	cmpb	$47, %al
	jne	.L51
	movq	$8, -16(%rbp)
	jmp	.L43
.L51:
	movq	$12, -16(%rbp)
	jmp	.L43
.L26:
	movzbl	-29(%rbp), %eax
	cmpb	$70, %al
	jne	.L53
	movq	$11, -16(%rbp)
	jmp	.L43
.L53:
	movq	$16, -16(%rbp)
	jmp	.L43
.L58:
	nop
.L43:
	jmp	.L55
.L57:
	call	__stack_chk_fail@PLT
.L56:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.globl	multiplicar
	.type	multiplicar, @function
multiplicar:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movss	%xmm0, -20(%rbp)
	movss	%xmm1, -24(%rbp)
	movq	%rdi, -32(%rbp)
	movq	$0, -8(%rbp)
.L64:
	cmpq	$0, -8(%rbp)
	je	.L60
	cmpq	$1, -8(%rbp)
	jne	.L66
	jmp	.L65
.L60:
	movss	-20(%rbp), %xmm0
	mulss	-24(%rbp), %xmm0
	movq	-32(%rbp), %rax
	movss	%xmm0, (%rax)
	movq	$1, -8(%rbp)
	jmp	.L63
.L66:
	nop
.L63:
	jmp	.L64
.L65:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	multiplicar, .-multiplicar
	.section	.rodata
.LC9:
	.string	"Erro: Divis\303\243o por zero."
	.text
	.globl	dividir
	.type	dividir, @function
dividir:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movss	%xmm0, -20(%rbp)
	movss	%xmm1, -24(%rbp)
	movq	%rdi, -32(%rbp)
	movq	$1, -8(%rbp)
.L78:
	cmpq	$4, -8(%rbp)
	je	.L80
	cmpq	$4, -8(%rbp)
	ja	.L81
	cmpq	$3, -8(%rbp)
	je	.L70
	cmpq	$3, -8(%rbp)
	ja	.L81
	cmpq	$1, -8(%rbp)
	je	.L71
	cmpq	$2, -8(%rbp)
	je	.L72
	jmp	.L81
.L71:
	pxor	%xmm0, %xmm0
	ucomiss	-24(%rbp), %xmm0
	jp	.L79
	pxor	%xmm0, %xmm0
	ucomiss	-24(%rbp), %xmm0
	je	.L74
.L79:
	movq	$3, -8(%rbp)
	jmp	.L77
.L74:
	movq	$2, -8(%rbp)
	jmp	.L77
.L70:
	movss	-20(%rbp), %xmm0
	divss	-24(%rbp), %xmm0
	movq	-32(%rbp), %rax
	movss	%xmm0, (%rax)
	movq	$4, -8(%rbp)
	jmp	.L77
.L72:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-32(%rbp), %rax
	pxor	%xmm0, %xmm0
	movss	%xmm0, (%rax)
	movq	$4, -8(%rbp)
	jmp	.L77
.L81:
	nop
.L77:
	jmp	.L78
.L80:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	dividir, .-dividir
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
