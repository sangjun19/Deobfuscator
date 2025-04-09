	.file	"aigora_sentencias_control_area_triangulo_Circulo_flatten.c"
	.text
	.globl	_TIG_IZ_RdqY_argc
	.bss
	.align 4
	.type	_TIG_IZ_RdqY_argc, @object
	.size	_TIG_IZ_RdqY_argc, 4
_TIG_IZ_RdqY_argc:
	.zero	4
	.globl	_TIG_IZ_RdqY_argv
	.align 8
	.type	_TIG_IZ_RdqY_argv, @object
	.size	_TIG_IZ_RdqY_argv, 8
_TIG_IZ_RdqY_argv:
	.zero	8
	.globl	_TIG_IZ_RdqY_envp
	.align 8
	.type	_TIG_IZ_RdqY_envp, @object
	.size	_TIG_IZ_RdqY_envp, 8
_TIG_IZ_RdqY_envp:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Introduce el radio del circulo: "
.LC1:
	.string	"%f"
.LC2:
	.string	"El area del circulo es %f"
	.align 8
.LC4:
	.string	"Elija entra calcular el area de un circulo (c) o el de un triangulo (t): "
.LC5:
	.string	"%c"
	.align 8
.LC6:
	.string	"Introduce la base y la altura del triangulo: "
.LC7:
	.string	"%f %f"
.LC9:
	.string	"El area del triangulo es %f"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_RdqY_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_RdqY_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_RdqY_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 113 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-RdqY--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_RdqY_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_RdqY_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_RdqY_envp(%rip)
	nop
	movq	$3, -16(%rbp)
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
	.long	.L23-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L23-.L8
	.long	.L23-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L9:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movss	-32(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-28(%rbp), %xmm1
	movss	-32(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -24(%rbp)
	pxor	%xmm2, %xmm2
	cvtss2sd	-24(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L15
.L12:
	movq	$0, -16(%rbp)
	jmp	.L15
.L7:
	movq	$7, -16(%rbp)
	jmp	.L15
.L11:
	movzbl	-41(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$99, %eax
	je	.L16
	cmpl	$116, %eax
	jne	.L17
	movq	$2, -16(%rbp)
	jmp	.L18
.L16:
	movq	$8, -16(%rbp)
	jmp	.L18
.L17:
	movq	$9, -16(%rbp)
	nop
.L18:
	jmp	.L15
.L14:
	movss	.LC3(%rip), %xmm0
	movss	%xmm0, -28(%rbp)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-41(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$6, -16(%rbp)
	jmp	.L15
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L21
	jmp	.L22
.L13:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rdx
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movss	-40(%rbp), %xmm1
	movss	-36(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	movss	.LC8(%rip), %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm3, %xmm3
	cvtss2sd	-20(%rbp), %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
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
	.align 4
.LC3:
	.long	1078530000
	.align 4
.LC8:
	.long	1073741824
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
