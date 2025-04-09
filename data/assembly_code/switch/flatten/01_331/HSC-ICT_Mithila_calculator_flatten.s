	.file	"HSC-ICT_Mithila_calculator_flatten.c"
	.text
	.globl	_TIG_IZ_rpSd_envp
	.bss
	.align 8
	.type	_TIG_IZ_rpSd_envp, @object
	.size	_TIG_IZ_rpSd_envp, 8
_TIG_IZ_rpSd_envp:
	.zero	8
	.globl	_TIG_IZ_rpSd_argc
	.align 4
	.type	_TIG_IZ_rpSd_argc, @object
	.size	_TIG_IZ_rpSd_argc, 4
_TIG_IZ_rpSd_argc:
	.zero	4
	.globl	_TIG_IZ_rpSd_argv
	.align 8
	.type	_TIG_IZ_rpSd_argv, @object
	.size	_TIG_IZ_rpSd_argv, 8
_TIG_IZ_rpSd_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%.2lf + %.2lf = %.2lf\n"
.LC1:
	.string	"%.2lf - %.2lf = %.2lf\n"
.LC2:
	.string	"%.2lf / %.2lf = %.2lf\n"
	.align 8
.LC3:
	.string	"Enter an operator (+, -, *, /): "
.LC4:
	.string	"%c"
.LC5:
	.string	"Enter two operands: "
.LC6:
	.string	"%lf %lf"
.LC7:
	.string	"%lf"
.LC8:
	.string	"%.2lf * %.2lf = %.2lf\n"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_rpSd_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_rpSd_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_rpSd_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 150 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-rpSd--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_rpSd_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_rpSd_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_rpSd_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L25:
	cmpq	$11, -16(%rbp)
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
	.long	.L28-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L28-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L28-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	movsd	-40(%rbp), %xmm1
	movsd	-32(%rbp), %xmm0
	addsd	%xmm0, %xmm1
	movsd	-32(%rbp), %xmm0
	movq	-40(%rbp), %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L17
.L14:
	movzbl	-41(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	je	.L18
	cmpl	$47, %eax
	jg	.L19
	cmpl	$45, %eax
	je	.L20
	cmpl	$45, %eax
	jg	.L19
	cmpl	$42, %eax
	je	.L21
	cmpl	$43, %eax
	je	.L22
	jmp	.L19
.L18:
	movq	$6, -16(%rbp)
	jmp	.L23
.L21:
	movq	$0, -16(%rbp)
	jmp	.L23
.L20:
	movq	$11, -16(%rbp)
	jmp	.L23
.L22:
	movq	$8, -16(%rbp)
	jmp	.L23
.L19:
	movq	$2, -16(%rbp)
	nop
.L23:
	jmp	.L17
.L7:
	movsd	-40(%rbp), %xmm0
	movsd	-32(%rbp), %xmm1
	movapd	%xmm0, %xmm2
	subsd	%xmm1, %xmm2
	movsd	-32(%rbp), %xmm0
	movq	-40(%rbp), %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L17
.L12:
	movsd	-40(%rbp), %xmm0
	movsd	-32(%rbp), %xmm1
	movapd	%xmm0, %xmm2
	divsd	%xmm1, %xmm2
	movsd	-32(%rbp), %xmm0
	movq	-40(%rbp), %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L17
.L13:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L26
	jmp	.L27
.L9:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-41(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-32(%rbp), %rdx
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$3, -16(%rbp)
	jmp	.L17
.L16:
	movsd	-40(%rbp), %xmm1
	movsd	-32(%rbp), %xmm0
	mulsd	%xmm0, %xmm1
	movsd	-32(%rbp), %xmm0
	movq	-40(%rbp), %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$5, -16(%rbp)
	jmp	.L17
.L11:
	movq	$10, -16(%rbp)
	jmp	.L17
.L15:
	movq	$5, -16(%rbp)
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
.LFE5:
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
