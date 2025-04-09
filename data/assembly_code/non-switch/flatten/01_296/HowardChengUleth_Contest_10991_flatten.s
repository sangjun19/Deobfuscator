	.file	"HowardChengUleth_Contest_10991_flatten.c"
	.text
	.globl	_TIG_IZ_a7mI_argc
	.bss
	.align 4
	.type	_TIG_IZ_a7mI_argc, @object
	.size	_TIG_IZ_a7mI_argc, 4
_TIG_IZ_a7mI_argc:
	.zero	4
	.globl	_TIG_IZ_a7mI_argv
	.align 8
	.type	_TIG_IZ_a7mI_argv, @object
	.size	_TIG_IZ_a7mI_argv, 8
_TIG_IZ_a7mI_argv:
	.zero	8
	.globl	_TIG_IZ_a7mI_envp
	.align 8
	.type	_TIG_IZ_a7mI_envp, @object
	.size	_TIG_IZ_a7mI_envp, 8
_TIG_IZ_a7mI_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%lf %lf %lf"
.LC2:
	.string	"%.6f\n"
.LC3:
	.string	"%d"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movl	%edi, -116(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_a7mI_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_a7mI_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_a7mI_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 111 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-a7mI--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_a7mI_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_a7mI_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_a7mI_envp(%rip)
	nop
	movq	$2, -72(%rbp)
.L17:
	cmpq	$6, -72(%rbp)
	ja	.L20
	movq	-72(%rbp), %rax
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
	.long	.L20-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L20-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L12:
	movl	-104(%rbp), %eax
	movl	%eax, -100(%rbp)
	movl	-104(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -104(%rbp)
	movq	$6, -72(%rbp)
	jmp	.L13
.L10:
	leaq	-80(%rbp), %rcx
	leaq	-88(%rbp), %rdx
	leaq	-96(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movsd	-96(%rbp), %xmm1
	movsd	-88(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -64(%rbp)
	movsd	-88(%rbp), %xmm1
	movsd	-80(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -56(%rbp)
	movsd	-96(%rbp), %xmm1
	movsd	-80(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -48(%rbp)
	movsd	-48(%rbp), %xmm1
	movsd	-56(%rbp), %xmm0
	movq	-64(%rbp), %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	heron
	movq	%xmm0, %rax
	movq	%rax, -40(%rbp)
	movsd	-56(%rbp), %xmm1
	movsd	-48(%rbp), %xmm0
	movq	-64(%rbp), %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	angle
	movq	%xmm0, %rax
	movq	%rax, -32(%rbp)
	movsd	-32(%rbp), %xmm0
	movsd	.LC1(%rip), %xmm2
	movapd	%xmm0, %xmm1
	divsd	%xmm2, %xmm1
	movsd	-96(%rbp), %xmm0
	mulsd	%xmm0, %xmm1
	movsd	-96(%rbp), %xmm0
	mulsd	%xmm0, %xmm1
	movsd	-40(%rbp), %xmm0
	subsd	%xmm1, %xmm0
	movsd	%xmm0, -40(%rbp)
	movsd	-48(%rbp), %xmm1
	movsd	-56(%rbp), %xmm0
	movq	-64(%rbp), %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	angle
	movq	%xmm0, %rax
	movq	%rax, -24(%rbp)
	movsd	-24(%rbp), %xmm0
	movsd	.LC1(%rip), %xmm2
	movapd	%xmm0, %xmm1
	divsd	%xmm2, %xmm1
	movsd	-88(%rbp), %xmm0
	mulsd	%xmm0, %xmm1
	movsd	-88(%rbp), %xmm0
	mulsd	%xmm0, %xmm1
	movsd	-40(%rbp), %xmm0
	subsd	%xmm1, %xmm0
	movsd	%xmm0, -40(%rbp)
	movsd	-64(%rbp), %xmm1
	movsd	-48(%rbp), %xmm0
	movq	-56(%rbp), %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	angle
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	movsd	-16(%rbp), %xmm0
	movsd	.LC1(%rip), %xmm2
	movapd	%xmm0, %xmm1
	divsd	%xmm2, %xmm1
	movsd	-80(%rbp), %xmm0
	mulsd	%xmm0, %xmm1
	movsd	-80(%rbp), %xmm0
	mulsd	%xmm0, %xmm1
	movsd	-40(%rbp), %xmm0
	subsd	%xmm1, %xmm0
	movsd	%xmm0, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$1, -72(%rbp)
	jmp	.L13
.L7:
	cmpl	$0, -100(%rbp)
	jle	.L14
	movq	$3, -72(%rbp)
	jmp	.L13
.L14:
	movq	$5, -72(%rbp)
	jmp	.L13
.L9:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L18
	jmp	.L19
.L11:
	leaq	-104(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$1, -72(%rbp)
	jmp	.L13
.L20:
	nop
.L13:
	jmp	.L17
.L19:
	call	__stack_chk_fail@PLT
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.globl	angle
	.type	angle, @function
angle:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movsd	%xmm0, -24(%rbp)
	movsd	%xmm1, -32(%rbp)
	movsd	%xmm2, -40(%rbp)
	movq	$1, -8(%rbp)
.L26:
	cmpq	$0, -8(%rbp)
	je	.L22
	cmpq	$1, -8(%rbp)
	jne	.L28
	movsd	-24(%rbp), %xmm0
	movapd	%xmm0, %xmm1
	mulsd	%xmm0, %xmm1
	movsd	-32(%rbp), %xmm0
	mulsd	%xmm0, %xmm0
	addsd	%xmm0, %xmm1
	movsd	-40(%rbp), %xmm0
	mulsd	%xmm0, %xmm0
	subsd	%xmm0, %xmm1
	movsd	-24(%rbp), %xmm0
	addsd	%xmm0, %xmm0
	mulsd	-32(%rbp), %xmm0
	divsd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	call	acos@PLT
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L24
.L22:
	movsd	-16(%rbp), %xmm0
	jmp	.L27
.L28:
	nop
.L24:
	jmp	.L26
.L27:
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	angle, .-angle
	.globl	heron
	.type	heron, @function
heron:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movsd	%xmm0, -40(%rbp)
	movsd	%xmm1, -48(%rbp)
	movsd	%xmm2, -56(%rbp)
	movq	$1, -16(%rbp)
.L35:
	cmpq	$2, -16(%rbp)
	je	.L30
	cmpq	$2, -16(%rbp)
	ja	.L37
	cmpq	$0, -16(%rbp)
	je	.L32
	cmpq	$1, -16(%rbp)
	jne	.L37
	movq	$0, -16(%rbp)
	jmp	.L33
.L32:
	movsd	-40(%rbp), %xmm0
	addsd	-48(%rbp), %xmm0
	addsd	-56(%rbp), %xmm0
	movsd	.LC1(%rip), %xmm1
	divsd	%xmm1, %xmm0
	movsd	%xmm0, -8(%rbp)
	movsd	-8(%rbp), %xmm0
	subsd	-40(%rbp), %xmm0
	movapd	%xmm0, %xmm1
	mulsd	-8(%rbp), %xmm1
	movsd	-8(%rbp), %xmm0
	subsd	-48(%rbp), %xmm0
	mulsd	%xmm0, %xmm1
	movsd	-8(%rbp), %xmm0
	subsd	-56(%rbp), %xmm0
	mulsd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	call	sqrt@PLT
	movq	%xmm0, %rax
	movq	%rax, -24(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L33
.L30:
	movsd	-24(%rbp), %xmm0
	jmp	.L36
.L37:
	nop
.L33:
	jmp	.L35
.L36:
	movq	%xmm0, %rax
	movq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	heron, .-heron
	.section	.rodata
	.align 8
.LC1:
	.long	0
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
