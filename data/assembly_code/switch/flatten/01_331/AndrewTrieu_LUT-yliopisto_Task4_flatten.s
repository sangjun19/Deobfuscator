	.file	"AndrewTrieu_LUT-yliopisto_Task4_flatten.c"
	.text
	.globl	_TIG_IZ_XCxE_envp
	.bss
	.align 8
	.type	_TIG_IZ_XCxE_envp, @object
	.size	_TIG_IZ_XCxE_envp, 8
_TIG_IZ_XCxE_envp:
	.zero	8
	.globl	_TIG_IZ_XCxE_argc
	.align 4
	.type	_TIG_IZ_XCxE_argc, @object
	.size	_TIG_IZ_XCxE_argc, 4
_TIG_IZ_XCxE_argc:
	.zero	4
	.globl	_TIG_IZ_XCxE_argv
	.align 8
	.type	_TIG_IZ_XCxE_argv, @object
	.size	_TIG_IZ_XCxE_argv, 8
_TIG_IZ_XCxE_argv:
	.zero	8
	.section	.rodata
.LC1:
	.string	"Pi / %.3f = %.3f."
.LC2:
	.string	"Unknown selection."
.LC3:
	.string	"Pi * %.3f = %.3f."
	.align 8
.LC4:
	.string	"Enter a floating-point number:"
.LC5:
	.string	"%f"
.LC6:
	.string	"MENU"
.LC7:
	.string	"1: Multiply Pi by %.3f.\n"
.LC8:
	.string	"2: Divide Pi by %.3f.\n"
.LC9:
	.string	"%d"
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
	movq	$0, _TIG_IZ_XCxE_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_XCxE_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_XCxE_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 111 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-XCxE--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_XCxE_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_XCxE_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_XCxE_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L20:
	cmpq	$8, -16(%rbp)
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
	.long	.L23-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L23-.L8
	.long	.L7-.L8
	.text
.L11:
	movss	-20(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movsd	.LC0(%rip), %xmm0
	divsd	%xmm1, %xmm0
	movss	-20(%rbp), %xmm1
	pxor	%xmm2, %xmm2
	cvtss2sd	%xmm1, %xmm2
	movq	%xmm2, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$2, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L15
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L21
	jmp	.L22
.L14:
	movq	$5, -16(%rbp)
	jmp	.L15
.L12:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L15
.L9:
	movss	-20(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movsd	.LC0(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	movss	-20(%rbp), %xmm1
	pxor	%xmm3, %xmm3
	cvtss2sd	%xmm1, %xmm3
	movq	%xmm3, %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$2, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L15
.L10:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movss	-20(%rbp), %xmm0
	pxor	%xmm4, %xmm4
	cvtss2sd	%xmm0, %xmm4
	movq	%xmm4, %rax
	movq	%rax, %xmm0
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movss	-20(%rbp), %xmm0
	pxor	%xmm5, %xmm5
	cvtss2sd	%xmm0, %xmm5
	movq	%xmm5, %rax
	movq	%rax, %xmm0
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -16(%rbp)
	jmp	.L15
.L13:
	movl	-24(%rbp), %eax
	cmpl	$1, %eax
	je	.L17
	cmpl	$2, %eax
	jne	.L18
	movq	$4, -16(%rbp)
	jmp	.L19
.L17:
	movq	$6, -16(%rbp)
	jmp	.L19
.L18:
	movq	$3, -16(%rbp)
	nop
.L19:
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
.LC0:
	.long	-1683627180
	.long	1074340036
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
