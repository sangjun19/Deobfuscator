	.file	"dhawansolanki_c_1_flatten.c"
	.text
	.globl	_TIG_IZ_zmwG_argv
	.bss
	.align 8
	.type	_TIG_IZ_zmwG_argv, @object
	.size	_TIG_IZ_zmwG_argv, 8
_TIG_IZ_zmwG_argv:
	.zero	8
	.globl	_TIG_IZ_zmwG_argc
	.align 4
	.type	_TIG_IZ_zmwG_argc, @object
	.size	_TIG_IZ_zmwG_argc, 4
_TIG_IZ_zmwG_argc:
	.zero	4
	.globl	_TIG_IZ_zmwG_envp
	.align 8
	.type	_TIG_IZ_zmwG_envp, @object
	.size	_TIG_IZ_zmwG_envp, 8
_TIG_IZ_zmwG_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"The Product of %f & %f is %f"
	.align 8
.LC1:
	.string	"The Difference of %f & %f is %f"
.LC2:
	.string	"The Quotient of %f & %f is %f"
.LC3:
	.string	"The Sum of %f & %f is %f"
.LC4:
	.string	"Enter the Operation : "
.LC5:
	.string	"%f%c%f"
.LC7:
	.string	"Invalid Division."
.LC8:
	.string	"Something Went Wrong..."
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
	movq	$0, _TIG_IZ_zmwG_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_zmwG_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_zmwG_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 127 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-zmwG--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_zmwG_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_zmwG_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_zmwG_envp(%rip)
	nop
	movq	$13, -16(%rbp)
.L30:
	cmpq	$18, -16(%rbp)
	ja	.L34
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
	.long	.L18-.L8
	.long	.L34-.L8
	.long	.L17-.L8
	.long	.L34-.L8
	.long	.L34-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L35-.L8
	.long	.L13-.L8
	.long	.L34-.L8
	.long	.L34-.L8
	.long	.L12-.L8
	.long	.L34-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L34-.L8
	.long	.L9-.L8
	.long	.L34-.L8
	.long	.L7-.L8
	.text
.L7:
	movss	-28(%rbp), %xmm1
	movss	-24(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm1, %xmm1
	cvtss2sd	-20(%rbp), %xmm1
	movss	-24(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movss	-28(%rbp), %xmm2
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
	movq	$7, -16(%rbp)
	jmp	.L19
.L10:
	movss	-28(%rbp), %xmm0
	movss	-24(%rbp), %xmm1
	subss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm1, %xmm1
	cvtss2sd	-20(%rbp), %xmm1
	movss	-24(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movss	-28(%rbp), %xmm2
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
	movq	$7, -16(%rbp)
	jmp	.L19
.L13:
	movss	-28(%rbp), %xmm0
	movss	-24(%rbp), %xmm1
	divss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm1, %xmm1
	cvtss2sd	-20(%rbp), %xmm1
	movss	-24(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movss	-28(%rbp), %xmm2
	pxor	%xmm5, %xmm5
	cvtss2sd	%xmm2, %xmm5
	movq	%xmm5, %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L19
.L9:
	movss	-28(%rbp), %xmm1
	movss	-24(%rbp), %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	pxor	%xmm1, %xmm1
	cvtss2sd	-20(%rbp), %xmm1
	movss	-24(%rbp), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movss	-28(%rbp), %xmm2
	pxor	%xmm6, %xmm6
	cvtss2sd	%xmm2, %xmm6
	movq	%xmm6, %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L19
.L12:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rcx
	leaq	-29(%rbp), %rdx
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$5, -16(%rbp)
	jmp	.L19
.L11:
	movq	$11, -16(%rbp)
	jmp	.L19
.L15:
	movss	-24(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jp	.L20
	pxor	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jne	.L20
	movq	$0, -16(%rbp)
	jmp	.L19
.L20:
	movq	$8, -16(%rbp)
	jmp	.L19
.L16:
	movzbl	-29(%rbp), %eax
	movsbl	%al, %eax
	cmpl	$47, %eax
	je	.L23
	cmpl	$47, %eax
	jg	.L24
	cmpl	$45, %eax
	je	.L25
	cmpl	$45, %eax
	jg	.L24
	cmpl	$42, %eax
	je	.L26
	cmpl	$43, %eax
	je	.L27
	jmp	.L24
.L23:
	movq	$6, -16(%rbp)
	jmp	.L28
.L26:
	movq	$18, -16(%rbp)
	jmp	.L28
.L25:
	movq	$14, -16(%rbp)
	jmp	.L28
.L27:
	movq	$16, -16(%rbp)
	jmp	.L28
.L24:
	movq	$2, -16(%rbp)
	nop
.L28:
	jmp	.L19
.L18:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L19
.L17:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -16(%rbp)
	jmp	.L19
.L34:
	nop
.L19:
	jmp	.L30
.L35:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L32
	call	__stack_chk_fail@PLT
.L32:
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
