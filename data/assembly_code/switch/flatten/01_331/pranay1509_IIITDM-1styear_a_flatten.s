	.file	"pranay1509_IIITDM-1styear_a_flatten.c"
	.text
	.globl	_TIG_IZ_yfLZ_argv
	.bss
	.align 8
	.type	_TIG_IZ_yfLZ_argv, @object
	.size	_TIG_IZ_yfLZ_argv, 8
_TIG_IZ_yfLZ_argv:
	.zero	8
	.globl	_TIG_IZ_yfLZ_argc
	.align 4
	.type	_TIG_IZ_yfLZ_argc, @object
	.size	_TIG_IZ_yfLZ_argc, 4
_TIG_IZ_yfLZ_argc:
	.zero	4
	.globl	_TIG_IZ_yfLZ_envp
	.align 8
	.type	_TIG_IZ_yfLZ_envp, @object
	.size	_TIG_IZ_yfLZ_envp, 8
_TIG_IZ_yfLZ_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%d"
.LC1:
	.string	"%d %d"
.LC2:
	.string	"%f"
	.align 8
.LC3:
	.string	"enter values between 1 and 4 only"
.LC5:
	.string	"enter a and b"
.LC6:
	.string	"enter the option:"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_yfLZ_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_yfLZ_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_yfLZ_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-yfLZ--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_yfLZ_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_yfLZ_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_yfLZ_envp(%rip)
	nop
	movq	$4, -16(%rbp)
.L38:
	cmpq	$18, -16(%rbp)
	ja	.L41
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
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L41-.L8
	.long	.L41-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L41-.L8
	.long	.L12-.L8
	.long	.L41-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L41-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	movl	-28(%rbp), %eax
	cmpl	$4, %eax
	je	.L22
	cmpl	$4, %eax
	jg	.L23
	cmpl	$3, %eax
	je	.L24
	cmpl	$3, %eax
	jg	.L23
	cmpl	$1, %eax
	je	.L25
	cmpl	$2, %eax
	je	.L26
	jmp	.L23
.L22:
	movq	$12, -16(%rbp)
	jmp	.L27
.L24:
	movq	$14, -16(%rbp)
	jmp	.L27
.L26:
	movq	$0, -16(%rbp)
	jmp	.L27
.L25:
	movq	$15, -16(%rbp)
	jmp	.L27
.L23:
	movq	$3, -16(%rbp)
	nop
.L27:
	jmp	.L28
.L17:
	movq	$10, -16(%rbp)
	jmp	.L28
.L11:
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L28
.L10:
	movl	-32(%rbp), %edx
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L28
.L12:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L39
	jmp	.L40
.L15:
	pxor	%xmm2, %xmm2
	cvtss2sd	-20(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L28
.L20:
	movl	-32(%rbp), %eax
	cltd
	shrl	$31, %edx
	addl	%edx, %eax
	andl	$1, %eax
	subl	%edx, %eax
	cmpl	$1, %eax
	jne	.L30
	movq	$8, -16(%rbp)
	jmp	.L28
.L30:
	movq	$2, -16(%rbp)
	jmp	.L28
.L18:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L28
.L14:
	movl	-32(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	je	.L32
	movq	$17, -16(%rbp)
	jmp	.L28
.L32:
	movq	$12, -16(%rbp)
	jmp	.L28
.L9:
	pxor	%xmm1, %xmm1
	cvtss2sd	-20(%rbp), %xmm1
	movsd	.LC4(%rip), %xmm0
	addsd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L28
.L13:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-32(%rbp), %rdx
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-36(%rbp), %edx
	movl	-32(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movss	%xmm0, -20(%rbp)
	movl	-36(%rbp), %eax
	movl	-32(%rbp), %ecx
	cltd
	idivl	%ecx
	movl	%edx, -24(%rbp)
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$18, -16(%rbp)
	jmp	.L28
.L21:
	movl	-36(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	je	.L34
	movq	$1, -16(%rbp)
	jmp	.L28
.L34:
	movq	$2, -16(%rbp)
	jmp	.L28
.L16:
	pxor	%xmm1, %xmm1
	cvtss2sd	-20(%rbp), %xmm1
	movsd	.LC4(%rip), %xmm0
	addsd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L28
.L19:
	movl	-36(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	je	.L36
	movq	$7, -16(%rbp)
	jmp	.L28
.L36:
	movq	$9, -16(%rbp)
	jmp	.L28
.L41:
	nop
.L28:
	jmp	.L38
.L40:
	call	__stack_chk_fail@PLT
.L39:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC4:
	.long	0
	.long	1071644672
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
