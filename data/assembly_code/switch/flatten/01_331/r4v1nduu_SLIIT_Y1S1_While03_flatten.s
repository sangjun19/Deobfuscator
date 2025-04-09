	.file	"r4v1nduu_SLIIT_Y1S1_While03_flatten.c"
	.text
	.globl	_TIG_IZ_Ucq9_argc
	.bss
	.align 4
	.type	_TIG_IZ_Ucq9_argc, @object
	.size	_TIG_IZ_Ucq9_argc, 4
_TIG_IZ_Ucq9_argc:
	.zero	4
	.globl	_TIG_IZ_Ucq9_argv
	.align 8
	.type	_TIG_IZ_Ucq9_argv, @object
	.size	_TIG_IZ_Ucq9_argv, 8
_TIG_IZ_Ucq9_argv:
	.zero	8
	.globl	_TIG_IZ_Ucq9_envp
	.align 8
	.type	_TIG_IZ_Ucq9_envp, @object
	.size	_TIG_IZ_Ucq9_envp, 8
_TIG_IZ_Ucq9_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"\nTotal Price to Pay : %.2lf"
.LC2:
	.string	"Enter quantity : "
.LC3:
	.string	"%d"
.LC4:
	.string	"\nEnter item no. : "
.LC8:
	.string	"Enter item no. : "
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
	movq	$0, _TIG_IZ_Ucq9_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Ucq9_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Ucq9_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 106 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Ucq9--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_Ucq9_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_Ucq9_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_Ucq9_envp(%rip)
	nop
	movq	$0, -16(%rbp)
.L37:
	cmpq	$23, -16(%rbp)
	ja	.L40
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
	.long	.L40-.L8
	.long	.L40-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L40-.L8
	.long	.L14-.L8
	.long	.L40-.L8
	.long	.L40-.L8
	.long	.L13-.L8
	.long	.L40-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L40-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L40-.L8
	.long	.L40-.L8
	.long	.L40-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	-40(%rbp), %eax
	cmpl	$3, %eax
	jne	.L22
	movq	$3, -16(%rbp)
	jmp	.L24
.L22:
	movq	$2, -16(%rbp)
	jmp	.L24
.L17:
	movq	-24(%rbp), %rax
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$23, -16(%rbp)
	jmp	.L24
.L12:
	movl	-40(%rbp), %eax
	cmpl	$2, %eax
	jne	.L25
	movq	$3, -16(%rbp)
	jmp	.L24
.L25:
	movq	$18, -16(%rbp)
	jmp	.L24
.L15:
	movl	-40(%rbp), %eax
	cmpl	$1, %eax
	jne	.L27
	movq	$3, -16(%rbp)
	jmp	.L24
.L27:
	movq	$15, -16(%rbp)
	jmp	.L24
.L20:
	movsd	.LC1(%rip), %xmm0
	movsd	%xmm0, -32(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L24
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L38
	jmp	.L39
.L18:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-36(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$13, -16(%rbp)
	jmp	.L24
.L11:
	movl	-36(%rbp), %eax
	pxor	%xmm0, %xmm0
	cvtsi2sdl	%eax, %xmm0
	mulsd	-32(%rbp), %xmm0
	movsd	-24(%rbp), %xmm1
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -24(%rbp)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -16(%rbp)
	jmp	.L24
.L13:
	movl	-40(%rbp), %eax
	cmpl	$3, %eax
	je	.L30
	cmpl	$3, %eax
	jg	.L31
	cmpl	$1, %eax
	je	.L32
	cmpl	$2, %eax
	je	.L33
	jmp	.L31
.L30:
	movq	$19, -16(%rbp)
	jmp	.L34
.L33:
	movq	$1, -16(%rbp)
	jmp	.L34
.L32:
	movq	$10, -16(%rbp)
	jmp	.L34
.L31:
	movq	$16, -16(%rbp)
	nop
.L34:
	jmp	.L24
.L9:
	movsd	.LC5(%rip), %xmm0
	movsd	%xmm0, -32(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L24
.L14:
	movsd	.LC6(%rip), %xmm0
	movsd	%xmm0, -32(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L24
.L21:
	movq	$7, -16(%rbp)
	jmp	.L24
.L16:
	pxor	%xmm0, %xmm0
	movsd	%xmm0, -24(%rbp)
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -16(%rbp)
	jmp	.L24
.L19:
	movl	-40(%rbp), %eax
	cmpl	$-1, %eax
	je	.L35
	movq	$8, -16(%rbp)
	jmp	.L24
.L35:
	movq	$4, -16(%rbp)
	jmp	.L24
.L40:
	nop
.L24:
	jmp	.L37
.L39:
	call	__stack_chk_fail@PLT
.L38:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC1:
	.long	0
	.long	1080176640
	.align 8
.LC5:
	.long	0
	.long	1082157056
	.align 8
.LC6:
	.long	0
	.long	1081263104
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
