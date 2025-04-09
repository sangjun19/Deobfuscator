	.file	"Evelyn-dr_Programacion-para-Mecatronicos_main_flatten.c"
	.text
	.globl	_TIG_IZ_4Ypo_argv
	.bss
	.align 8
	.type	_TIG_IZ_4Ypo_argv, @object
	.size	_TIG_IZ_4Ypo_argv, 8
_TIG_IZ_4Ypo_argv:
	.zero	8
	.globl	_TIG_IZ_4Ypo_envp
	.align 8
	.type	_TIG_IZ_4Ypo_envp, @object
	.size	_TIG_IZ_4Ypo_envp, 8
_TIG_IZ_4Ypo_envp:
	.zero	8
	.globl	_TIG_IZ_4Ypo_argc
	.align 4
	.type	_TIG_IZ_4Ypo_argc, @object
	.size	_TIG_IZ_4Ypo_argc, 4
_TIG_IZ_4Ypo_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"\nLa clave del tratamiento es incorrcto "
	.align 8
.LC4:
	.string	"Ingrese tipo de tratamiento, edad y dias: "
.LC5:
	.string	"%d %d %d"
	.align 8
.LC6:
	.string	"\nClave tratamineto: %d/t Dias: %d/t Costo total: 8.2f"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_4Ypo_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_4Ypo_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_4Ypo_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 106 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-4Ypo--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_4Ypo_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_4Ypo_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_4Ypo_envp(%rip)
	nop
	movq	$6, -16(%rbp)
.L39:
	cmpq	$17, -16(%rbp)
	ja	.L43
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
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L43-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L43-.L8
	.long	.L9-.L8
	.long	.L44-.L8
	.text
.L19:
	movl	-28(%rbp), %eax
	cmpl	$59, %eax
	jle	.L24
	movq	$13, -16(%rbp)
	jmp	.L26
.L24:
	movq	$14, -16(%rbp)
	jmp	.L26
.L10:
	movl	-28(%rbp), %eax
	cmpl	$25, %eax
	jg	.L27
	movq	$3, -16(%rbp)
	jmp	.L26
.L27:
	movq	$2, -16(%rbp)
	jmp	.L26
.L15:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$17, -16(%rbp)
	jmp	.L26
.L22:
	movl	-24(%rbp), %eax
	imull	$1150, %eax, %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L26
.L20:
	pxor	%xmm1, %xmm1
	cvtss2sd	-20(%rbp), %xmm1
	movsd	.LC1(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L26
.L9:
	movl	-24(%rbp), %eax
	imull	$1950, %eax, %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L26
.L12:
	movq	$9, -16(%rbp)
	jmp	.L26
.L14:
	movss	.LC2(%rip), %xmm0
	ucomiss	-20(%rbp), %xmm0
	jp	.L42
	movss	.LC2(%rip), %xmm0
	ucomiss	-20(%rbp), %xmm0
	je	.L29
.L42:
	movq	$4, -16(%rbp)
	jmp	.L26
.L29:
	movq	$8, -16(%rbp)
	jmp	.L26
.L11:
	movss	-20(%rbp), %xmm1
	movss	.LC3(%rip), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L26
.L17:
	movq	$10, -16(%rbp)
	jmp	.L26
.L18:
	movl	-24(%rbp), %eax
	imull	$2800, %eax, %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L26
.L13:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rcx
	leaq	-28(%rbp), %rdx
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L23:
	movl	-32(%rbp), %eax
	cmpl	$4, %eax
	je	.L33
	cmpl	$4, %eax
	jg	.L34
	cmpl	$3, %eax
	je	.L35
	cmpl	$3, %eax
	jg	.L34
	cmpl	$1, %eax
	je	.L36
	cmpl	$2, %eax
	je	.L37
	jmp	.L34
.L33:
	movq	$1, -16(%rbp)
	jmp	.L38
.L35:
	movq	$7, -16(%rbp)
	jmp	.L38
.L37:
	movq	$16, -16(%rbp)
	jmp	.L38
.L36:
	movq	$5, -16(%rbp)
	jmp	.L38
.L34:
	movq	$11, -16(%rbp)
	nop
.L38:
	jmp	.L26
.L16:
	movl	-24(%rbp), %eax
	imull	$2500, %eax, %eax
	pxor	%xmm0, %xmm0
	cvtsi2ssl	%eax, %xmm0
	movss	%xmm0, -20(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L26
.L21:
	pxor	%xmm2, %xmm2
	cvtss2sd	-20(%rbp), %xmm2
	movq	%xmm2, %rcx
	movl	-24(%rbp), %edx
	movl	-32(%rbp), %eax
	movq	%rcx, %xmm0
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$17, -16(%rbp)
	jmp	.L26
.L43:
	nop
.L26:
	jmp	.L39
.L44:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L41
	call	__stack_chk_fail@PLT
.L41:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC1:
	.long	858993459
	.long	1072378675
	.align 4
.LC2:
	.long	-1082130432
	.align 4
.LC3:
	.long	1061158912
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
