	.file	"GANESH-ICMC_Apresentacoes_ifs_switch_flatten.c"
	.text
	.globl	_TIG_IZ_ZOoa_envp
	.bss
	.align 8
	.type	_TIG_IZ_ZOoa_envp, @object
	.size	_TIG_IZ_ZOoa_envp, 8
_TIG_IZ_ZOoa_envp:
	.zero	8
	.globl	_TIG_IZ_ZOoa_argc
	.align 4
	.type	_TIG_IZ_ZOoa_argc, @object
	.size	_TIG_IZ_ZOoa_argc, 4
_TIG_IZ_ZOoa_argc:
	.zero	4
	.globl	_TIG_IZ_ZOoa_argv
	.align 8
	.type	_TIG_IZ_ZOoa_argv, @object
	.size	_TIG_IZ_ZOoa_argv, 8
_TIG_IZ_ZOoa_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Apos o switch:"
.LC1:
	.string	"%d %f\n"
.LC5:
	.string	"Apos o IF:"
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
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_ZOoa_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_ZOoa_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_ZOoa_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 125 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ZOoa--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_ZOoa_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_ZOoa_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_ZOoa_envp(%rip)
	nop
	movq	$12, -8(%rbp)
.L29:
	cmpq	$16, -8(%rbp)
	ja	.L31
	movq	-8(%rbp), %rax
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
	.long	.L31-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L31-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L31-.L8
	.long	.L31-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L31-.L8
	.long	.L7-.L8
	.text
.L9:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	pxor	%xmm1, %xmm1
	cvtss2sd	-12(%rbp), %xmm1
	movq	%xmm1, %rdx
	movl	-16(%rbp), %eax
	movq	%rdx, %xmm0
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$6, -8(%rbp)
	jmp	.L20
.L11:
	movq	$10, -8(%rbp)
	jmp	.L20
.L19:
	movss	.LC2(%rip), %xmm0
	movss	%xmm0, -12(%rbp)
	movss	.LC3(%rip), %xmm0
	movss	%xmm0, -12(%rbp)
	movss	.LC4(%rip), %xmm0
	movss	%xmm0, -12(%rbp)
	movq	$11, -8(%rbp)
	jmp	.L20
.L17:
	movss	.LC3(%rip), %xmm0
	movss	%xmm0, -12(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L20
.L7:
	movss	.LC2(%rip), %xmm0
	movss	%xmm0, -12(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L20
.L12:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	pxor	%xmm2, %xmm2
	cvtss2sd	-12(%rbp), %xmm2
	movq	%xmm2, %rdx
	movl	-16(%rbp), %eax
	movq	%rdx, %xmm0
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$2, -8(%rbp)
	jmp	.L20
.L14:
	movss	.LC6(%rip), %xmm0
	movss	%xmm0, -12(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L20
.L10:
	movss	.LC4(%rip), %xmm0
	movss	%xmm0, -12(%rbp)
	movq	$14, -8(%rbp)
	jmp	.L20
.L15:
	cmpl	$1, -16(%rbp)
	jne	.L21
	movq	$1, -8(%rbp)
	jmp	.L20
.L21:
	movq	$11, -8(%rbp)
	jmp	.L20
.L16:
	cmpl	$3, -16(%rbp)
	je	.L23
	cmpl	$3, -16(%rbp)
	jg	.L24
	cmpl	$1, -16(%rbp)
	je	.L25
	cmpl	$2, -16(%rbp)
	je	.L26
	jmp	.L24
.L23:
	movq	$13, -8(%rbp)
	jmp	.L27
.L26:
	movq	$3, -8(%rbp)
	jmp	.L27
.L25:
	movq	$16, -8(%rbp)
	jmp	.L27
.L24:
	movq	$9, -8(%rbp)
	nop
.L27:
	jmp	.L20
.L13:
	movl	$1, -16(%rbp)
	pxor	%xmm0, %xmm0
	movss	%xmm0, -12(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L20
.L18:
	movl	$0, %eax
	jmp	.L30
.L31:
	nop
.L20:
	jmp	.L29
.L30:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC2:
	.long	1092616192
	.align 4
.LC3:
	.long	1101004800
	.align 4
.LC4:
	.long	1106247680
	.align 4
.LC6:
	.long	-1082130432
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
