	.file	"hirebbec_KAI_main_flatten.c"
	.text
	.globl	_TIG_IZ_vZGm_argc
	.bss
	.align 4
	.type	_TIG_IZ_vZGm_argc, @object
	.size	_TIG_IZ_vZGm_argc, 4
_TIG_IZ_vZGm_argc:
	.zero	4
	.globl	_TIG_IZ_vZGm_envp
	.align 8
	.type	_TIG_IZ_vZGm_envp, @object
	.size	_TIG_IZ_vZGm_envp, 8
_TIG_IZ_vZGm_envp:
	.zero	8
	.globl	_TIG_IZ_vZGm_argv
	.align 8
	.type	_TIG_IZ_vZGm_argv, @object
	.size	_TIG_IZ_vZGm_argv, 8
_TIG_IZ_vZGm_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%.4f %.4f"
.LC1:
	.string	"NO"
.LC2:
	.string	"%.4f"
.LC3:
	.string	"%d %d %d"
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
	movq	$0, _TIG_IZ_vZGm_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_vZGm_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_vZGm_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 90 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-vZGm--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_vZGm_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_vZGm_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_vZGm_envp(%rip)
	nop
	movq	$3, -32(%rbp)
.L22:
	cmpq	$8, -32(%rbp)
	ja	.L25
	movq	-32(%rbp), %rax
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
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L25-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L11:
	pxor	%xmm3, %xmm3
	cvtsi2sdl	-36(%rbp), %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	call	sqrt@PLT
	movq	%xmm0, %rax
	movq	%rax, -24(%rbp)
	pxor	%xmm4, %xmm4
	cvtsi2sdl	-36(%rbp), %xmm4
	movq	%xmm4, %rax
	movq	%rax, %xmm0
	call	sqrt@PLT
	movq	%xmm0, %rax
	movq	%rax, -16(%rbp)
	movl	-44(%rbp), %eax
	negl	%eax
	pxor	%xmm0, %xmm0
	cvtsi2sdl	%eax, %xmm0
	addsd	-24(%rbp), %xmm0
	movl	-48(%rbp), %eax
	addl	%eax, %eax
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%eax, %xmm1
	movapd	%xmm0, %xmm2
	divsd	%xmm1, %xmm2
	movl	-44(%rbp), %eax
	negl	%eax
	pxor	%xmm0, %xmm0
	cvtsi2sdl	%eax, %xmm0
	subsd	-16(%rbp), %xmm0
	movl	-48(%rbp), %eax
	addl	%eax, %eax
	pxor	%xmm1, %xmm1
	cvtsi2sdl	%eax, %xmm1
	divsd	%xmm1, %xmm0
	movq	%xmm0, %rax
	movapd	%xmm2, %xmm1
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$2, %eax
	call	printf@PLT
	movq	$8, -32(%rbp)
	jmp	.L16
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L23
	jmp	.L24
.L14:
	cmpl	$0, -36(%rbp)
	jle	.L18
	movq	$4, -32(%rbp)
	jmp	.L16
.L18:
	movq	$7, -32(%rbp)
	jmp	.L16
.L12:
	movq	$2, -32(%rbp)
	jmp	.L16
.L10:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -32(%rbp)
	jmp	.L16
.L15:
	movl	-44(%rbp), %eax
	negl	%eax
	movl	-48(%rbp), %edx
	leal	(%rdx,%rdx), %esi
	cltd
	idivl	%esi
	pxor	%xmm5, %xmm5
	cvtsi2sdl	%eax, %xmm5
	movq	%xmm5, %rax
	movq	%rax, %xmm0
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$8, -32(%rbp)
	jmp	.L16
.L9:
	cmpl	$0, -36(%rbp)
	jne	.L20
	movq	$0, -32(%rbp)
	jmp	.L16
.L20:
	movq	$5, -32(%rbp)
	jmp	.L16
.L13:
	leaq	-40(%rbp), %rcx
	leaq	-44(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-44(%rbp), %edx
	movl	-44(%rbp), %eax
	imull	%edx, %eax
	movl	-48(%rbp), %ecx
	movl	-40(%rbp), %edx
	imull	%ecx, %edx
	sall	$2, %edx
	subl	%edx, %eax
	movl	%eax, -36(%rbp)
	movq	$1, -32(%rbp)
	jmp	.L16
.L25:
	nop
.L16:
	jmp	.L22
.L24:
	call	__stack_chk_fail@PLT
.L23:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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
