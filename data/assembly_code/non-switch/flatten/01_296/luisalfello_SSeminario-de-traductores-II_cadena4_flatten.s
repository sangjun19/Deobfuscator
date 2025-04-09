	.file	"luisalfello_SSeminario-de-traductores-II_cadena4_flatten.c"
	.text
	.globl	_TIG_IZ_ENEb_envp
	.bss
	.align 8
	.type	_TIG_IZ_ENEb_envp, @object
	.size	_TIG_IZ_ENEb_envp, 8
_TIG_IZ_ENEb_envp:
	.zero	8
	.globl	_TIG_IZ_ENEb_argc
	.align 4
	.type	_TIG_IZ_ENEb_argc, @object
	.size	_TIG_IZ_ENEb_argc, 4
_TIG_IZ_ENEb_argc:
	.zero	4
	.globl	_TIG_IZ_ENEb_argv
	.align 8
	.type	_TIG_IZ_ENEb_argv, @object
	.size	_TIG_IZ_ENEb_argv, 8
_TIG_IZ_ENEb_argv:
	.zero	8
	.globl	a
	.align 4
	.type	a, @object
	.size	a, 4
a:
	.zero	4
	.text
	.globl	suma
	.type	suma, @function
suma:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L4:
	cmpq	$0, -8(%rbp)
	jne	.L7
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	addl	%edx, %eax
	jmp	.L6
.L7:
	nop
	jmp	.L4
.L6:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	suma, .-suma
	.globl	main
	.type	main, @function
main:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$56, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movl	$0, a(%rip)
	nop
.L9:
	movq	$0, _TIG_IZ_ENEb_envp(%rip)
	nop
.L10:
	movq	$0, _TIG_IZ_ENEb_argv(%rip)
	nop
.L11:
	movl	$0, _TIG_IZ_ENEb_argc(%rip)
	nop
	nop
.L12:
.L13:
#APP
# 177 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ENEb--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_ENEb_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_ENEb_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_ENEb_envp(%rip)
	nop
	movq	$1, -8(%rbp)
.L19:
	cmpq	$2, -8(%rbp)
	je	.L14
	cmpq	$2, -8(%rbp)
	ja	.L21
	cmpq	$0, -8(%rbp)
	je	.L16
	cmpq	$1, -8(%rbp)
	jne	.L21
	movq	$0, -8(%rbp)
	jmp	.L17
.L16:
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-20(%rbp), %xmm0
	addss	-16(%rbp), %xmm0
	cvttss2sil	%xmm0, %eax
	movl	%eax, -12(%rbp)
	movl	$9, %esi
	movl	$8, %edi
	call	suma
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L17
.L14:
	movl	$0, %eax
	jmp	.L20
.L21:
	nop
.L17:
	jmp	.L19
.L20:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
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
